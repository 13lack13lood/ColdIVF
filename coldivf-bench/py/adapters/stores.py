"""The files of a system that writes to its own index each time it opens or closes, kept as the build left them.

qdrant-edge rewrites its config and segment flags at every load, and AsterVec's RocksDB store writes a new manifest,
options file and log at every open. Opened in place, such an index changes with every trial. No two trials would open
the same bytes, and the index would stop matching what its build recorded. So a build seals its store. It runs one
session on a copy, notes every file the session changes or removes, and keeps a clone of each in stash/. After every
session, `restore` deletes what the session made and puts the stashed files back.

A clone on the Apple File System (APFS) shares the original's disk blocks and keeps its size and change time. So a
restore copies nothing, and between sessions the index keeps the identity its build recorded. The restored files are
new files, which a cold trial drops through new mappings as it does any file it hasn't seen before. A process of the
disk-resident steady state (C1) keeps its session open for its whole cell, so its index differs from the build until
that process ends.

Only one process opens the store in place at a time, the one that holds the lock on store.lock. A second process
can't share the store, eg. RocksDB locks its folder, and its restore would undo the first one's session. So it opens a
private copy under <index>.sessions/ instead. The copy hard-links the bulk data a search reads, so both processes read
the same cached pages, and clones the rest, eg. the lock files that two processes can't share.
"""

from __future__ import annotations

import ctypes
import fcntl
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from adapters import graphs
from adapters.base import Adapter, ServeContext, Unsupported

RECORD = "store.json"
STASH = "stash"
LEARN = "learn"
CLAIM = "store.lock"

_libc = ctypes.CDLL(None, use_errno=True) if sys.platform == "darwin" else None

if _libc is not None:
    _libc.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    _libc.clonefile.restype = ctypes.c_int


class StoreChanged(RuntimeError):
    """Raised when a session changed a file that has no stashed clone, since no session changed it when the build
    sealed the store."""


def clone(src: Path, dst: Path) -> None:
    """Copies the file `src` to `dst` as an APFS clone, which keeps its size and change time. Where clones don't work,
    it makes a plain copy with the same times."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if _libc is not None and _libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) == 0:
        return

    shutil.copy2(src, dst)


@dataclass
class Layout:
    """The files and folders under one folder, by path relative to it. Each file has its inode, size and change time in
    nanoseconds."""

    files: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    dirs: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, root: Path) -> Layout:
        out = cls()

        for path in sorted(Path(root).rglob("*")):
            rel = path.relative_to(root).as_posix()

            if path.is_symlink():
                continue

            if path.is_dir():
                out.dirs.append(rel)
            elif path.is_file():
                st = path.stat()
                out.files[rel] = (st.st_ino, st.st_size, st.st_mtime_ns)

        return out

    def sizes(self) -> dict[str, tuple[int, int]]:
        """Each file's size and change time, which a restore brings back. Inodes change with every restore."""
        return {rel: (size, mtime) for rel, (_, size, mtime) in self.files.items()}


def changed_by(before: Layout, after: Layout) -> list[str]:
    """The files of `before` that are gone from `after` or differ in inode, size or change time."""
    return sorted(rel for rel, state in before.files.items() if after.files.get(rel) != state)


def copy_tree(source: Path, dest: Path, clones: set[str]) -> None:
    """Mirrors `source` at `dest`. Files named in `clones` become clones, and the rest become hard links."""
    layout = Layout.of(source)

    for rel in layout.dirs:
        (dest / rel).mkdir(parents=True, exist_ok=True)

    for rel in layout.files:
        if rel in clones:
            clone(source / rel, dest / rel)
        else:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            os.link(source / rel, dest / rel)


def read_data(path: Path, chunk: int = 1 << 24) -> int:
    """Reads every byte of the file at `path` that holds data and returns how many. It skips holes, so the holes of a
    sparse file, eg. a log the system made at a fixed size, don't fill the page cache with pages of zeros."""
    done = 0
    fd = os.open(path, os.O_RDONLY)

    try:
        size, at = os.fstat(fd).st_size, 0

        while at < size:
            try:
                start = os.lseek(fd, at, os.SEEK_DATA)
            except OSError:
                # No data lies past `at`.
                break

            end = os.lseek(fd, start, os.SEEK_HOLE)
            os.lseek(fd, start, os.SEEK_SET)

            while start < end:
                got = len(os.read(fd, min(chunk, end - start)))

                if got == 0:
                    break

                start, done = start + got, done + got

            at = end
    finally:
        os.close(fd)

    return done


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True


def sweep(sessions: Path) -> list[str]:
    """Removes the private copies under `sessions` whose process is gone, and the folder itself once it is empty."""
    removed = []

    if not sessions.is_dir():
        return removed

    for folder in sorted(sessions.iterdir()):
        owner = folder.name.split("-")[0]

        if owner.isdigit() and not _alive(int(owner)):
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(str(folder))

    try:
        sessions.rmdir()
    except OSError:
        pass

    return removed


class Store:
    """The folder `data` under an index folder, which the system opens and writes to, with its stash and record."""

    def __init__(self, index: Path | str, data: str):
        self.index = Path(index)
        self.data = self.index / data
        self.stash = self.index / STASH
        self.sessions = self.index.with_name(self.index.name + ".sessions")
        self._record: dict | None = None
        self._claim: int | None = None

    @property
    def record(self) -> dict:
        """The store as its build left it, from store.json, read once."""
        if self._record is None:
            self._record = json.loads((self.index / RECORD).read_text())

        return self._record

    def read(self) -> None:
        """Reads the record now if it isn't read yet, eg. before a timed open."""
        _ = self.record

    def seal(self, session: Callable[[Path], None]) -> dict:
        """Learns which files a session changes, stashes them and writes the record. `session` opens the store in the
        folder it gets, eg. runs one search, and closes it. It runs on a copy made of clones, so the store itself stays
        as the build left it."""
        scratch = self.index / LEARN
        shutil.rmtree(scratch, ignore_errors=True)
        before = Layout.of(self.data)
        copy_tree(self.data, scratch, set(before.files))
        start = Layout.of(scratch)
        session(scratch)
        end = Layout.of(scratch)
        shutil.rmtree(scratch)

        if Layout.of(self.data).files != before.files:
            raise StoreChanged(f"a session on a copy of {self.data} changed the store itself, so the store names its "
                               f"files by absolute paths and can't be copied")

        changed = changed_by(start, end)

        for rel in changed:
            clone(self.data / rel, self.stash / rel)

        (self.index / CLAIM).touch()
        files = {rel: list(state) for rel, state in before.sizes().items()}
        self._record = {"data": self.data.name, "files": files, "dirs": before.dirs, "stash": changed,
                        "made": sorted(set(end.files) - set(start.files))}
        (self.index / RECORD).write_text(json.dumps(self._record, indent=1))
        return self._record

    def store_bytes(self) -> int:
        """The bytes of the store's own files, as the build left them."""
        return sum(size for size, _ in self.record["files"].values())

    def stash_bytes(self) -> int:
        """The bytes of the stash, whose clones share their disk blocks with the store but count again in a listing."""
        return sum(self.record["files"][rel][0] for rel in self.record["stash"])

    def claim(self) -> bool:
        """Takes the lock that lets this process open the store in place, and returns whether it got it. The lock holds
        until `release` or the end of the process."""
        if self._claim is not None:
            return True

        fd = os.open(self.index / CLAIM, os.O_RDONLY | os.O_CREAT, 0o644)

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False

        self._claim = fd
        return True

    @property
    def claimed(self) -> bool:
        return self._claim is not None

    def release(self) -> None:
        if self._claim is not None:
            os.close(self._claim)
            self._claim = None

    def restore(self) -> list[str]:
        """Puts the store back as its build left it. It deletes the files and folders a session made and brings back the
        stashed files a session changed or removed. It returns what it fixed, and raises StoreChanged when a file
        without a stashed copy changed."""
        want = {rel: tuple(v) for rel, v in self.record["files"].items()}
        stash = set(self.record["stash"])
        now = Layout.of(self.data)
        sizes = now.sizes()
        lost = sorted(rel for rel, state in want.items() if rel not in stash and sizes.get(rel) != state)

        if lost:
            raise StoreChanged(f"a session changed {lost[:5]} in {self.data}, which no session changed when the build "
                               f"sealed the store")

        fixed = []

        for rel in sorted(set(now.files) - set(want)):
            (self.data / rel).unlink()
            fixed.append(rel)

        for rel in sorted(set(now.dirs) - set(self.record["dirs"]), reverse=True):
            shutil.rmtree(self.data / rel, ignore_errors=True)

        for rel in sorted(stash):
            if sizes.get(rel) == want[rel]:
                continue

            (self.data / rel).unlink(missing_ok=True)
            clone(self.stash / rel, self.data / rel)
            fixed.append(rel)

        for rel in self.record["dirs"]:
            (self.data / rel).mkdir(parents=True, exist_ok=True)

        return fixed

    def private_copy(self, shared: Callable[[str], bool]) -> Path:
        """A copy of the store for a second process, under <index>.sessions/<pid>-<hex>/. The files `shared` names are
        hard links to the store's, so both processes read the same cached pages. They must be files no session
        changes. The rest are clones, eg. of lock files, which two processes can't share."""
        sweep(self.sessions)
        dest = self.sessions / f"{os.getpid()}-{os.urandom(3).hex()}"
        stash = set(self.record["stash"])

        for rel in self.record["dirs"]:
            (dest / rel).mkdir(parents=True, exist_ok=True)

        for rel in self.record["files"]:
            if rel in stash:
                clone(self.stash / rel, dest / rel)
            elif shared(rel):
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                os.link(self.data / rel, dest / rel)
            else:
                clone(self.data / rel, dest / rel)

        return dest

    def drop(self, copy: Path) -> None:
        """Deletes a private copy, and the sessions folder once it is empty."""
        shutil.rmtree(copy, ignore_errors=True)

        try:
            self.sessions.rmdir()
        except OSError:
            pass


class StoreAdapter(Adapter):
    """An adapter whose index folder holds a sealed store in the folder named `data`. A session opens the store in
    place and restores it when it closes. A worker process that measures queries per second (QPS), or a process that
    finds the store claimed by another, opens a private copy instead. Before READY a cold trial claims and restores
    the stores it will open, so the timed open reads no record and writes nothing first."""

    data = "store"

    def __init__(self) -> None:
        super().__init__()
        self.private = False
        self._stores: dict[str, Store] = {}
        self._meta: dict[str, dict] = {}

    def store(self, path: Path | str) -> Store:
        key = str(path)

        if key not in self._stores:
            self._stores[key] = Store(path, self.data)

        return self._stores[key]

    def meta(self, path: Path | str) -> dict:
        """The build's meta.json of the index at `path`, read once."""
        key = str(path)

        if key not in self._meta:
            self._meta[key] = graphs.read_meta(path)

        return self._meta[key]

    def prepare(self, path: Path) -> None:
        """Reads what the timed open of the index at `path` needs, its build record and settings. Runs before READY."""
        self.meta(path)
        self.store(path).read()

    def claim(self, path: Path) -> Store:
        """Claims the store at `path` for this process unless it is a QPS worker, and restores it once claimed."""
        store = self.store(path)

        if not self.private and not store.claimed and store.claim():
            store.restore()

        return store

    def begin(self, path: Path, meta: dict) -> tuple[Store, Path]:
        """The store at `path` and the folder this process opens. That is the store itself once this process claimed
        it, else a private copy, which shares the bulk data a search reads."""
        store = self.claim(path)

        if store.claimed and not self.private:
            return store, store.data

        return store, store.private_copy(lambda rel: self.kind(rel, meta) == "bulk")

    def end(self, store: Store, folder: Path) -> None:
        """Ends a session after the system closed. The store itself is restored and released, and a private copy is
        deleted."""
        if folder == store.data:
            store.restore()
            store.release()
        else:
            store.drop(folder)

    def cold_setup(self, ctx: ServeContext) -> None:
        for path in (ctx.index, ctx.dummy):
            if path is not None:
                self.claim(path)
                self.prepare(path)

        super().cold_setup(ctx)

    def warm_run(self, ctx: ServeContext, grid: list[dict], qids: np.ndarray, ids_dir: Path, passes: int,
                 warmup: int, qps_seconds: float, qps_repeats: int) -> Iterator[dict]:
        self.claim(ctx.index)
        yield from super().warm_run(ctx, grid, qids, ids_dir, passes, warmup, qps_seconds, qps_repeats)

    def qps_worker(self, ctx: ServeContext, wire: Any, seconds: float) -> None:
        self.private = True
        super().qps_worker(ctx, wire, seconds)

    def warm_up(self, handle: Any, ctx: ServeContext) -> None:
        """Mode "whole" reads the data of every store file, skipping holes. The stash and the records stay unread,
        since no session reads them, and so do files marked "loaded", which the system holds in its own memory."""
        if ctx.mode != "whole":
            raise Unsupported(f"{self.system} has no warm mode {ctx.mode}")

        for spec in self.files(ctx.index):
            ranges = spec["ranges"]

            if ranges and not all(r.get("loaded") for r in ranges) and os.path.isfile(spec["path"]):
                read_data(Path(spec["path"]))

    def stats(self, handle: Any) -> dict:
        """Marks a search that ran on a private copy, since another process held the store."""
        return {} if handle.folder == handle.store.data else {"private_copy": True}

    def kind(self, rel: str, meta: dict) -> str:
        """Whether the store's file `rel` is "bulk" or "routing" data, given the build's meta.json."""
        return "routing"

    def loaded(self, rel: str, meta: dict) -> bool:
        """Whether the system copies the store's file `rel` into its own memory at open, given the build's
        meta.json."""
        return False

    def store_files(self, path: Path, meta: dict) -> list[dict]:
        """The files of the index at `path` as the build left them. Store files get one range of the kind `kind`
        gives, marked "loaded" where `loaded` says so. The stash, the records and the claim file get no range, since
        no session reads them."""
        store = self.store(path)
        out = []

        for rel, (size, _) in sorted(store.record["files"].items()):
            span = {"offset": 0, "length": size, "kind": self.kind(rel, meta)}
            ranges = [span | ({"loaded": True} if self.loaded(rel, meta) else {})] if size else []
            out.append({"path": str(store.data / rel), "bytes": size, "ranges": ranges})

        for rel in store.record["stash"]:
            out.append({"path": str(store.stash / rel), "bytes": store.record["files"][rel][0], "ranges": []})

        for name in (RECORD, CLAIM, graphs.META):
            side = store.index / name
            out.append({"path": str(side), "bytes": side.stat().st_size if side.exists() else 0, "ranges": []})

        return out
