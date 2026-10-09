"""Empties the page cache of an index's files and checks which of their pages are still cached.

On this Mac a new memory map of a file can make another process, eg. an endpoint security agent, read the file's first
pages back into the cache 45 to 170 ms later. So every file is mapped once, when it is first seen, and every drop and
every check goes through that held mapping. A drop is msync with MS_INVALIDATE over the mapping, and a check is mincore
over it. Neither reads the file.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

MACOS = sys.platform == "darwin"
PAGE = os.sysconf("SC_PAGE_SIZE")

PROT_READ = 0x1
MAP_SHARED = 0x1
MS_INVALIDATE = 0x2
MINCORE_INCORE = 0x1
MAP_FAILED = ctypes.c_void_p(-1).value

# Every byte value mincore can report for a cached page.
_INCORE = bytes(b for b in range(256) if b & MINCORE_INCORE)

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int
_libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.msync.restype = ctypes.c_int
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_libc.mincore.restype = ctypes.c_int


def _os_error(what: str, path: Path) -> OSError:
    err = ctypes.get_errno()
    return OSError(err, f"{what} {path}: {os.strerror(err)}")


def _pages(length: int) -> int:
    return -(-length // PAGE)


class HeldMap:
    """One file mapped read-only, held until `close`. It never reads the mapped bytes."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        st = os.fstat(self.fd)
        self.size = st.st_size
        self.inode = st.st_ino
        self.mtime_ns = st.st_mtime_ns
        self.addr = None

        if self.size > 0:
            addr = _libc.mmap(None, self.size, PROT_READ, MAP_SHARED, self.fd, 0)

            if addr is None or addr == MAP_FAILED:
                err = _os_error("mapping", self.path)
                os.close(self.fd)
                raise err

            self.addr = addr

    @property
    def pages(self) -> int:
        return _pages(self.size)

    def changed(self) -> bool:
        """Whether the path now names another file, or the file's size changed, since it was mapped."""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return True

        return st.st_ino != self.inode or st.st_size != self.size

    def _span(self, start: int, end: int | None, inward: bool) -> tuple[int, int]:
        """The page-aligned span of bytes start to end, cut to the file. Rounded inward it leaves out partial pages."""
        end = self.size if end is None else min(end, self.size)
        start = max(0, start)

        if inward:
            lo, hi = -(-start // PAGE) * PAGE, end // PAGE * PAGE

            if end == self.size:
                hi = _pages(self.size) * PAGE
        else:
            lo, hi = start // PAGE * PAGE, _pages(end) * PAGE

        return lo, max(lo, min(hi, _pages(self.size) * PAGE))

    def sync(self) -> None:
        """Writes the file's dirty pages out, so dropping them loses nothing."""
        os.fsync(self.fd)

    def evict(self, start: int = 0, end: int | None = None, inward: bool = False) -> None:
        """Drops the cached pages of bytes start to end, the whole file by default."""
        if self.addr is None:
            return

        lo, hi = self._span(start, end, inward)

        if hi > lo and _libc.msync(self.addr + lo, hi - lo, MS_INVALIDATE) != 0:
            raise _os_error("invalidating", self.path)

    def resident(self, start: int = 0, end: int | None = None, inward: bool = False) -> tuple[int, int]:
        """How many pages of bytes start to end are cached, and how many pages that span holds."""
        if self.addr is None:
            return 0, 0

        lo, hi = self._span(start, end, inward)
        count = (hi - lo) // PAGE

        if count == 0:
            return 0, 0

        vec = (ctypes.c_uint8 * count)()

        if _libc.mincore(self.addr + lo, hi - lo, vec) != 0:
            raise _os_error("checking", self.path)

        not_cached = len(bytes(vec).translate(None, _INCORE))
        return count - not_cached, count

    def resident_bytes(self) -> int:
        pages, _ = self.resident()
        return min(pages * PAGE, self.size)

    def close(self) -> None:
        if self.addr is not None:
            _libc.munmap(self.addr, self.size)
            self.addr = None

        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass
class Range:
    """A byte range of one file, and whether it is bulk data or routing data."""

    path: Path
    offset: int
    length: int
    kind: str = "bulk"


@dataclass
class Residency:
    """Cached pages per file at one check."""

    pages: dict[str, int] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def resident_pages(self) -> int:
        return sum(self.pages.values())

    @property
    def total_pages(self) -> int:
        return sum(self.totals.values())

    @property
    def resident_bytes(self) -> int:
        return sum(min(self.pages[p] * PAGE, self.sizes.get(p, self.pages[p] * PAGE)) for p in self.pages)

    @property
    def files(self) -> int:
        return len(self.totals)

    @property
    def touched(self) -> int:
        return sum(1 for p in self.pages.values() if p > 0)

    @property
    def share(self) -> float:
        return self.resident_pages / self.total_pages if self.total_pages else 0.0

    def record(self) -> dict:
        return {
            "resident_pages": self.resident_pages,
            "total_pages": self.total_pages,
            "resident_bytes": self.resident_bytes,
            "files": self.files,
            "touched": self.touched,
        }


class Evictor:
    """The files of one or more indexes, each mapped once and held.

    `paths` names files and folders. A folder stands for every file under it, and `refresh` maps files that appear
    later and remaps files that change. Paths that don't exist yet, eg. a log the system writes on first use, are
    watched too.
    """

    def __init__(self, paths: list[Path | str]):
        self.paths = [Path(p) for p in paths]
        self.maps: dict[str, HeldMap] = {}
        self.events: list[str] = []
        self.refresh()

    def _walk(self) -> list[Path]:
        out: list[Path] = []

        for p in self.paths:
            if p.is_dir():
                out.extend(sorted(f for f in p.rglob("*") if f.is_file()))
            elif p.is_file():
                out.append(p)

        return out

    def refresh(self) -> list[str]:
        """Maps files that appeared and remaps files that changed. It returns their paths."""
        seen = set()
        new: list[str] = []

        for path in self._walk():
            key = str(path)
            seen.add(key)
            held = self.maps.get(key)

            if held is not None and not held.changed():
                continue

            if held is not None:
                held.close()

            self.maps[key] = HeldMap(path)
            new.append(key)

        for key in [k for k in self.maps if k not in seen]:
            self.maps.pop(key).close()
            new.append(key)

        if new:
            self.events.append(f"mapped {len(new)} new or changed files")

        return new

    def sync(self) -> None:
        for held in self.maps.values():
            held.sync()

    def evict(self) -> None:
        for held in self.maps.values():
            held.evict()

    def residency(self) -> Residency:
        out = Residency()

        for key, held in self.maps.items():
            out.pages[key], out.totals[key] = held.resident()
            out.sizes[key] = held.size

        return out

    def make_cold(self, wait: float = 0.5, sync: bool = True) -> Residency:
        """Drops every file, waits `wait` seconds for anything that reads a file back, drops them again, and checks
        every page. Any page still cached in the result rejects a cold trial."""
        if self.refresh():
            # A file mapped just now can be read back by another process later than usual, so wait twice as long.
            wait *= 2

        if sync:
            self.sync()

        self.evict()
        time.sleep(wait)
        self.evict()
        return self.residency()

    def ranges_of(self, ranges: list[Range]) -> list[tuple[HeldMap, int, int]]:
        out = []

        for r in ranges:
            held = self.maps.get(str(Path(r.path)))

            if held is None:
                raise KeyError(f"{r.path} isn't one of the held files")

            out.append((held, r.offset, r.offset + r.length))

        return out

    def evict_ranges(self, ranges: list[Range]) -> None:
        """Drops the whole pages inside each range. Pages shared with bytes outside a range stay."""
        for held, start, end in self.ranges_of(ranges):
            held.evict(start, end, inward=True)

    def resident_in(self, ranges: list[Range]) -> tuple[int, int]:
        """Cached pages and all pages over the whole pages inside each range."""
        cached = total = 0

        for held, start, end in self.ranges_of(ranges):
            c, t = held.resident(start, end, inward=True)
            cached, total = cached + c, total + t

        return cached, total

    def resident_bytes_in(self, ranges: list[Range]) -> int:
        """The cached bytes of the whole pages inside each range, where a file's last page counts only up to the end of
        the file. Ranges of files that aren't held count as 0, eg. a file listed before it exists."""
        total = 0

        for r in ranges:
            held = self.maps.get(str(Path(r.path)))

            if held is None:
                continue

            start, end = r.offset, r.offset + r.length
            cached, _ = held.resident(start, end, inward=True)
            lo, hi = held._span(start, end, inward=True)
            total += min(cached * PAGE, max(0, min(hi, held.size) - lo))

        return total

    def close(self) -> None:
        for held in self.maps.values():
            held.close()

        self.maps.clear()

    def __enter__(self) -> Evictor:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def residency_of(paths: list[Path | str]) -> Residency:
    """The cached pages of files and folders now, through fresh mappings. Use it only where a new mapping can't spoil a
    measurement, eg. after a warm run."""
    with Evictor(paths) as ev:
        return ev.residency()


def loaded_ranges(files: list[dict]) -> list[Range]:
    """The ranges of an index's file list, as an adapter's files() gives it, that are marked "loaded". The system copies
    those bytes into its own memory when it opens the index or first uses them, and never reads them from the file
    again, so the page cache's copy of them isn't memory the system needs. Ranges of one file that touch are joined, so
    a page they share counts as loaded too."""
    spans = sorted((str(Path(f["path"])), int(r["offset"]), int(r["length"]), r.get("kind", "bulk")) for f in files
                   for r in f.get("ranges") or [] if r.get("loaded") and int(r["length"]) > 0)
    out: list[Range] = []

    for path, offset, length, kind in spans:
        last = out[-1] if out else None

        if last is not None and str(last.path) == path and offset <= last.offset + last.length:
            last.length = max(last.length, offset + length - last.offset)
        else:
            out.append(Range(Path(path), offset, length, kind))

    return out


def resident_split(files: list[dict]) -> tuple[int, int]:
    """The cached bytes of an index's files now, through fresh mappings, as the bytes outside the "loaded" ranges and
    the bytes inside them. Use it only where a new mapping can't spoil a measurement, eg. after a warm run."""
    with Evictor([Path(f["path"]) for f in files]) as ev:
        loaded = ev.resident_bytes_in(loaded_ranges(files))
        return ev.residency().resident_bytes - loaded, loaded


def drop_loaded(files: list[dict]) -> int:
    """Drops the cached pages of the "loaded" ranges of an index's files, through fresh mappings, and returns the bytes
    that were cached there. Pages shared with bytes outside those ranges stay. Use it only where a new mapping can't
    spoil a measurement."""
    ranges = loaded_ranges(files)

    if not ranges:
        return 0

    with Evictor(sorted({r.path for r in ranges})) as ev:
        held = [r for r in ranges if str(r.path) in ev.maps]
        cached = ev.resident_bytes_in(held)
        ev.evict_ranges(held)
        return cached
