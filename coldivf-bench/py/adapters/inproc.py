"""Small helpers for adapters that run a library in this process.

`settings` checks a build config or a search setting against the keys a system takes. `as_ids` turns one row of a
library's results into the ids the driver expects. `check_unit` makes sure a cosine dataset has unit vectors, for
libraries that rank by inner product or by L2 distance, which rank unit vectors as the cosine does. `LiveRows` and
`HarnessWrites` are the writes the harness makes itself for the exact reference of the write benchmark.
"""

from __future__ import annotations

import json
import os
import platform
import struct
from pathlib import Path
from typing import Any

import numpy as np

import writes

# The default of a key that has to be given.
REQUIRED = object()


def settings(given: dict | None, defaults: dict[str, Any], what: str) -> dict[str, Any]:
    """`given` laid over `defaults`. A key that isn't in `defaults` is an error, so a typo never runs a default, and
    so is a key whose default is REQUIRED and that `given` leaves out."""
    given = dict(given or {})
    unknown = sorted(set(given) - set(defaults))

    if unknown:
        raise ValueError(f"unknown {what} {unknown}, expected some of {sorted(defaults)}")

    out = defaults | given
    missing = sorted(k for k, v in out.items() if v is REQUIRED)

    if missing:
        raise ValueError(f"the {what} need {missing}")

    return out


def as_ids(found: Any, k: int, empty: int | None = None) -> np.ndarray:
    """One row of results as k int64 ids, nearest first. Slots the library left empty, which hold `empty`, become -1,
    and a short row is padded with -1."""
    row = np.asarray(found).ravel()[:k]
    ids = row.astype(np.int64)

    if empty is not None:
        ids[row == empty] = -1

    out = np.full(k, -1, dtype=np.int64)
    out[: len(ids)] = ids
    return out


def check_unit(rows: np.ndarray, samples: int = 1000, tol: float = 1e-3) -> None:
    """Raises ValueError unless `samples` rows spread evenly over `rows` have unit length, within `tol`."""
    if len(rows) == 0:
        return

    picks = np.unique(np.linspace(0, len(rows) - 1, min(samples, len(rows))).astype(np.int64))
    norms = np.linalg.norm(np.asarray(rows[picks], dtype=np.float64), axis=1)
    worst = int(np.argmax(np.abs(norms - 1.0)))

    if abs(norms[worst] - 1.0) > tol:
        raise ValueError(f"row {int(picks[worst])} has length {norms[worst]:.6f}, and cosine here needs unit vectors")


def simd(arm: str, x86: str) -> str:
    """The SIMD (single instruction, multiple data) level of a wheel that uses NEON on Arm and picks its x86 kernels at
    run time."""
    return arm if platform.machine() in ("arm64", "aarch64") else x86


class LiveRows:
    """The harness's own writes for the exact reference, in the index folder beside the build's files. Inserted rows go
    to inserts.fbin, whose 8-byte header holds their count and width, and every id's state to live.bin, one byte each,
    1 while it is live. Each call writes its bytes to both files, so a change survives the process dying, and `sync`
    makes both survive a power loss with F_FULLFSYNC. An id is a row's place, so it numbers its inserts itself, as
    faiss does, which in a stream are their rows."""

    INSERTS = "inserts.fbin"
    LIVE = "live.bin"

    def __init__(self, folder: Path | str, base_rows: int, dim: int):
        self.folder, self.base_rows, self.dim = Path(folder), base_rows, dim
        inserts, live = self.folder / self.INSERTS, self.folder / self.LIVE

        if not inserts.exists():
            inserts.write_bytes(struct.pack("<ii", 0, dim))

        if not live.exists():
            live.write_bytes(b"\1" * base_rows)

        self.ins_fd = os.open(inserts, os.O_RDWR)
        self.live_fd = os.open(live, os.O_RDWR)
        count, width = struct.unpack("<ii", os.pread(self.ins_fd, 8, 0))

        if width != dim or os.fstat(self.live_fd).st_size != base_rows + count:
            self.close()
            raise ValueError(f"{self.folder} holds {count} inserts of width {width} and a live mask that don't match "
                             f"a base of {base_rows} rows of width {dim}")

        self.inserted = count

    @property
    def count(self) -> int:
        """One past the last id the index holds or held."""
        return self.base_rows + self.inserted

    def insert(self, rows: np.ndarray) -> np.ndarray:
        """Appends `rows` and returns the ids it gave them, the next ones in order. The header's count goes last, so a
        kill between the writes leaves the rows out."""
        block = np.ascontiguousarray(rows, dtype="<f4").reshape(-1, self.dim)
        ids = np.arange(self.count, self.count + len(block), dtype=np.int64)
        os.pwrite(self.ins_fd, block.tobytes(), 8 + self.inserted * self.dim * 4)
        os.pwrite(self.live_fd, b"\1" * len(block), self.count)
        self.inserted += len(block)
        os.pwrite(self.ins_fd, struct.pack("<i", self.inserted), 0)
        return ids

    def delete(self, ids: np.ndarray) -> None:
        for i in np.asarray(ids, dtype=np.int64).tolist():
            if not 0 <= i < self.count:
                raise ValueError(f"no id {i} in an index of {self.count} ids")

            os.pwrite(self.live_fd, b"\0", i)

    def sync(self) -> dict:
        for name in (self.INSERTS, self.LIVE):
            writes.fullfsync(self.folder / name)

        return {"files": [self.INSERTS, self.LIVE], "call": "F_FULLFSYNC"}

    def write_meta(self) -> None:
        """Puts the state's id count and live count in meta.json, synced and renamed into place."""
        meta = json.loads((self.folder / "meta.json").read_text())
        live = np.fromfile(self.folder / self.LIVE, dtype=np.uint8)
        meta.update(state_rows=self.count, live_rows=int(np.count_nonzero(live)))
        writes.write_json(self.folder / "meta.json", meta)
        writes.fullfsync(self.folder / "meta.json")

    def close(self) -> None:
        for fd in (getattr(self, "ins_fd", None), getattr(self, "live_fd", None)):
            if fd is not None:
                os.close(fd)

        self.ins_fd = self.live_fd = None

    @classmethod
    def view(cls, folder: Path | str) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The inserted rows, mapped, and whether each id is live, for a search, or None for each that a folder lacks,
        eg. a build that took no writes. It maps as many rows as the header counts."""
        folder = Path(folder)
        inserts, live = folder / cls.INSERTS, folder / cls.LIVE
        rows = mask = None

        if inserts.exists():
            with open(inserts, "rb") as f:
                count, width = struct.unpack("<ii", f.read(8))

            rows = np.memmap(inserts, dtype="<f4", mode="r", offset=8, shape=(count, width)) if count else \
                np.zeros((0, width), dtype=np.float32)

        if live.exists():
            mask = np.fromfile(live, dtype=np.uint8).astype(bool)

        return rows, mask


class HarnessWrites:
    """The write hooks of an adapter whose writes the harness makes itself, i.e. the exact reference. Put it before
    Adapter among the bases, and make the adapter's search read `LiveRows.view` of the index folder, so it scans the
    inserted rows too and skips the ids that aren't live. The build's meta.json must hold its row count as "rows"."""

    write_role = "exact"
    durability = {"call": "process", "sync": "power"}
    sync_kind = "incremental"

    def writer_open(self, ctx: Any, path: Path) -> LiveRows:
        meta = json.loads((Path(path) / "meta.json").read_text())
        return LiveRows(path, int(meta["rows"]), ctx.dataset.dim)

    def insert(self, w: LiveRows, ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return w.insert(rows)

    def delete(self, w: LiveRows, ids: np.ndarray) -> None:
        w.delete(ids)

    def sync(self, w: LiveRows) -> dict:
        return w.sync()

    def durable_point(self, w: LiveRows) -> dict:
        done = w.sync()
        w.write_meta()
        return done | {"meta": True}

    def writer_close(self, w: LiveRows) -> None:
        w.close()
