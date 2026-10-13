"""turbovec, a flat index of TurboQuant codes with 2, 3 or 4 bits per dimension.

Every vector is turned by one shared random rotation, and each coordinate is coded against a fixed codebook, so a build
trains nothing. A search scans every vector. From 32,768 vectors on it scans the sign bits for a shortlist first, then
scores the shortlist with the full codes. Scores are inner products, so a cosine dataset runs on its unit vectors.

The config takes `bits` and `calibrate`, the size of a seeded uniform sample of rows that fits the library's
per-coordinate calibration before any row goes in. It is 0 by default, which is the library's plain TurboQuant. Rows go
in blocks of 65,536, which codes them as one call would. A search takes no settings. The index is one file, which
loads whole at open.

With no variant the index is a TurboQuantIndex, which numbers its vectors by their place, written to index.tv. It has
no delete by id, so it takes writes only as a rebuild. The variant "idmap" builds an IdMapIndex instead, which keeps
each vector under the id it was added with, row i under id i, and writes it to index.tvim with `sync`, the call that
later saves only what changed. That variant takes writes: an insert is add_with_ids of the stream's ids, a delete is
remove of one id per call, which takes the row out of the index, and the sync is `sync`, which writes what changed
since the last one into the file and makes it survive a power loss before it returns. A sync after many changes, eg.
2,000 removes, or 1,100 inserts and 1,100 removes, writes the whole file again.

Searches and adds run on Rayon's global pool, which takes its size from RAYON_NUM_THREADS when the process first uses
it. The library has no call that sets its threads, so the environment is the setting. A batch search releases Python's
global interpreter lock and spreads over that pool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adapters import inproc
from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, WriteContext, main
import schema

FILE = "index.tv"
IDMAP_FILE = "index.tvim"
IDMAP = "idmap"
VARIANTS = ("", IDMAP)
BLOCK = 65536
BITS = (2, 3, 4)
BUILD = {"bits": inproc.REQUIRED, "calibrate": 0}


def sample_rows(rows: int, count: int, seed: int) -> np.ndarray:
    """`count` distinct row numbers below `rows`, drawn uniformly with `seed`, in increasing order."""
    count = min(int(count), rows)
    return np.sort(np.random.default_rng(seed).choice(rows, size=count, replace=False))


def code_bytes(rows: int, dim: int, bits: int) -> int:
    """The bytes of the codes, without the length kept per vector. At 3 bits a byte holds two codes."""
    per_byte = 2 if bits == 3 else 8 // bits
    return rows * (-(-dim // per_byte))


@dataclass
class Writer:
    """The loaded IdMapIndex of a write session and the file its syncs append to."""

    index: Any
    file: Path


class TurboVec(Adapter):
    system = "turbovec"
    binding = "python"
    batch_api = True
    workers = "Rayon's global pool, which runs every search"
    # The index file loads whole at open.
    loads_at_open = True

    def load(self) -> None:
        import turbovec

        if self.variant not in VARIANTS:
            raise ValueError(f"turbovec has no variant {self.variant!r}, only {[v for v in VARIANTS if v]}")

        self.tv = turbovec

        if self.idmap:
            self.write_role, self.sync_kind = "full", "incremental"
            self.durability = {"call": "memory", "sync": "power"}
            # A sync writes the rows that changed and both headers over their old bytes in index.tvim.
            self.writes_in_place = True

    @property
    def idmap(self) -> bool:
        return self.variant == IDMAP

    def version(self) -> str:
        return str(self.tv.__version__)

    def wheels(self) -> dict[str, str | None]:
        return {"turbovec": schema.dist_sha256("turbovec")}

    def simd_level(self) -> str:
        return inproc.simd("neon", "avx512 or avx2, picked at run time")

    def thread_env(self, threads: int) -> dict[str, str]:
        return {"RAYON_NUM_THREADS": str(threads)}

    def overhead_config(self) -> dict:
        return {"bits": 4}

    def index_class(self) -> Any:
        return self.tv.IdMapIndex if self.idmap else self.tv.TurboQuantIndex

    def index_file(self, path: Path) -> Path:
        path = Path(path)
        return path / (IDMAP_FILE if self.idmap else FILE) if path.is_dir() else path

    def build(self, ctx: BuildContext) -> dict:
        cfg = inproc.settings(ctx.config, BUILD, "turbovec build settings")
        bits = int(cfg["bits"])

        if bits not in BITS:
            raise ValueError(f"bits is {bits}, not 2, 3 or 4")

        if ctx.metric == "l2":
            raise Unsupported("turbovec ranks by inner product, which isn't L2 distance in general")

        rows = ctx.data

        if ctx.metric == "cosine":
            inproc.check_unit(rows)

        index = self.index_class()(dim=ctx.dim, bit_width=bits)
        start = time.perf_counter()

        if int(cfg["calibrate"]) > 0:
            picks = sample_rows(ctx.rows, int(cfg["calibrate"]), ctx.seed)
            index.calibrate(np.ascontiguousarray(rows[picks], dtype=np.float32))

        for lo in range(0, ctx.rows, BLOCK):
            block = np.ascontiguousarray(rows[lo : lo + BLOCK], dtype=np.float32)

            if self.idmap:
                index.add_with_ids(block, np.arange(lo, lo + len(block), dtype=np.uint64))
            else:
                index.add(block)

        encode = time.perf_counter() - start
        file = self.index_file(ctx.out)
        start = time.perf_counter()

        # An IdMapIndex is written with sync, so the first sync of a write session appends to it rather than writing
        # the whole file again.
        if self.idmap:
            index.sync(str(file))
        else:
            index.write(str(file))

        saved = time.perf_counter() - start
        return {
            "encode_seconds": encode,
            "stored_vector_bytes": code_bytes(ctx.rows, ctx.dim, bits),
            "exact_f32_stored": False,
            "extra": {"bits": bits, "calibrate": int(cfg["calibrate"]), "calibration": index.calibration_state,
                      "save_seconds": saved, "vectors": len(index), "index_class": type(index).__name__},
        }

    def open(self, ctx: ServeContext, path: Path) -> Any:
        return self.index_class().load(str(self.index_file(path)))

    def configure(self, handle: Any, params: dict) -> None:
        inproc.settings(params, {}, "turbovec search settings")

    def cold_setup(self, ctx: ServeContext) -> None:
        inproc.settings(ctx.params, {}, "turbovec search settings")
        super().cold_setup(ctx)

    def warm_up(self, handle: Any, ctx: ServeContext) -> None:
        super().warm_up(handle, ctx)
        handle.prepare()

    def search(self, handle: Any, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        _, ids = handle.search(np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1), k)
        return inproc.as_ids(ids[0] if len(ids) else [], k)

    def search_batch(self, handle: Any, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        _, ids = handle.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return np.stack([inproc.as_ids(row, k) for row in ids])

    def writer_open(self, ctx: WriteContext, path: Path) -> Writer:
        if not self.idmap:
            raise Unsupported("a TurboQuantIndex numbers its vectors by their place and has no delete by id, so it "
                              "takes the stream as a fresh build. The idmap variant takes writes")

        file = self.index_file(path)
        return Writer(index=self.tv.IdMapIndex.load(str(file)), file=file)

    def insert(self, w: Writer, ids: np.ndarray, rows: np.ndarray) -> None:
        """One add_with_ids, the library's only add, which takes a batch of one as one row."""
        w.index.add_with_ids(np.ascontiguousarray(rows, dtype=np.float32), np.asarray(ids, dtype=np.uint64))

    def delete(self, w: Writer, ids: np.ndarray) -> None:
        """A remove for each id, since the library removes one id per call."""
        missing = 0

        for i in np.asarray(ids, dtype=np.int64).tolist():
            if not w.index.remove(i):
                missing += 1

        if missing:
            raise RuntimeError(f"turbovec held {len(ids) - missing} of the {len(ids)} ids it was asked to remove")

    def sync(self, w: Writer) -> dict:
        w.index.sync(str(w.file))
        return {"call": "sync", "file": w.file.name}

    def pending(self, w: Writer) -> dict:
        return {"vectors": len(w.index)}

    def writer_close(self, w: Writer) -> None:
        w.index = None


if __name__ == "__main__":
    main(TurboVec)
