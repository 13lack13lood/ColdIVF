"""Exact search with numpy over a copy of the base rows. It is the smallest complete adapter, and its warm recall checks
the ground truth.

The index folder holds vectors.npy, the rows as float32, which `open` maps. A search scans every row in chunks.

It is also the exact reference of the write benchmark, whose writes the harness makes. Inserted rows go to inserts.fbin
beside vectors.npy and each id's state to live.bin, both synced with F_FULLFSYNC, and a search scans the inserted rows
too and skips every id that isn't live, so its recall is 1 at every state when the state's truth is right.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import Adapter, BuildContext, ServeContext, main
from adapters.inproc import HarnessWrites, LiveRows

CHUNK = 65536


def lengths(rows: np.ndarray) -> np.ndarray:
    """Each row's length, with 1 for a row of zeros."""
    norms = np.sqrt(np.einsum("ij,ij->i", rows, rows, dtype=np.float64)).astype(np.float32)
    norms[norms == 0] = 1
    return norms


def score_into(rows: np.ndarray, query: np.ndarray, metric: str, out: np.ndarray) -> None:
    """Each row's score into `out`, smaller is nearer: the squared distance for L2, and minus the inner product
    otherwise, which a cosine search then divides by the row's length."""
    for start in range(0, len(rows), CHUNK):
        block = rows[start : start + CHUNK]

        if metric == "l2":
            diff = block - query
            out[start : start + len(block)] = np.einsum("ij,ij->i", diff, diff)
        else:
            out[start : start + len(block)] = -(block @ query)


class Flat(Adapter):
    system = "flat"

    def version(self) -> str:
        return f"numpy {np.__version__}"

    def build(self, ctx: BuildContext) -> dict:
        rows = ctx.data
        out = np.lib.format.open_memmap(ctx.out / "vectors.npy", mode="w+", dtype=np.float32, shape=rows.shape)

        for start in range(0, len(rows), CHUNK):
            out[start : start + CHUNK] = rows[start : start + CHUNK]

        out.flush()
        del out
        (ctx.out / "meta.json").write_text(json.dumps({"metric": ctx.metric, "rows": ctx.rows}))
        return {"stored_vector_bytes": int(rows.nbytes), "exact_f32_stored": True}

    def open(self, ctx: ServeContext, path: Path) -> Any:
        """Maps the build's rows and the inserted rows, which an id numbers in that order, and reads the mask of live
        ids, which a folder that took no writes lacks."""
        meta = json.loads((Path(path) / "meta.json").read_text())
        vectors = np.load(Path(path) / "vectors.npy", mmap_mode="r")
        inserts, live = LiveRows.view(path)
        blocks = [vectors] + ([inserts] if inserts is not None and len(inserts) else [])
        norms = np.concatenate([lengths(b) for b in blocks]) if meta["metric"] == "cosine" else None
        return {"vectors": vectors, "inserts": inserts, "live": live, "blocks": blocks, "metric": meta["metric"],
                "norms": norms}

    def search(self, handle: Any, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        blocks, live, metric = handle["blocks"], handle["live"], handle["metric"]
        scores = np.empty(sum(len(b) for b in blocks), dtype=np.float32)
        at = 0

        for rows in blocks:
            score_into(rows, query, metric, scores[at : at + len(rows)])
            at += len(rows)

        if metric == "cosine":
            scores /= handle["norms"]

        # A deleted id never comes back, so a search returns fewer ids only when fewer are live.
        if live is not None:
            alive = live[: len(scores)]
            scores[~alive] = np.inf
            k = min(k, int(np.count_nonzero(alive)))

        k = min(k, len(scores))

        if k == 0:
            return np.zeros(0, dtype=np.int64)

        top = np.argpartition(scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        order = np.lexsort((top, scores[top]))
        return top[order].astype(np.int64)


class ExactReference(HarnessWrites, Flat):
    """The flat scan as the driver runs it, with the harness's writes, as the write benchmark's exact reference. The
    writes go on this class rather than on Flat, since the test adapter puts HarnessWrites before Flat among its own
    bases."""


if __name__ == "__main__":
    main(ExactReference)
