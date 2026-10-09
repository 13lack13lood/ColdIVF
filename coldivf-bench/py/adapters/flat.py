"""Exact search with numpy over a copy of the base rows. It is the smallest complete adapter, and its warm recall checks
the ground truth.

The index folder holds vectors.npy, the rows as float32, which `open` maps. A search scans every row in chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import Adapter, BuildContext, ServeContext, main

CHUNK = 65536


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
        meta = json.loads((Path(path) / "meta.json").read_text())
        vectors = np.load(Path(path) / "vectors.npy", mmap_mode="r")
        norms = None

        if meta["metric"] == "cosine":
            norms = np.sqrt(np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64)).astype(np.float32)
            norms[norms == 0] = 1

        return {"vectors": vectors, "metric": meta["metric"], "norms": norms}

    def search(self, handle: Any, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        vectors, metric = handle["vectors"], handle["metric"]
        scores = np.empty(len(vectors), dtype=np.float32)

        for start in range(0, len(vectors), CHUNK):
            block = vectors[start : start + CHUNK]

            if metric == "l2":
                diff = block - query
                scores[start : start + len(block)] = np.einsum("ij,ij->i", diff, diff)
            else:
                scores[start : start + len(block)] = -(block @ query)

        if metric == "cosine":
            scores /= handle["norms"]

        k = min(k, len(scores))
        top = np.argpartition(scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        order = np.lexsort((top, scores[top]))
        return top[order].astype(np.int64)


if __name__ == "__main__":
    main(Flat)
