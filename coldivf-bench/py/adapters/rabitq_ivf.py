"""RaBitQ-Library's IVF (inverted file index) over RaBitQ codes of 1 to 9 bits per dimension, or over raw float32 rows.

A build takes its partition from --clusters or from the library's k-means, then encodes every cluster in one call. The
config's `bits` is 1 to 9, or 32 for raw float32 rows, which rerank the candidates that the 1-bit codes let through.
The cluster count is `nlist` or `vectors_per_list`, and a shared partition brings its own. `kmeans` ("qg" or
"rabitq"), `niter` (25) and `fast_quantization` (false) keep the library's defaults unless given.

A search takes `nprobe` and, optionally, `high_accuracy`, which overrides the library's choice of scan precision. The
index is one file, plus a .hnsw file when the library routes queries through a graph of the centers, which it does from
20,000 clusters on. Both load whole at open.

An add copies the whole index and grows it by its rows, so it needs no room made first. A removed row keeps its slot
and its code, which searches still scan.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from adapters import inproc
from adapters.base import BuildContext, main
from adapters.rabitq_common import (
    RabitqAdapter, RabitqWrites, Writer, cluster_count, code_bytes, library_metric, list_sizes,
)

BITS = (*range(1, 10), 32)
BUILD = {"bits": inproc.REQUIRED, "nlist": None, "vectors_per_list": None, "kmeans": None, "niter": 25,
         "fast_quantization": False}
SEARCH = {"nprobe": inproc.REQUIRED, "high_accuracy": None}


def bits_of(value: Any) -> int:
    bits = int(value)

    if bits not in BITS:
        raise ValueError(f"bits is {bits}, not 1 to 9 or 32")

    return bits


class RabitqIvf(RabitqWrites, RabitqAdapter):
    system = "rabitq-ivf"
    file_name = "ivf.rabitq"
    kind = "IvfIndex"

    def overhead_config(self) -> dict:
        return {"nlist": 8, "bits": 1}

    def overhead_params(self) -> dict:
        return {"nprobe": 1}

    def check_search(self, params: dict) -> dict:
        out = inproc.settings(params, SEARCH, "rabitq-ivf search settings")

        if int(out["nprobe"]) < 1:
            raise ValueError("nprobe must be at least 1")

        return out

    def build(self, ctx: BuildContext) -> dict:
        cfg = inproc.settings(ctx.config, BUILD, "rabitq-ivf build settings")
        bits, metric = bits_of(cfg["bits"]), library_metric(ctx.metric)

        if ctx.metric == "cosine":
            inproc.check_unit(ctx.data)

        if ctx.clusters is not None:
            # A shared partition sets the count, which a given nlist must match.
            count = int(cfg["nlist"]) if cfg["nlist"] is not None else None
        else:
            count = cluster_count(ctx.rows, cfg["nlist"], cfg["vectors_per_list"])

        part = self.partition(ctx, count, cfg["kmeans"], cfg["niter"])
        index = self.lib.IvfIndex(dim=ctx.dim, max_elements=ctx.rows, num_clusters=part.count, nbits=bits,
                                  metric=metric)
        start = time.perf_counter()
        index.build(ctx.data, part.centers, part.assign, num_threads=ctx.threads,
                    fast_quantization=bool(cfg["fast_quantization"]))
        encode = time.perf_counter() - start
        saved = self.save(index, ctx.out)
        return {
            "cluster_seconds": part.seconds,
            "encode_seconds": encode,
            "stored_vector_bytes": code_bytes(ctx.rows, ctx.dim, bits),
            "exact_f32_stored": bits == 32,
            "extra": part.info | list_sizes(part) | {
                "bits": bits, "metric": metric, "save_seconds": saved,
                "library": {"num_clusters": int(index.num_clusters), "nbits": int(index.nbits),
                            "metric": str(index.metric), "max_elements": int(index.max_elements)},
            },
        }

    def query(self, index: Any, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        ids, _ = index.search(queries, k=k, nprobe=int(params["nprobe"]), high_accuracy=params.get("high_accuracy"),
                              num_threads=threads)
        return ids

    def pending(self, w: Writer) -> dict:
        """The index's slots, the removed rows' among them."""
        return {"slots": int(w.index.max_elements)}


if __name__ == "__main__":
    main(RabitqIvf)
