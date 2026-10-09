"""RaBitQ-Library's HNSW (hierarchical navigable small world) graph over RaBitQ codes of 1 to 9 bits per dimension.

The library links the graph with distances between the raw rows, then keeps only the links and the codes. Each row is
coded relative to the center of its cluster, so a build clusters the rows first. The config takes `bits`, `M` (16),
`ef_construction` (200), the cluster count `clusters` (16, as in the library's examples), `niter` (25) and
`fast_quantization` (false). A shared partition from --clusters replaces the library's k-means, and a given `clusters`
must match it. Clusters that no row of the prefix uses are dropped. The graph's random levels take the build's seed. A
search takes `ef`. The index is one file, which loads whole at open.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from adapters import inproc
from adapters.base import BuildContext, main
from adapters.rabitq_common import RabitqAdapter, code_bytes, compact, library_metric, list_sizes

BITS = tuple(range(1, 10))
BUILD = {"bits": inproc.REQUIRED, "M": 16, "ef_construction": 200, "clusters": None, "niter": 25,
         "fast_quantization": False}
CLUSTERS = 16
SEARCH = {"ef": inproc.REQUIRED}


class RabitqHnsw(RabitqAdapter):
    system = "rabitq-hnsw"
    file_name = "hnsw.rabitq"
    kind = "HnswIndex"

    def overhead_config(self) -> dict:
        return {"bits": 4, "ef_construction": 64, "clusters": 1}

    def overhead_params(self) -> dict:
        return {"ef": 10}

    def check_search(self, params: dict) -> dict:
        out = inproc.settings(params, SEARCH, "rabitq-hnsw search settings")

        if int(out["ef"]) < 1:
            raise ValueError("ef must be at least 1")

        return out

    def build(self, ctx: BuildContext) -> dict:
        cfg = inproc.settings(ctx.config, BUILD, "rabitq-hnsw build settings")
        bits, metric = int(cfg["bits"]), library_metric(ctx.metric)

        if bits not in BITS:
            raise ValueError(f"bits is {bits}, not 1 to 9")

        if ctx.metric == "cosine":
            inproc.check_unit(ctx.data)

        if cfg["clusters"] is not None:
            count = int(cfg["clusters"])
        else:
            count = None if ctx.clusters is not None else CLUSTERS

        part = compact(self.partition(ctx, count, None, cfg["niter"]))
        index = self.lib.HnswIndex(dim=ctx.dim, max_elements=ctx.rows, M=int(cfg["M"]),
                                   ef_construction=int(cfg["ef_construction"]), nbits=bits, metric=metric,
                                   random_seed=int(ctx.seed))
        start = time.perf_counter()
        index.build(ctx.data, part.centers, part.assign, num_threads=ctx.threads,
                    fast_quantization=bool(cfg["fast_quantization"]))
        encode = time.perf_counter() - start
        saved = self.save(index, ctx.out)
        return {
            "cluster_seconds": part.seconds,
            "encode_seconds": encode,
            "stored_vector_bytes": code_bytes(ctx.rows, ctx.dim, bits),
            "exact_f32_stored": False,
            "extra": part.info | list_sizes(part) | {
                "bits": bits, "M": int(cfg["M"]), "ef_construction": int(cfg["ef_construction"]), "metric": metric,
                "save_seconds": saved,
                "library": {"num_points": int(index.num_points), "num_clusters": int(index.num_clusters),
                            "nbits": int(index.nbits), "metric": str(index.metric)},
            },
        }

    def query(self, index: Any, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        ids, _ = index.search(queries, k=k, ef=int(params["ef"]), num_threads=threads)
        return ids


if __name__ == "__main__":
    main(RabitqHnsw)
