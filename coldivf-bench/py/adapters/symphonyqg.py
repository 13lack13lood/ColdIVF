"""RaBitQ-Library's SymphonyQG, a graph in which every row holds the 1-bit codes of its neighbors, so one step of the
search scores a whole neighbor list in one batch.

The config takes `degree` (32), a multiple of 32 below the row count, `ef_construction` (200), `quantization_bits`
(0, which keeps raw float32 rows, or 4 or 8 for codes) and `init` ("pipnn" or "random"). The library's build takes no
seed, so two builds of one config differ. A search takes `ef`, which must be at least k. The index is one file, which
loads whole at open. The library has no call that adds or removes a row, so the write benchmark rebuilds it.

A cosine dataset runs as L2 distance on its unit vectors, which ranks them as the cosine does. With "ip" the library's
default PiPNN build links a much weaker graph. On 10,000 Cohere rows it reached recall@10 0.59 at ef 64, against 0.97
for "l2" and for "ip" with random init.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from adapters import inproc
from adapters.base import Adapter, BuildContext, Unsupported, main
from adapters.rabitq_common import RabitqAdapter

BUILD = {"degree": 32, "ef_construction": 200, "quantization_bits": 0, "init": "pipnn"}
SEARCH = {"ef": inproc.REQUIRED}


def graph_metric(metric: str) -> str:
    """The library's metric for a dataset's, "l2" for cosine and L2 datasets and "ip" for inner product."""
    if metric in ("cosine", "l2"):
        return "l2"

    if metric == "ip":
        return "ip"

    raise Unsupported(f"SymphonyQG has no metric {metric}")


def stored_bytes(rows: int, dim: int, bits: int) -> int:
    """The bytes of the vectors that a visited row is scored with, raw float32 rows or codes in the padded dimension,
    without the neighbor codes and factors."""
    padded = -(-dim // 32) * 32
    return rows * dim * 4 if bits == 0 else rows * padded * bits // 8


class SymphonyQG(RabitqAdapter):
    system = "symphonyqg"
    file_name = "symqg.rabitq"
    kind = "SymqgIndex"
    # SymqgIndex has no add or remove, so it keeps the base's rebuild-only role in place of the writes the other
    # RaBitQ-Library indexes share.
    write_role = "rebuild"
    writer_open = Adapter.writer_open

    def overhead_config(self) -> dict:
        return {"degree": 32, "ef_construction": 64}

    def overhead_params(self) -> dict:
        return {"ef": 10}

    def check_search(self, params: dict) -> dict:
        out = inproc.settings(params, SEARCH, "symphonyqg search settings")

        if int(out["ef"]) < 1:
            raise ValueError("ef must be at least 1")

        return out

    def build(self, ctx: BuildContext) -> dict:
        cfg = inproc.settings(ctx.config, BUILD, "symphonyqg build settings")
        degree, bits, metric = int(cfg["degree"]), int(cfg["quantization_bits"]), graph_metric(ctx.metric)

        if ctx.clusters is not None:
            raise ValueError("SymphonyQG takes no partition")

        if degree % 32 or not 0 < degree < ctx.rows:
            raise ValueError(f"degree is {degree}, and it must be a multiple of 32 below the {ctx.rows} rows")

        if bits not in (0, 4, 8):
            raise ValueError(f"quantization_bits is {bits}, not 0, 4 or 8")

        if ctx.metric == "cosine":
            inproc.check_unit(ctx.data)

        index = self.lib.SymqgIndex(dim=ctx.dim, max_degree=degree, metric=metric, quantization_bits=bits)
        start = time.perf_counter()
        index.build(ctx.data, ef_construction=int(cfg["ef_construction"]), num_threads=ctx.threads,
                    init=str(cfg["init"]))
        encode = time.perf_counter() - start
        saved = self.save(index, ctx.out)
        return {
            "encode_seconds": encode,
            "stored_vector_bytes": stored_bytes(ctx.rows, ctx.dim, bits),
            "exact_f32_stored": bits == 0,
            "extra": {
                "degree": degree, "ef_construction": int(cfg["ef_construction"]), "quantization_bits": bits,
                "init": str(cfg["init"]), "metric": metric, "save_seconds": saved, "seeded": False,
                "library": {"num_points": int(index.num_points), "max_degree": int(index.max_degree),
                            "quantization_bits": int(index.quantization_bits), "metric": str(index.metric)},
            },
        }

    def query(self, index: Any, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        ef = int(params["ef"])

        if len(queries) == 1:
            ids, _ = index.search(queries, k=k, ef=ef, num_threads=1)
        else:
            ids, _ = index.search_batch(queries, k=k, ef=ef, num_threads=threads)

        return ids


if __name__ == "__main__":
    main(SymphonyQG)
