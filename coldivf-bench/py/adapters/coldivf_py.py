"""ColdIVF through its Python module, coldivf, in this process, as every other system's adapter runs its library, so its
times hold the module's per-call cost as theirs hold their bindings'. adapters/coldivf.py runs the release binaries
instead, which time each search inside their own process, and suits comparing ColdIVF with itself.

Build the module into the adapter's Python with `maturin develop --release` in coldivf-python. The adapter builds
indexes with the build_index binary, as adapters/coldivf.py does, so both search the same files, and opens and searches
them through the module. A cold trial times Index() and prepare() as the open and then the first search, as the cold
binary times Index::open, prepare and the search. A warm run starts the module's warm-up and waits for it, as the warm
binary's whole mode does.
"""

from __future__ import annotations

import importlib.metadata
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, main
from adapters.coldivf import ColdIVF, index_file
from adapters.inproc import as_ids
import schema

# The search settings the module takes, and the value of each a setting leaves out. A search needs nprobe and rerank.
SEARCH_KEYS = ("nprobe", "rerank", "reach")

# How long a warm run waits for the warm-up before it marks the index warm anyway, as the warm binary does.
WARM_TIMEOUT_S = 60.0


@dataclass
class Handle:
    index: Any
    nprobe: int
    rerank: int
    reach: float | None


def search_settings(params: dict) -> tuple[int, int, float | None]:
    unknown = sorted(set(params) - set(SEARCH_KEYS))

    if unknown:
        raise Unsupported(f"the coldivf module takes no search setting {unknown}")

    missing = [k for k in ("nprobe", "rerank") if params.get(k) is None]

    if missing:
        raise ValueError(f"a ColdIVF search needs {missing}")

    reach = params.get("reach")
    return int(params["nprobe"]), int(params["rerank"]), None if reach is None else float(reach)


class ColdIVFPython(Adapter):
    system = "coldivf"
    binding = "python"
    # C1 needs the module to drop the cached pages of its own mapping, which it can't yet.
    regimes = ("c0", "intent")
    # A cold search reads in F_RDADVISE waves from the calling thread, so it runs no read threads.
    workers = ""

    # The builds and the index's files are those of the binaries' adapter, so both adapters search the same files.
    binary = ColdIVF.binary
    env = ColdIVF.env
    run_build = ColdIVF.run_build
    files = ColdIVF.files

    def __init__(self) -> None:
        super().__init__()
        self.lib: Any = None

    def load(self) -> None:
        import coldivf

        self.lib = coldivf

    def module_file(self) -> Path:
        """The module's compiled file. maturin installs it in a package of the same name."""
        inner = getattr(self.lib, "coldivf", self.lib)
        return Path(inner.__file__)

    def version(self) -> str | None:
        try:
            return f"coldivf {importlib.metadata.version('coldivf')}"
        except importlib.metadata.PackageNotFoundError:
            return None

    def wheels(self) -> dict[str, str | None]:
        build_index = self.binary("build_index")
        return {"coldivf": schema.file_sha256(self.module_file()), "build_index": schema.file_sha256(build_index)}

    def simd_level(self) -> str:
        # The library picks its kernels at run time, which is NEON with dot products on Apple silicon.
        return "neon" if platform.machine() == "arm64" else "runtime"

    def build(self, ctx: BuildContext) -> dict:
        # The index of the per-call cost's measurement, built as every other.
        return ColdIVF.run_build(self, ctx)

    def overhead_params(self) -> dict:
        return {"nprobe": 2, "rerank": 10}

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        nprobe, rerank, reach = search_settings(ctx.params)
        index = self.lib.Index(str(index_file(path)), max_k=ctx.k, max_nprobe=nprobe, max_rerank=max(rerank, ctx.k))
        index.prepare()
        return Handle(index, nprobe, rerank, reach)

    def configure(self, handle: Handle, params: dict) -> None:
        handle.nprobe, handle.rerank, handle.reach = search_settings(params)

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        ids, _ = handle.index.search(query, k, nprobe=handle.nprobe, rerank=handle.rerank, reach=handle.reach)
        return as_ids(ids, k)

    def close(self, handle: Handle) -> None:
        handle.index = None

    def warm_up(self, handle: Handle, ctx: ServeContext) -> None:
        # The whole file read through the cache, then the module's warm-up, as the warm binary's whole mode does.
        super().warm_up(handle, ctx)

        if not handle.index.warm_up(wait=True, timeout=WARM_TIMEOUT_S):
            # The hits are the same either way, and the first searches fault their pages in.
            handle.index.set_warm(True)


if __name__ == "__main__":
    main(ColdIVFPython)
