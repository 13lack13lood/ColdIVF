"""faiss inverted file (IVF) indexes of RaBitQ codes, in the setup the variant names.

Every setup turns vectors with a random rotation of all their dimensions, eg. RR768 at 768 dimensions, routes a query
to its nearest coarse centroids, scans those clusters' lists of RaBitQ codes, and rescores the best k × k_factor
candidates against 8-bit scalar quantized (SQ8) rows.

| Variant | Files | Open | Search |
|---|---|---|---|
| shipped | index.faiss | read_index with IO_FLAG_MMAP_IFC copies the rotation and maps the rest | One thread faults in the coarse table, the probed lists and the candidates' rows as it reads them |
| ondisk | ivf.index, lists.ivfdata, sq8.index | The rotation and the coarse table load. Lists and rows stay mapped | Up to 32 threads read the probed lists ahead, set by prefetch_threads |
| parallel | index.faiss | As shipped | parallel_mode 1 splits the probed lists over the trial's threads, eg. 6 |
| graph | index.faiss | As shipped | The coarse quantizer is a hierarchical navigable small world (HNSW) graph, so routing reads only the centroids it visits |
| batched | index.faiss | As shipped, after one read-ahead of the file's routing part | It routes, reads the probed lists ahead in one wave, scans, then reads the candidates' rows ahead in a second wave and rescores |
| fastscan | index.faiss | Everything loads, with the codes in FastScan blocks of 32 | One thread |

The batched variant only attributes what batched reads gain. It isn't a setup faiss offers.

A build is split, so it holds little beyond the index. It trains the SQ8 rows and the IVF on one sample of at most 500K
rows. It fills the rows with a single add over the mapped base, which sizes them once, and adds to the IVF in chunks of
100K rows, each rotated on its own. Then it joins the two as an IndexRefine. At 10M rows that peaks at ~10 GB. With
--clusters the coarse centroids come from the shared partition, turned by the same rotation, and every row goes to its
cluster there instead of k-means. The FastScan setup is then filled as the plain one and copied into FastScan blocks,
since faiss's FastScan add can only place rows by their nearest centroid.

A true-cold trial (C0) times read_index and the first search. The disk-resident steady state (C1) copies the coarse
quantizer into memory before READY and keeps it there, and drops the coarse table's cached pages, which its searches no
longer read. files() marks the coarse table and every header as routing data, the lists and rows as bulk data, and
what an open copies into memory as loaded. Each cold trial also reports how many bytes of each part were cached right
after its first query, eg. how much of the coarse table it faulted in. A warm setting that leaves out a search setting
searches with its value from the open, as a fresh open would.

Writes go to the index loaded whole, since a mapped one can't take them, with the same calls in every layout that
takes them. An insert is add_with_ids on the IVF behind the rotation with the next ids, then add on the SQ8 rows, so
each id stays the position of the row its refine index rescores it with. A delete is remove_ids with an
IDSelectorBatch on the IVF, which moves each list's last entries into the holes and leaves the SQ8 rows. A search
finds a write at once. The durable point saves the whole file and rewrites meta.json with the new parts and list
offsets, so the mapped open, the batched variant and FastScan's open search a state as they search a build. The
on-disk layout takes no writes, since an insert into its lists corrupts them.

FastScan's remove leaves the removed rows' codes in the slots it frees in a list's last block, and faiss 1.15.1's next
add into that list packs its own codes over them without clearing them first, so a row added after a delete searches
as another vector. A write entry's params may set clear_removed, which packs zero codes into those slots after each
delete. It is off by default, since it calls the lists' own code packer, which faiss doesn't document for this.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adapters import faiss_layout
from adapters.base import BuildContext, ServeContext, Unsupported, WriteContext, main, now_ns
import evictor
from adapters.faiss_common import (
    CHUNK_ROWS, META, TRAIN_ROWS, FaissAdapter, Partition, Writer, check_unit, chunks, faiss_metric, merge_spans,
    metric_type, parsed, parts_record, partition, read_ahead, read_meta, sample_rows, set_threads, whole,
    write_meta,
)
from adapters.inproc import settings

# Each variant's file layout and coarse quantizer.
VARIANTS = {
    "shipped": ("mapped", "flat"),
    "ondisk": ("ondisk", "flat"),
    "parallel": ("mapped", "flat"),
    "graph": ("mapped", "hnsw"),
    "batched": ("mapped", "flat"),
    "fastscan": ("fastscan", "flat"),
}
DEFAULT_VARIANT = "shipped"

INDEX = "index.faiss"
ONDISK_FILES = ("ivf.index", "lists.ivfdata", "sq8.index")

BUILD_KEYS = {"nlist", "vectors_per_list", "bits", "refine", "train_rows", "chunk_rows", "quantizer_m",
              "quantizer_ef_construction"}
REFINES = ("sq8", "none")
MAX_BITS = 9

SEARCH_KEYS = {"nprobe", "k_factor", "qb"}
# How many threads read the probed on-disk lists ahead, faiss's own default, set explicitly when a setting leaves it out.
PREFETCH_THREADS = 32
VARIANT_SEARCH_KEYS = {"graph": {"quantizer_ef"}, "ondisk": {"prefetch_threads"}}


@dataclass
class Spec:
    """A build's settings, from its config."""

    nlist: int | None = None
    vectors_per_list: int | None = None
    bits: int = 1
    refine: str = "sq8"
    train_rows: int = TRAIN_ROWS
    chunk_rows: int = CHUNK_ROWS
    quantizer_m: int = 32
    quantizer_ef_construction: int | None = None

    @classmethod
    def parse(cls, config: dict) -> Spec:
        unknown = sorted(set(config) - BUILD_KEYS)

        if unknown:
            raise ValueError(f"the faiss IVF build takes no {unknown}")

        spec = cls(**config)

        if spec.nlist is not None and spec.vectors_per_list is not None:
            raise ValueError("give nlist or vectors_per_list, not both")

        if not 1 <= int(spec.bits) <= MAX_BITS:
            raise ValueError(f"RaBitQ takes 1 to {MAX_BITS} bits, not {spec.bits}")

        if spec.refine not in REFINES:
            raise ValueError(f"refine is one of {REFINES}, not {spec.refine!r}")

        for name in ("nlist", "vectors_per_list", "train_rows", "chunk_rows", "quantizer_m"):
            value = getattr(spec, name)

            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be at least 1, not {value}")

        return spec

    def lists(self, rows: int, shared: int | None = None) -> int:
        """The cluster count. A shared partition sets it, then nlist, then rows / vectors_per_list rounded."""
        if shared is not None:
            if self.nlist is not None and int(self.nlist) != shared:
                raise ValueError(f"nlist {self.nlist} doesn't match the {shared} shared clusters")

            return shared

        if self.nlist is not None:
            return int(self.nlist)

        if self.vectors_per_list is None:
            raise ValueError("the faiss IVF build needs nlist, vectors_per_list or shared clusters")

        return max(1, round(rows / int(self.vectors_per_list)))


def factory_string(dim: int, nlist: int, bits: int, quantizer: str, quantizer_m: int, fastscan: bool) -> str:
    """The faiss index_factory string of the rotated IVF, eg. "RR768,IVF6250,RaBitQ" or
    "RR768,IVF6250_HNSW32,RaBitQfs4"."""
    coarse = f"_HNSW{quantizer_m}" if quantizer == "hnsw" else ""
    codes = "RaBitQfs" if fastscan else "RaBitQ"
    return f"RR{dim},IVF{nlist}{coarse},{codes}{bits if bits > 1 else ''}"


def variant_of(label: str) -> str:
    name = label or DEFAULT_VARIANT

    if name not in VARIANTS:
        raise ValueError(f"the faiss IVF adapter has no variant {name!r}, only {sorted(VARIANTS)}")

    return name


def search_keys(variant: str) -> set[str]:
    return SEARCH_KEYS | VARIANT_SEARCH_KEYS.get(variant, set())


def check_params(params: dict, variant: str) -> None:
    unknown = sorted(set(params) - search_keys(variant))

    if unknown:
        raise Unsupported(f"the {variant} variant takes no search setting {unknown}")


@dataclass
class Waves:
    """What the batched variant keeps to read lists and rows ahead. The lists sit in the index file, each list's codes
    and then its ids."""

    fd: int
    rotation: Any
    lists: faiss_layout.Lists
    refine_index: Any
    refine_offset: int
    row_bytes: int
    metric: str
    # Bytes asked for at open that the first search reports.
    pending: int = 0
    stages: dict = field(default_factory=dict)
    requested: int = 0


@dataclass
class Handle:
    """An open IVF index, the parts of it that searches set, and the batched variant's state."""

    top: Any
    ivf: Any
    refine: Any = None
    # Objects that must live as long as the index, eg. the parts of an IndexRefine built here.
    keep: list = field(default_factory=list)
    code_size: int = 0
    refine_bytes: int = 0
    k_factor: float = 1.0
    k_base: int = 0
    waves: Waves | None = None
    # The search settings as read_index left them, which a setting that leaves one out gets back.
    fresh: dict = field(default_factory=dict)


def fastscan_of(ivf: Any, rotation: Any) -> tuple[Any, Any]:
    """A FastScan copy of the plain IVF RaBitQ index `ivf` behind `rotation`, with the same lists, its codes regrouped
    into blocks of 32. faiss 1.15.1's FastScan add_core stores codes outside those blocks, so a FastScan index filled
    on given lists through it searches as noise, and crashes once a call holds ~3.4 MB of codes. The copy points at
    the plain index's coarse quantizer and rotation, so the plain index must outlive it."""
    import faiss

    fast = faiss.IndexIVFRaBitQFastScan(ivf)
    top = faiss.IndexPreTransform(rotation, fast)
    top.ntotal = fast.ntotal
    return top, fast


def add_assigned(ivf: Any, rotated: np.ndarray, lists: np.ndarray, start: int) -> None:
    """Adds rotated rows to their given lists, with ids from `start`. faiss.contrib's add_preassigned frees a
    temporary ids array before faiss reads it, so the arrays stay named here until add_core returns."""
    import faiss

    x = np.ascontiguousarray(rotated, dtype=np.float32)
    ids = np.arange(start, start + len(x), dtype=np.int64)
    assign = np.ascontiguousarray(lists, dtype=np.int64)
    ivf.add_core(len(x), faiss.swig_ptr(x), faiss.swig_ptr(ids), faiss.swig_ptr(assign))


def list_sizes(ivf: Any) -> np.ndarray:
    lists = ivf.invlists
    return np.array([lists.list_size(i) for i in range(ivf.nlist)], dtype=np.int64)


def clear_vacated(ivf: Any, before: np.ndarray) -> int:
    """Packs zero codes into the slots that a remove left in the last block of each FastScan list it shrank, and
    returns how many it cleared. faiss 1.15.1 leaves the removed rows' codes in them, and its next add into the list
    packs its own codes over those without clearing them, so without this a row added after a delete searches as
    another vector."""
    import faiss

    lists = faiss.downcast_InvertedLists(ivf.invlists)
    per, size = int(lists.n_per_block), int(lists.block_size)
    zero = np.zeros(int(lists.packer.code_size), dtype=np.uint8)
    after = list_sizes(ivf)
    cleared = 0

    for no in np.flatnonzero(after < before).tolist():
        n = int(after[no])

        if n % per == 0:
            continue

        blocks = faiss.rev_swig_ptr(lists.get_codes(no), (n // per + 1) * size).reshape(-1, size)

        for slot in range(n % per, per):
            lists.packer.pack_1(zero, slot, blocks[-1:])

        cleared += per - n % per

    return cleared


def offsets_record(layout: faiss_layout.Layout) -> dict:
    """The offsets the batched variant needs, from the layout of index.faiss."""
    lists, rows = layout.lists, layout.part("refine")
    first = layout.part("lists")
    return {
        "routing_end": first.offset if first else layout.size,
        "lists_offset": first.offset if first else None,
        "sizes_offset": lists.sizes_offset if lists else None,
        "sizes_type": lists.sizes_type if lists else None,
        "refine_offset": rows.offset if rows else None,
        "refine_row_bytes": layout.row_bytes.get("refine"),
    }


class FaissIVF(FaissAdapter):
    system = "faiss-ivf"

    def load(self) -> None:
        super().load()
        import faiss

        self.name = variant_of(self.variant)
        self.layout, self.quantizer = VARIANTS[self.name]
        self._stats = faiss.cvar.indexIVF_stats
        # An insert into faiss's on-disk lists corrupts them, so that layout takes the stream as a fresh build.
        self.write_role = "rebuild" if self.layout == "ondisk" else "full"

        if self.name == "batched":
            # Its search is a Python routine over one query, so QPS runs one process per worker.
            self.batch_api = False
            self.qps_mode = "processes"
            self.workers = ""
        elif self.name == "parallel":
            self.workers = "OpenMP's pool, which also splits each query's lists"
        elif self.name == "ondisk":
            self.workers = "OpenMP's pool for batch searches, and the threads that read the on-disk lists ahead"

    def overhead_config(self) -> dict:
        return {"nlist": 4}

    def overhead_params(self) -> dict:
        return {"nprobe": 1}

    def build(self, ctx: BuildContext) -> dict:
        import faiss

        set_threads(ctx.threads)
        spec = Spec.parse(ctx.config)
        metric = faiss_metric(ctx.metric)
        rows = ctx.data
        check_unit(rows, ctx.metric)
        shared = partition(ctx.clusters, ctx.rows, ctx.dim) if ctx.clusters is not None else None
        nlist = spec.lists(ctx.rows, shared.clusters if shared else None)
        factory = factory_string(ctx.dim, nlist, int(spec.bits), self.quantizer, int(spec.quantizer_m),
                                 self.layout == "fastscan")
        # On given lists the FastScan layout is filled as the plain one, then copied into FastScan blocks.
        convert = self.layout == "fastscan" and shared is not None
        built = factory_string(ctx.dim, nlist, int(spec.bits), self.quantizer, int(spec.quantizer_m), False) \
            if convert else factory
        index = faiss.index_factory(ctx.dim, built, metric_type(metric))
        rotation = faiss.downcast_VectorTransform(index.chain.at(0))
        rotation.init(ctx.seed)
        ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        ivf.cp.seed = ctx.seed

        if self.quantizer == "hnsw" and spec.quantizer_ef_construction is not None:
            faiss.downcast_index(ivf.quantizer).hnsw.efConstruction = int(spec.quantizer_ef_construction)

        sample = sample_rows(rows, int(spec.train_rows), ctx.seed)
        trained_rows = len(sample)
        t0 = time.perf_counter()

        if shared is not None:
            ivf.quantizer.add(rotation.apply(shared.centroids))

        # With shared centroids in place, faiss skips k-means and trains only the codes.
        index.train(sample)
        t1 = time.perf_counter()
        rows_index = None

        if spec.refine == "sq8":
            rows_index = faiss.IndexScalarQuantizer(ctx.dim, faiss.ScalarQuantizer.QT_8bit, metric_type(metric))
            rows_index.train(sample)

        del sample
        empty = faiss.serialize_index(index) if self.layout == "ondisk" else None
        self.add(index, ivf, rotation, rows, int(spec.chunk_rows), shared)
        plain = None

        if convert:
            plain = index
            index, ivf = fastscan_of(ivf, rotation)

        if rows_index is not None:
            # One add over the whole mapped base sizes the rows once, where adds in chunks would regrow them.
            rows_index.add(np.ascontiguousarray(rows, dtype=np.float32))

        t2 = time.perf_counter()
        names = self.write(ctx.out, index, rows_index, empty)
        del plain
        parts = parts_record(ctx.out, names, loaded=self.layout == "fastscan")

        for p in parts:
            if p["file"] == "sq8.index" and p["name"] == "codes":
                p["name"] = "refine"

        refine_bytes = int(rows_index.code_size) if rows_index is not None else 0
        record = {
            "system": self.system, "variant": self.name, "layout": self.layout, "quantizer": self.quantizer,
            "factory": factory, "config": ctx.config, "metric": ctx.metric, "faiss_metric": metric, "dim": ctx.dim,
            "rows": ctx.rows, "seed": ctx.seed, "nlist": nlist, "bits": int(spec.bits),
            "code_size": int(ivf.code_size), "refine": spec.refine, "refine_bytes": refine_bytes,
            "clusters": str(ctx.clusters) if ctx.clusters is not None else None, "train_rows": trained_rows,
            "faiss_version": faiss.__version__, "files": names, "parts": parts,
        }

        if self.layout == "mapped":
            record["offsets"] = offsets_record(faiss_layout.read(ctx.out / INDEX))

        write_meta(ctx.out, record)
        return {
            "cluster_seconds": t1 - t0 if shared is None else None,
            "encode_seconds": t2 - t1,
            "stored_vector_bytes": ctx.rows * (int(ivf.code_size) + refine_bytes),
            "exact_f32_stored": False,
            "extra": {
                "factory": factory, "nlist": nlist, "code_size": int(ivf.code_size), "refine_bytes": refine_bytes,
                "list_bytes": ctx.rows * (int(ivf.code_size) + 8), "train_rows": trained_rows,
                "train_seconds": t1 - t0, "clusters_imported": record["clusters"], "layout": self.layout,
                "quantizer": self.quantizer, "faiss_metric": metric, "filled_as": built,
            },
        }

    def add(self, index: Any, ivf: Any, rotation: Any, rows: np.ndarray, size: int, shared: Partition | None) -> None:
        """Adds every row in chunks, so only one chunk's rotated copy exists at a time."""
        for lo, hi in chunks(len(rows), size):
            chunk = np.ascontiguousarray(rows[lo:hi], dtype=np.float32)

            if shared is None:
                index.add(chunk)
            else:
                add_assigned(ivf, rotation.apply(chunk), shared.assign[lo:hi], lo)

        # Adds straight to the IVF leave the rotation's own count behind.
        index.ntotal = ivf.ntotal

    def write(self, out: Path, index: Any, rows_index: Any, empty: Any) -> list[str]:
        """Writes the index files of this layout and returns their names."""
        import faiss
        from faiss.contrib.ondisk import merge_ondisk

        if self.layout != "ondisk":
            top = faiss.IndexRefine(index, rows_index) if rows_index is not None else index
            faiss.write_index(top, str(out / INDEX))
            return [INDEX]

        # The on-disk lists come from a filled copy written as a shard, merged into a trained empty copy. The shard's
        # name marks it as a temporary file the driver removes if the build dies.
        shard = out / f"shard.{os.getpid()}.0.tmp"
        faiss.write_index(index, str(shard))
        trained = faiss.deserialize_index(empty)
        merge_ondisk(trained, [str(shard)], str(out / ONDISK_FILES[1]))
        faiss.write_index(trained, str(out / ONDISK_FILES[0]))
        shard.unlink()
        names = list(ONDISK_FILES[:2])

        if rows_index is not None:
            faiss.write_index(rows_index, str(out / ONDISK_FILES[2]))
            names.append(ONDISK_FILES[2])

        return names

    def prepared(self, path: Path) -> dict:
        """The meta file of the index at `path`, checked against this variant."""
        meta = self.meta(path)

        if (meta.get("layout"), meta.get("quantizer")) != (self.layout, self.quantizer):
            raise Unsupported(f"the {self.name} variant needs a {self.layout} index with a {self.quantizer} coarse "
                              f"quantizer, and {path} holds a {meta.get('layout')} one with {meta.get('quantizer')}")

        return meta

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        import faiss

        path = Path(path)
        meta = self.prepared(path)
        keep: list = []
        fd = None

        if self.name == "batched":
            # One read-ahead of the headers, the rotation, the coarse table and the list sizes, which read_index and
            # the first route read next.
            fd = os.open(path / INDEX, os.O_RDONLY)
            read_ahead(fd, 0, int(meta["offsets"]["routing_end"]))

        if self.layout == "ondisk":
            top = faiss.read_index(str(path / ONDISK_FILES[0]), faiss.IO_FLAG_ONDISK_SAME_DIR)

            if meta["refine"] == "sq8":
                rows = faiss.read_index(str(path / ONDISK_FILES[2]), faiss.IO_FLAG_MMAP_IFC)
                keep += [top, rows]
                top = faiss.IndexRefine(top, rows)
        elif self.layout == "fastscan":
            top = faiss.read_index(str(path / INDEX))
        else:
            top = faiss.read_index(str(path / INDEX), faiss.IO_FLAG_MMAP_IFC)

        refine = top if isinstance(top, faiss.IndexRefine) else None
        base = faiss.downcast_index(refine.base_index) if refine is not None else top
        ivf = faiss.downcast_index(faiss.extract_index_ivf(base))
        keep.append(base)
        h = Handle(top=top, ivf=ivf, refine=refine, keep=keep, code_size=int(meta["code_size"]),
                   refine_bytes=int(meta.get("refine_bytes") or 0))
        h.fresh = {"nprobe": int(ivf.nprobe), "qb": getattr(ivf, "qb", None),
                   "k_factor": float(refine.k_factor) if refine is not None else 1.0,
                   "quantizer_ef": (int(faiss.downcast_index(ivf.quantizer).hnsw.efSearch)
                                    if self.quantizer == "hnsw" else None)}

        if self.name == "parallel":
            ivf.parallel_mode = 1

        if fd is not None:
            h.waves = self.waves(fd, base, refine, meta)

        self.configure(h, ctx.params)
        return h

    def waves(self, fd: int, base: Any, refine: Any, meta: dict) -> Waves:
        """The batched variant's table of list offsets. read_index has just read the list sizes, so reading them again
        here comes from the cache."""
        import faiss

        off = meta["offsets"]

        if off.get("sizes_type") != "full":
            raise Unsupported("the batched variant needs a full table of list sizes")

        nlist = int(meta["nlist"])
        raw = os.pread(fd, nlist * 8, int(off["sizes_offset"]))
        sizes = np.frombuffer(raw, dtype="<u8").astype(np.int64)
        lists = faiss_layout.array_lists(int(off["lists_offset"]), sizes, int(meta["code_size"]))
        rows = faiss.downcast_index(refine.refine_index) if refine is not None else None
        return Waves(fd=fd, rotation=faiss.downcast_VectorTransform(base.chain.at(0)), lists=lists,
                     refine_index=rows, refine_offset=int(off["refine_offset"] or 0),
                     row_bytes=int(off["refine_row_bytes"] or 0), metric=meta["faiss_metric"],
                     pending=int(off["routing_end"]))

    def configure(self, h: Handle, params: dict) -> None:
        """Applies a search setting. A key the setting leaves out gets its value from the open, so a warm setting
        searches exactly as a fresh open with the same settings does, whatever setting ran before it."""
        import faiss

        check_params(params, self.name)
        values = h.fresh | {k: v for k, v in params.items() if v is not None}
        h.ivf.nprobe = int(values["nprobe"])
        k_factor = float(values["k_factor"])

        if h.refine is None and k_factor != 1:
            raise Unsupported("an index without refine rows takes only k_factor 1")

        if h.refine is not None:
            h.refine.k_factor = k_factor

        h.k_factor = k_factor

        if values.get("qb") is not None:
            h.ivf.qb = int(values["qb"])

        if values.get("quantizer_ef") is not None:
            faiss.downcast_index(h.ivf.quantizer).hnsw.efSearch = int(values["quantizer_ef"])

        if self.layout == "ondisk":
            threads = params.get("prefetch_threads")
            lists = faiss.downcast_InvertedLists(h.ivf.invlists)
            lists.prefetch_nthread = PREFETCH_THREADS if threads is None else int(threads)

    def hold_routing(self, h: Handle) -> None:
        """Copies a mapped coarse quantizer into memory, which C1 keeps for every query. Searches then read the copy,
        so the coarse table's cached pages are dropped, and the memory C1's state costs is counted once. The other
        layouts load it at open."""
        import faiss
        from faiss.contrib.ivf_tools import replace_ivf_quantizer

        if self.layout != "mapped":
            return

        copy = faiss.deserialize_index(faiss.serialize_index(h.ivf.quantizer))
        old = replace_ivf_quantizer(h.ivf, copy)
        h.keep += [copy, old]

        if self._serving is not None:
            parts = [p for p in self.meta(self._serving).get("parts") or []
                     if p["file"] == INDEX and p["name"] in ("coarse", "coarse_graph")]
            ranges = [{"offset": p["offset"], "length": p["length"], "kind": p["kind"], "loaded": True} for p in parts]
            evictor.drop_loaded([{"path": str(Path(self._serving) / INDEX), "bytes": 0, "ranges": ranges}])

    def search(self, h: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        self._stats.reset()

        if h.waves is not None:
            return self.search_waves(h, query, k)

        h.k_base = int(k * h.k_factor) if h.refine is not None else 0
        _, ids = h.top.search(query.reshape(1, -1), k)
        return ids[0]

    def search_waves(self, h: Handle, query: np.ndarray, k: int) -> np.ndarray:
        """Routes, reads the probed lists ahead, scans them, then reads the candidates' rows ahead and rescores."""
        w = h.waves
        q = query.reshape(1, -1)
        t0 = now_ns()
        rotated = w.rotation.apply(q)
        coarse, probed = h.ivf.quantizer.search(rotated, h.ivf.nprobe)
        t1 = now_ns()
        lists = merge_spans(w.lists.spans(probed[0]))

        for offset, length in lists:
            read_ahead(w.fd, offset, length)

        t2 = now_ns()
        h.k_base = int(k * h.k_factor) if h.refine is not None else k
        _, found = h.ivf.search_preassigned(rotated, max(k, h.k_base), probed, coarse)
        found = found[0][found[0] >= 0]
        t3 = now_ns()
        rows: list[tuple[int, int]] = []

        if w.refine_index is not None and len(found):
            rows = merge_spans([(w.refine_offset + int(i) * w.row_bytes, w.row_bytes) for i in found])

            for offset, length in rows:
                read_ahead(w.fd, offset, length)

        t4 = now_ns()

        if w.refine_index is not None and len(found):
            found = rescore(w.refine_index.reconstruct_batch(found), q[0], found, k, w.metric)

        t5 = now_ns()
        out = np.full(k, -1, dtype=np.int64)
        out[: min(k, len(found))] = found[:k]
        w.stages = {"route_us": (t1 - t0) / 1e3, "lists_wave_us": (t2 - t1) / 1e3, "scan_us": (t3 - t2) / 1e3,
                    "rows_wave_us": (t4 - t3) / 1e3, "rescore_us": (t5 - t4) / 1e3, "list_ranges": len(lists),
                    "row_ranges": len(rows)}
        w.requested = w.pending + sum(n for _, n in lists) + sum(n for _, n in rows)
        w.pending = 0
        return out

    def search_batch(self, h: Handle, queries: np.ndarray, k: int, params: dict, threads: int) -> np.ndarray:
        if h.waves is not None:
            raise Unsupported("the batched variant searches one query at a time")

        set_threads(threads)
        return h.top.search(np.ascontiguousarray(queries, dtype=np.float32), k)[1]

    def stats(self, h: Handle) -> dict:
        s = self._stats
        out: dict = {
            "dist_evals": float(s.ndis),
            "lists_visited": int(s.nlist),
            "bytes_scanned": float(s.ndis * h.code_size + h.k_base * h.refine_bytes),
            "stages": {"route_us": s.quantization_time * 1e3, "scan_us": s.search_time * 1e3},
        }

        if h.waves is not None:
            out["stages"] = dict(h.waves.stages)
            out["bytes_requested"] = h.waves.requested

        out.update(self.first_query_parts())
        return out

    def close(self, h: Handle) -> None:
        if h.waves is not None and h.waves.fd >= 0:
            os.close(h.waves.fd)
            h.waves.fd = -1

    def files(self, path: Path) -> list[dict]:
        """The index's files. What an open copies into memory is also "loaded". That is the whole file of a FastScan
        index, the on-disk layout's ivf.index with its rotation, coarse table and lists' table, and the rotation of a
        file read with IO_FLAG_MMAP_IFC, which maps the rest. The meta file is read once before any trial."""
        path = Path(path)

        if path.is_file():
            return [parsed(path)]

        layout = read_meta(path).get("layout") if (path / META).is_file() else None
        out = []

        for f in sorted(p for p in path.iterdir() if p.is_file()):
            if f.name == META:
                out.append(whole(f, "routing", loaded=True))
            elif f.suffix == ".ivfdata":
                out.append(whole(f, "bulk"))
            elif layout == "fastscan":
                out.append(whole(f, "bulk", loaded=True))
            elif layout == "ondisk" and f.name == ONDISK_FILES[0]:
                entry = parsed(f)
                out.append(entry | {"ranges": [r | {"loaded": True} for r in entry["ranges"]]})
            else:
                out.append(parsed(f, frozenset({"rotation"})))

        return out

    def writer_open(self, ctx: WriteContext, path: Path) -> Writer:
        """Reads the index whole, without a map, since faiss writes only to a loaded index. The write entry's params
        may set clear_removed for FastScan, which is off by default."""
        import faiss

        if self.layout == "ondisk":
            raise Unsupported("an insert into faiss's on-disk lists corrupts them, so that layout takes no writes")

        path = Path(path)
        meta = self.prepared(path)
        options = settings(ctx.params, {"clear_removed": False}, "faiss IVF write settings")

        if meta.get("refine") != "sq8":
            raise Unsupported("an insert takes the id of its SQ8 row, so an index without SQ8 rows takes no writes")

        if options["clear_removed"] and self.layout != "fastscan":
            raise ValueError("clear_removed mends FastScan's lists, and the plain lists need no mending")

        top = faiss.read_index(str(path / INDEX))
        pre = faiss.downcast_index(top.base_index)
        return Writer(folder=path, file=path / INDEX, top=top, meta=meta, pre=pre,
                      ivf=faiss.downcast_index(faiss.extract_index_ivf(pre)),
                      rows=faiss.downcast_index(top.refine_index), clear_removed=bool(options["clear_removed"]))

    def insert(self, w: Writer, ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Adds the rows to the IVF behind the rotation with the next ids, then to the SQ8 rows, so each id is the
        position of the row the refine index rescores it with. It returns those ids, which a stream checks."""
        x = np.ascontiguousarray(rows, dtype=np.float32)
        given = np.arange(w.rows.ntotal, w.rows.ntotal + len(x), dtype=np.int64)
        w.pre.add_with_ids(x, given)
        w.rows.add(x)
        w.top.ntotal = w.rows.ntotal
        return given

    def delete(self, w: Writer, ids: np.ndarray) -> None:
        """Removes the ids from their lists, which moves each list's last entries into the holes. Their SQ8 rows stay,
        since the refine index finds a row by its position."""
        import faiss

        ids = np.ascontiguousarray(ids, dtype=np.int64)
        before = list_sizes(w.ivf) if w.clear_removed else None
        removed = w.pre.remove_ids(faiss.IDSelectorBatch(len(ids), faiss.swig_ptr(ids)))

        if removed != len(ids):
            raise RuntimeError(f"faiss removed {removed} of {len(ids)} ids, so the others weren't in the index")

        if before is not None:
            w.cleared += clear_vacated(w.ivf, before)

    def pending(self, w: Writer) -> dict:
        """What the loaded index holds. faiss puts each insert in its list at once and takes each delete out of it, so
        nothing waits, but deleted ids keep their SQ8 rows, and the lists grow unevenly."""
        sizes = list_sizes(w.ivf)
        out = {"live_rows": int(w.ivf.ntotal), "state_rows": int(w.rows.ntotal),
               "dead_rows": int(w.rows.ntotal - w.ivf.ntotal), "largest_list": int(sizes.max(initial=0)),
               "imbalance": float(w.ivf.invlists.imbalance_factor())}

        if w.clear_removed:
            out["cleared_slots"] = w.cleared

        return out

    def saved_meta(self, w: Writer) -> dict:
        """The meta record of the saved index: its new parts, the list offsets the batched variant reads, and its ids
        and live rows."""
        meta = dict(w.meta)
        meta["parts"] = parts_record(w.folder, [INDEX], loaded=self.layout == "fastscan")

        if self.layout == "mapped":
            meta["offsets"] = offsets_record(faiss_layout.read(w.file))

        meta.update(state_rows=int(w.rows.ntotal), live_rows=int(w.ivf.ntotal))
        return meta


def rescore(rows: np.ndarray, query: np.ndarray, ids: np.ndarray, k: int, metric: str) -> np.ndarray:
    """The ids of the k best candidates by their decoded rows, nearest first, as IndexRefine ranks them."""
    if metric == "l2":
        diff = rows - query
        score = np.einsum("ij,ij->i", diff, diff)
    else:
        score = -(rows @ query)

    return ids[np.argsort(score, kind="stable")[:k]]


if __name__ == "__main__":
    main(FaissIVF)
