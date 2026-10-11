import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import helpers
from tests.faiss_files import flat, ilar, ivf_rabitq, pretransform, refine, sq8
from adapters import faiss_common as fc
from adapters import faiss_hnsw, faiss_ivf, faiss_layout
from adapters.base import BuildContext, ServeContext, Unsupported
import data
import driver
import evictor
import schema

HAS_FAISS = importlib.util.find_spec("faiss") is not None


class IVFSpecTest(unittest.TestCase):
    def test_defaults_and_list_counts(self):
        spec = faiss_ivf.Spec.parse({"vectors_per_list": 160})
        self.assertEqual((spec.bits, spec.refine, spec.train_rows, spec.chunk_rows), (1, "sq8", 500_000, 100_000))
        self.assertEqual(spec.lists(50_000), 312)
        self.assertEqual(spec.lists(1_000_000), 6250)
        self.assertEqual(spec.lists(10, None), 1)
        self.assertEqual(faiss_ivf.Spec.parse({"nlist": 7}).lists(50_000), 7)
        self.assertEqual(faiss_ivf.Spec.parse({}).lists(50_000, shared=99), 99)

    def test_bad_configs(self):
        for config in ({"nlist": 4, "vectors_per_list": 160}, {"bits": 0}, {"bits": 10}, {"refine": "pq"},
                       {"nprobe": 4}, {"train_rows": 0}):
            with self.assertRaises(ValueError, msg=config):
                faiss_ivf.Spec.parse(config)

        with self.assertRaises(ValueError):
            faiss_ivf.Spec.parse({}).lists(50_000)

        with self.assertRaises(ValueError):
            faiss_ivf.Spec.parse({"nlist": 5}).lists(50_000, shared=6)

    def test_factory_strings(self):
        self.assertEqual(faiss_ivf.factory_string(768, 6250, 1, "flat", 32, False), "RR768,IVF6250,RaBitQ")
        self.assertEqual(faiss_ivf.factory_string(768, 6250, 5, "flat", 32, False), "RR768,IVF6250,RaBitQ5")
        self.assertEqual(faiss_ivf.factory_string(768, 312, 1, "hnsw", 32, False), "RR768,IVF312_HNSW32,RaBitQ")
        self.assertEqual(faiss_ivf.factory_string(768, 312, 4, "flat", 32, True), "RR768,IVF312,RaBitQfs4")

    def test_variants_and_search_keys(self):
        self.assertEqual(faiss_ivf.variant_of(""), "shipped")
        self.assertEqual(faiss_ivf.variant_of("graph"), "graph")

        with self.assertRaises(ValueError):
            faiss_ivf.variant_of("fast")

        faiss_ivf.check_params({"nprobe": 8, "k_factor": 2, "qb": 8}, "shipped")
        faiss_ivf.check_params({"quantizer_ef": 64}, "graph")
        faiss_ivf.check_params({"prefetch_threads": 8}, "ondisk")

        for params, variant in (({"quantizer_ef": 64}, "shipped"), ({"prefetch_threads": 8}, "graph"),
                                ({"rerank": 9}, "batched")):
            with self.assertRaises(Unsupported):
                faiss_ivf.check_params(params, variant)

    def test_offsets_for_the_batched_variant(self):
        layout = faiss_ivf.faiss_layout.Layout(path="x", size=1000, parts=[
            faiss_ivf.faiss_layout.Part("coarse", 100, 50), faiss_ivf.faiss_layout.Part("lists", 200, 300),
            faiss_ivf.faiss_layout.Part("refine", 600, 384)], row_bytes={"refine": 8})
        layout.lists = faiss_ivf.faiss_layout.array_lists(200, np.array([1, 2]), 6)
        layout.lists.sizes_offset = 180
        off = faiss_ivf.offsets_record(layout)
        self.assertEqual(off, {"routing_end": 200, "lists_offset": 200, "sizes_offset": 180, "sizes_type": "full",
                               "refine_offset": 600, "refine_row_bytes": 8})

    def test_rescore(self):
        rows = np.array([[0, 0], [1, 0], [3, 0], [0.5, 0]], dtype=np.float32)
        ids = np.array([10, 11, 12, 13])
        q = np.array([0.9, 0], dtype=np.float32)
        self.assertEqual(faiss_ivf.rescore(rows, q, ids, 2, "l2").tolist(), [11, 13])
        self.assertEqual(faiss_ivf.rescore(rows, q, ids, 2, "ip").tolist(), [12, 11])


class HNSWSpecTest(unittest.TestCase):
    def test_defaults_per_storage(self):
        spec = faiss_hnsw.Spec.parse({})
        self.assertEqual((spec.m, spec.storage, spec.bits, spec.refine, spec.ef_construction),
                         (32, "sq8", None, "none", 200))
        spec = faiss_hnsw.Spec.parse({"storage": "rabitq"})
        self.assertEqual((spec.bits, spec.refine), (4, "sq8"))
        spec = faiss_hnsw.Spec.parse({"storage": "rabitq", "bits": 5, "refine": "none"})
        self.assertEqual((spec.bits, spec.refine), (5, "none"))

    def test_bad_configs(self):
        for config in ({"storage": "pq"}, {"bits": 4}, {"storage": "flat", "refine": "sq8"},
                       {"storage": "rabitq", "bits": 12}, {"m": 0}, {"efSearch": 16}):
            with self.assertRaises(ValueError, msg=config):
                faiss_hnsw.Spec.parse(config)


class CommonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_metric_and_simd(self):
        self.assertEqual(fc.faiss_metric("cosine"), "l2")
        self.assertEqual(fc.faiss_metric("l2"), "l2")
        self.assertEqual(fc.faiss_metric("ip"), "ip")

        with self.assertRaises(Unsupported):
            fc.faiss_metric("hamming")

        self.assertEqual(fc.compiled_simd("OPTIMIZE DD ARM_NEON MAC_METAL"), "ARM_NEON")
        self.assertEqual(fc.compiled_simd("AVX2 AVX512"), "AVX512")
        self.assertEqual(fc.compiled_simd("OPTIMIZE"), "NONE")

    def test_unit_check(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((50, 8)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        fc.check_unit(x, "cosine")
        fc.check_unit(2 * x, "l2")
        x[49] *= 1.01

        with self.assertRaises(ValueError):
            fc.check_unit(x, "cosine")

    def test_samples(self):
        x = np.arange(400, dtype=np.float32).reshape(100, 4)
        self.assertTrue(np.shares_memory(fc.sample_rows(x, 100, 1), x))
        a, b, c = (fc.sample_rows(x, 10, s) for s in (1, 1, 2))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertTrue(a.flags["C_CONTIGUOUS"] and a.dtype == np.float32)
        first = a[:, 0] / 4
        self.assertTrue(np.all(np.diff(first) > 0))
        self.assertEqual(list(fc.chunks(250, 100)), [(0, 100), (100, 200), (200, 250)])

    def test_merge_spans(self):
        spans = [(100, 10), (0, 10), (10, 5), (108, 20), (50, 0), (200, 1)]
        self.assertEqual(fc.merge_spans(spans), [(0, 15), (100, 28), (200, 1)])

    def test_partition(self):
        stem = self.dir / "part"
        data.write_fbin(f"{stem}.centroids.fbin", np.ones((3, 4), dtype=np.float32))
        data.write_ibin(f"{stem}.assign.ibin", np.array([[0], [2], [1], [2]], dtype=np.int32))
        part = fc.partition(stem, 3, 4)
        self.assertEqual(part.clusters, 3)
        self.assertEqual(part.assign.tolist(), [0, 2, 1])
        self.assertEqual(part.assign.dtype, np.int64)

        with self.assertRaises(ValueError):
            fc.partition(stem, 5, 4)

        with self.assertRaises(ValueError):
            fc.partition(stem, 3, 8)

    def test_read_ahead_and_residency(self):
        path = self.dir / "f.bin"
        path.write_bytes(os.urandom(4 * evictor.PAGE))

        with evictor.Evictor([path]) as ev:
            ev.make_cold(wait=0.0)
            fd = os.open(path, os.O_RDONLY)

            try:
                fc.read_ahead(fd, 0, 0)
                fc.read_ahead(fd, evictor.PAGE, evictor.PAGE)

                with open(path, "rb") as f:
                    f.read()
            finally:
                os.close(fd)

        parts = [{"file": "f.bin", "name": "head", "offset": 0, "length": 100},
                 {"file": "f.bin", "name": "tail", "offset": 2 * evictor.PAGE, "length": 2 * evictor.PAGE}]
        got = fc.resident_parts(self.dir, parts)
        self.assertEqual(got, {"head": evictor.PAGE, "tail": 2 * evictor.PAGE})

    def test_meta(self):
        fc.write_meta(self.dir, {"layout": "mapped", "rows": np.int64(5)})
        self.assertEqual(fc.read_meta(self.dir), {"layout": "mapped", "rows": 5})

        with self.assertRaises(FileNotFoundError):
            fc.read_meta(self.dir / "missing")


class FilesTest(unittest.TestCase):
    """files() of synthetic indexes, which needs no faiss."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        d, cs, n = 8, 6, 5
        centroids = np.ones((2, d), dtype=np.float32)
        lists = [np.array([0, 1, 2]), np.array([3, 4])]
        rows = np.zeros((n, d), dtype=np.uint8)
        inner = ivf_rabitq(d, centroids, ilar(cs, lists), n, cs)
        (self.dir / "index.faiss").write_bytes(refine(d, n, pretransform(d, n, inner), sq8(d, rows)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_mapped_index_marks_lists_and_rows_bulk(self):
        fc.write_meta(self.dir, {"layout": "mapped"})
        files = faiss_ivf.FaissIVF().files(self.dir)
        self.assertEqual([Path(f["path"]).name for f in files], ["index.faiss", "meta.json"])
        kinds = [(r["kind"], bool(r.get("loaded"))) for r in files[0]["ranges"]]
        # IO_FLAG_MMAP_IFC copies the rotation and maps the rest, so only the rotation is loaded.
        self.assertEqual(kinds, [("routing", False), ("routing", True), ("routing", False), ("bulk", False),
                                 ("routing", False), ("bulk", False), ("routing", False)])
        rotation = faiss_layout.read(self.dir / "index.faiss").part("rotation")
        loaded = [(r["offset"], r["length"]) for r in files[0]["ranges"] if r.get("loaded")]
        self.assertEqual(loaded, [(rotation.offset, rotation.length)])
        self.assertEqual((files[1]["ranges"][0]["kind"], files[1]["ranges"][0]["loaded"]), ("routing", True))
        self.assertEqual(sum(r["length"] for r in files[0]["ranges"]), files[0]["bytes"])

    def test_loaded_and_ondisk_files_are_bulk(self):
        fc.write_meta(self.dir, {"layout": "fastscan"})
        files = faiss_ivf.FaissIVF().files(self.dir)
        self.assertEqual(files[0]["ranges"], [{"offset": 0, "length": files[0]["bytes"], "kind": "bulk",
                                               "loaded": True}])
        (self.dir / "lists.ivfdata").write_bytes(b"x" * 64)
        (self.dir / "ivf.index").write_bytes((self.dir / "index.faiss").read_bytes())
        fc.write_meta(self.dir, {"layout": "ondisk"})
        files = {Path(f["path"]).name: f for f in faiss_ivf.FaissIVF().files(self.dir)}
        self.assertEqual(files["lists.ivfdata"]["ranges"], [{"offset": 0, "length": 64, "kind": "bulk"}])
        # read_index copies the whole of the on-disk layout's ivf.index.
        self.assertTrue(all(r.get("loaded") for r in files["ivf.index"]["ranges"]))
        self.assertEqual(sum(r["length"] for r in files["ivf.index"]["ranges"]), files["ivf.index"]["bytes"])

    def test_parts_record(self):
        (self.dir / "lists.ivfdata").write_bytes(b"x" * 64)
        parts = fc.parts_record(self.dir, ["index.faiss", "lists.ivfdata"])
        self.assertEqual([p["name"] for p in parts], ["rotation", "coarse", "list_sizes", "lists", "refine", "lists"])
        self.assertEqual(parts[-1], {"file": "lists.ivfdata", "name": "lists", "offset": 0, "length": 64,
                                     "kind": "bulk"})
        loaded = fc.parts_record(self.dir, ["index.faiss"], loaded=True)
        self.assertEqual([(p["name"], p["offset"]) for p in loaded], [("file", 0)])

    def test_hnsw_files(self):
        raw = flat(8, np.ones((3, 8), dtype=np.float32))
        (self.dir / "index.faiss").write_bytes(raw)
        files = faiss_hnsw.FaissHNSW().files(self.dir)
        self.assertEqual([r["kind"] for r in files[0]["ranges"]], ["routing", "bulk"])


@unittest.skipUnless(HAS_FAISS, "faiss isn't installed")
class AdapterTest(unittest.TestCase):
    """The adapters in this process on a small dataset, with faiss."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=2000, dim=32, nq=20)

    def tearDown(self):
        self.tmp.cleanup()

    def adapter(self, cls: type, variant: str) -> object:
        a = cls()
        a.variant = variant
        a.load()
        return a

    def build(self, a: object, config: dict, name: str, clusters: Path | None = None) -> Path:
        out = self.root / name
        out.mkdir()
        ctx = BuildContext(config=config, data_path=self.ds.base_path, rows=2000, out=out, clusters=clusters,
                           threads=2, seed=3, metric="cosine", variant=a.variant, dataset=self.ds)
        a.build(ctx)
        return out

    def serve(self, a: object, path: Path, params: dict) -> ServeContext:
        return ServeContext(index=path, dataset=self.ds, rows=2000, k=10, metric="cosine", threads=1,
                            variant=a.variant, params=params)

    def shared_clusters(self) -> tuple[Path, np.ndarray]:
        """12 clusters of random rows, so no row's cluster need be its nearest, as the stem of shared clusters."""
        rng = np.random.default_rng(4)
        assign = rng.integers(0, 12, 2000).astype(np.int32)
        stem = self.root / "shared"
        centroids = np.stack([np.asarray(self.ds.base[:2000])[assign == c].mean(axis=0) for c in range(12)])
        data.write_fbin(f"{stem}.centroids.fbin", centroids)
        data.write_ibin(f"{stem}.assign.ibin", assign[:, None])
        return stem, assign

    def test_shared_clusters_keep_their_assignment(self):
        import faiss

        stem, assign = self.shared_clusters()

        # The FastScan layout is filled as the plain one, here in adds of 300 rows, then copied into FastScan blocks.
        for variant, config in (("shipped", {}), ("fastscan", {"chunk_rows": 300})):
            a = self.adapter(faiss_ivf.FaissIVF, variant)
            out = self.build(a, config, f"shared-{variant}", clusters=stem)
            meta = fc.read_meta(out)
            self.assertEqual((meta["nlist"], meta["clusters"]), (12, str(stem)))
            h = a.open(self.serve(a, out, {"nprobe": 12}), out)

            for c in range(12):
                n = int(h.ivf.invlists.list_size(c))
                ids = faiss.rev_swig_ptr(h.ivf.invlists.get_ids(c), n).copy() if n else np.zeros(0)
                self.assertEqual(sorted(ids.tolist()), np.flatnonzero(assign == c).tolist(), variant)

            a.close(h)

    def test_fastscan_on_shared_clusters_finds_what_the_plain_layout_finds(self):
        # faiss's FastScan add_core stored codes outside the FastScan blocks, so a build on given lists found almost
        # none of the plain layout's results. Both probe every list here and rescore the same SQ8 rows.
        stem, _ = self.shared_clusters()
        params = {"nprobe": 12, "k_factor": 4}
        found = {}

        for variant in ("shipped", "fastscan"):
            a = self.adapter(faiss_ivf.FaissIVF, variant)
            out = self.build(a, {"chunk_rows": 300}, f"agree-{variant}", clusters=stem)
            h = a.open(self.serve(a, out, params), out)
            found[variant] = [set(a.search(h, np.asarray(q, dtype=np.float32), 10, params).tolist())
                              for q in self.ds.queries]
            a.close(h)

        same = sum(len(x & y) for x, y in zip(found["shipped"], found["fastscan"]))
        self.assertGreaterEqual(same, 0.9 * 10 * len(self.ds.queries))

    def test_batched_search_matches_the_shipped_search(self):
        shipped = self.adapter(faiss_ivf.FaissIVF, "shipped")
        out = self.build(shipped, {"vectors_per_list": 100}, "ivf")
        batched = self.adapter(faiss_ivf.FaissIVF, "batched")
        params = {"nprobe": 6, "k_factor": 4}
        a = shipped.open(self.serve(shipped, out, params), out)
        b = batched.open(self.serve(batched, out, params), out)
        same = 0

        for q in self.ds.queries:
            q = np.asarray(q, dtype=np.float32)
            same += len(set(shipped.search(a, q, 10, params)) & set(batched.search(b, q, 10, params)))

        self.assertGreaterEqual(same, 0.98 * 10 * len(self.ds.queries))
        stats = batched.stats(b)
        self.assertGreater(stats["bytes_requested"], 0)
        self.assertEqual(set(stats["stages"]) >= {"route_us", "lists_wave_us", "scan_us", "rows_wave_us"}, True)
        batched.close(b)
        self.assertEqual(b.waves.fd, -1)

    def test_c1_copies_the_coarse_quantizer(self):
        import faiss

        a = self.adapter(faiss_ivf.FaissIVF, "graph")
        out = self.build(a, {"vectors_per_list": 100}, "graph")
        params = {"nprobe": 8, "k_factor": 2, "quantizer_ef": 32}
        h = a.open(self.serve(a, out, params), out)
        q = np.asarray(self.ds.queries[0], dtype=np.float32)
        before = a.search(h, q, 10, params)
        mapped = faiss.downcast_index(h.ivf.quantizer)
        a.hold_routing(h)
        copied = faiss.downcast_index(h.ivf.quantizer)
        self.assertIsInstance(copied, faiss.IndexHNSWFlat)
        self.assertNotEqual(int(mapped.this), int(copied.this))
        self.assertEqual(copied.hnsw.efSearch, 32)
        self.assertEqual(a.search(h, q, 10, params).tolist(), before.tolist())

    def test_a_setting_without_a_key_searches_as_a_fresh_open(self):
        a = self.adapter(faiss_ivf.FaissIVF, "shipped")
        out = self.build(a, {"vectors_per_list": 100}, "ivf")
        h = a.open(self.serve(a, out, {"nprobe": 5, "k_factor": 4, "qb": 4}), out)
        fresh = a.open(self.serve(a, out, {"nprobe": 5}), out)
        a.configure(h, {"nprobe": 5})
        self.assertEqual((h.ivf.nprobe, h.k_factor, h.refine.k_factor, h.ivf.qb),
                         (5, fresh.k_factor, fresh.refine.k_factor, fresh.ivf.qb))
        self.assertEqual(h.k_factor, 1.0)
        a.configure(h, {})
        # The handle must stay alive while its index is read.
        blank = a.open(self.serve(a, out, {}), out)
        self.assertEqual(h.ivf.nprobe, blank.ivf.nprobe)
        g = self.adapter(faiss_hnsw.FaissHNSW, "")
        out = self.build(g, {"storage": "rabitq", "ef_construction": 40}, "hnsw-rq")
        h = g.open(self.serve(g, out, {"ef_search": 128, "k_factor": 4}), out)
        g.configure(h, {})
        self.assertEqual((h.ef_search, h.k_factor, h.keep[0].efSearch), (faiss_hnsw.DEFAULT_EF_SEARCH, 1.0, 16))
        g.configure(h, {"k_factor": 2})
        self.assertEqual((h.ef_search, h.k_factor), (faiss_hnsw.DEFAULT_EF_SEARCH, 2.0))

    def test_variant_must_match_the_index(self):
        a = self.adapter(faiss_ivf.FaissIVF, "shipped")
        out = self.build(a, {"vectors_per_list": 100}, "ivf")

        for variant in ("graph", "ondisk", "fastscan"):
            other = self.adapter(faiss_ivf.FaissIVF, variant)

            with self.assertRaises(Unsupported):
                other.open(self.serve(other, out, {}), out)

    def test_graph_settings(self):
        a = self.adapter(faiss_hnsw.FaissHNSW, "")
        out = self.build(a, {"storage": "sq8", "ef_construction": 40}, "hnsw")
        h = a.open(self.serve(a, out, {"ef_search": 64}), out)
        self.assertEqual(h.keep[0].efSearch, 64)

        with self.assertRaises(Unsupported):
            a.configure(h, {"k_factor": 2})

        with self.assertRaises(Unsupported):
            a.configure(h, {"nprobe": 2})

        q = np.asarray(self.ds.queries[0], dtype=np.float32)
        self.assertEqual(len(a.search(h, q, 10, {})), 10)
        stats = a.stats(h)
        self.assertGreater(stats["dist_evals"], 0)
        self.assertEqual(stats["bytes_scanned"], stats["dist_evals"] * 32)

    def test_levels_follow_the_seed(self):
        import faiss

        a = self.adapter(faiss_hnsw.FaissHNSW, "")
        levels = []

        for seed in (1, 2):
            out = self.root / f"seed{seed}"
            out.mkdir()
            ctx = BuildContext(config={"storage": "flat", "ef_construction": 16}, data_path=self.ds.base_path,
                               rows=2000, out=out, clusters=None, threads=2, seed=seed, metric="cosine",
                               variant="", dataset=self.ds)
            a.build(ctx)
            index = faiss.read_index(str(out / "index.faiss"))
            levels.append(faiss.vector_to_array(index.hnsw.levels))

        self.assertFalse(np.array_equal(levels[0], levels[1]))


SUITE = """
phase = "faiss"
dataset = "tiny-16-cosine"
n = 1500
index_root = "{root}/indexes"
results_root = "{root}/results"
seed = 2
max_attempts = 1
wait_s = 0.05

[gate]
mode = "off"
invalid_vm = []
"""

IVF = """
[[system]]
name = "faiss-ivf"
adapter = "faiss_ivf"
variant = "{variant}"

  [[system.build]]
  config_id = "vpl100"
  config = {{ vectors_per_list = 100 }}
  threads = 2

  [[system.warm]]
  config_id = "vpl100"
  split = "tune"
  passes = 1
  warmup = 4
  grid = {{ nprobe = [15], k_factor = [16] {more} }}

  [[system.cold]]
  config_id = "vpl100"
  params = {{ nprobe = 4, k_factor = 4 {more} }}
  trials = 2
  next = 2
  threads = {threads}

  [[system.cold]]
  config_id = "vpl100"
  regime = "c1"
  params = {{ nprobe = 4, k_factor = 4 {more} }}
  trials = 2
  group = "live"
  threads = {threads}
"""

HNSW = """
[[system]]
name = "faiss-hnsw"
adapter = "faiss_hnsw"
variant = "{storage}"

  [[system.build]]
  config_id = "{storage}"
  config = {{ storage = "{storage}", ef_construction = 64 }}
  threads = 2

  [[system.warm]]
  config_id = "{storage}"
  split = "tune"
  passes = 1
  warmup = 4
  grid = {{ ef_search = [128] {more} }}

  [[system.cold]]
  config_id = "{storage}"
  params = {{ ef_search = 64 {more} }}
  trials = 2
  next = 2

  [[system.cold]]
  config_id = "{storage}"
  regime = "c1"
  params = {{ ef_search = 64 {more} }}
  trials = 1
  group = "live"
"""


@unittest.skipUnless(HAS_FAISS, "faiss isn't installed")
class EndToEndTest(unittest.TestCase):
    """Every variant through the driver on a small dataset, with a build, a warm setting, and C0 and C1 trials."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        helpers.make_dataset(self.root / "data", n=1500, dim=32, nq=30)
        self.env = mock.patch.dict(os.environ, {"COLDIVF_DATA": str(self.root / "data")})
        self.env.start()
        self.quiet = mock.patch.object(driver.Driver, "log", lambda self, msg: None)
        self.quiet.start()

    def tearDown(self):
        self.quiet.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_every_variant(self):
        ivf = {"shipped": ("", 1), "ondisk": (", prefetch_threads = 4", 1), "parallel": ("", 3),
               "graph": (", quantizer_ef = 32", 1), "batched": ("", 1), "fastscan": (", qb = 8", 1)}
        text = SUITE.format(root=self.root)
        text += "".join(IVF.format(variant=v, more=m, threads=t) for v, (m, t) in ivf.items())
        text += "".join(HNSW.format(storage=s, more=m) for s, m in (("sq8", ""), ("rabitq", ", k_factor = 8"),
                                                                    ("flat", "")))
        path = self.root / "suite.toml"
        path.write_text(text)
        self.assertEqual(driver.Driver(driver.load_suite(path)).run(), 0)
        lines = [x for name in ("faiss-ivf", "faiss-hnsw")
                 for x in schema.read(self.root / "results" / "faiss" / f"{name}.jsonl")]
        bad = [(x["system"], x["variant"], x["kind"], x["error"]) for x in lines if x["status"] != "ok"]
        self.assertEqual(bad, [])
        self.assertEqual(sum(1 for x in lines if x["kind"] == "build"), 9)

        for x in lines:
            self.assertEqual(schema.validate(x), [])

        for warm in (x for x in lines if x["kind"] == "warm"):
            self.assertGreaterEqual(warm["recall10"], 0.9, (warm["variant"], warm["search_params"]))

        cold = [x for x in lines if x["kind"] == "cold"]
        self.assertEqual(len(cold), 6 * 4 + 3 * 3)

        for x in cold:
            self.assertEqual(x["resident_pages_at_go"], 0)
            self.assertIn("resident_after_first", x["extra"])
            self.assertEqual(len(x["ids"]), 10)

        c0 = [x for x in cold if x["regime"] == "c0"]
        self.assertTrue(all(x["files_touched"] >= 1 and x["open_us"] > 0 for x in c0))
        batched = [x for x in cold if x["variant"] == "batched"]
        self.assertTrue(all(x["bytes_requested"] > 0 and "lists_wave_us" in x["stages"] for x in batched))
        coarse = [x["extra"]["resident_after_first"].get("coarse", 0) for x in c0 if x["variant"] == "shipped"]
        self.assertTrue(all(c > 0 for c in coarse))
        meta = json.loads(next((self.root / "indexes").glob("faiss-ivf-ondisk/*/meta.json")).read_text())
        self.assertEqual(meta["files"], ["ivf.index", "lists.ivfdata", "sq8.index"])


if __name__ == "__main__":
    unittest.main()
