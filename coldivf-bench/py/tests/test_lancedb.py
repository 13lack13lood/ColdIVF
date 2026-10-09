import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters import lancedb
from adapters.base import ServeContext, Unsupported

# analyze_plan's output for one query with nprobes 32 and refine_factor 4, from pylance 13 on a 50K table.
ANALYZED = """\
AnalyzeExec verbose=true, elapsed=3.25ms, metrics=[]
  TracedExec, elapsed=3.25ms, metrics=[]
    ProjectionExec: elapsed=3.25ms, expr=[id@3 as id, _distance@2 as _distance], metrics=[output_rows=10, \
elapsed_compute=1.00µs, output_bytes=32.1 KB, output_batches=1, expr_0_eval_time=250ns, expr_1_eval_time=83ns]
      LanceRead: elapsed=3.248ms, uri=data/t.lance/data, projection=[id], source=stream(_rowid), \
metrics=[output_rows=10, elapsed_compute=7.12µs, output_bytes=126.1 KB, output_batches=1, fragments_scanned=1, \
ranges_scanned=10, rows_scanned=10, bytes_read=25.95 K, iops=6, requests=1]
        FilterExec: elapsed=2.765ms, _distance@2 IS NOT NULL, metrics=[output_rows=10, elapsed_compute=8.88µs, \
output_bytes=126.0 KB, output_batches=1, selectivity=100% (10/10)]
          SortExec: elapsed=2.745ms, TopK(fetch=10), expr=[_distance@2 ASC NULLS LAST, _rowid@0 ASC NULLS LAST], \
preserve_partitioning=[false], filter=[_distance@2 < 0.22365391 OR _distance@2 = 0.22365391 AND _rowid@0 < 2368], \
metrics=[output_rows=10, elapsed_compute=13.96µs, output_bytes=30.1 KB, output_batches=1, row_replacements=11]
            KNNVectorDistance: elapsed=2.745ms, metric=dot, metrics=[output_rows=40, elapsed_compute=23.79µs, \
output_bytes=120.5 KB, output_batches=1]
              LanceRead: elapsed=2.709ms, uri=data/t.lance/data, projection=[vector], source=stream(_rowid), \
metrics=[output_rows=40, elapsed_compute=24.50µs, output_bytes=120.5 KB, output_batches=1, fragments_scanned=1, \
ranges_scanned=40, rows_scanned=40, bytes_read=122.9 K, iops=40, requests=1]
                SortExec: elapsed=1.97ms, TopK(fetch=40), expr=[_distance@0 ASC NULLS LAST], \
preserve_partitioning=[false], metrics=[output_rows=40, elapsed_compute=17.25µs, output_bytes=480.0 B, \
output_batches=1, row_replacements=40]
                  ANNSubIndex: elapsed=1.97ms, name=vector_idx, k=40, deltas=1, metric=Dot, metrics=[output_rows=40, \
elapsed_compute=1.97ms, output_bytes=0.0 B, output_batches=0, index_cache_hits=0, index_cache_misses=32, \
index_comparisons=5.70 K, indices_loaded=0, partitions_searched=32, parts_loaded=32, bytes_read=1.50 M, iops=215, \
requests=24]
                    ANNIvfPartition: elapsed=1.734ms, uuid=a165d04e, minimum_nprobes=32, maximum_nprobes=Some(32), \
deltas=1, metrics=[output_rows=1, elapsed_compute=1.73ms, output_bytes=0.0 B, output_batches=0, deltas_searched=1, \
partitions_ranked=256, parts_loaded=0, bytes_read=797.2 K, iops=5, requests=1, find_partitions_elapsed=59.71µs]
"""


class DistanceTest(unittest.TestCase):
    def test_metric_mapping(self):
        self.assertEqual(lancedb.distance_type("l2", True), "l2")
        self.assertEqual(lancedb.distance_type("ip", False), "dot")
        self.assertEqual(lancedb.distance_type("cosine", True), "dot")
        self.assertEqual(lancedb.distance_type("cosine", False), "cosine")

    def test_unit_rows(self):
        rng = np.random.default_rng(1)
        rows = rng.standard_normal((500, 8)).astype(np.float32)
        self.assertFalse(lancedb.unit_rows(rows))
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        self.assertTrue(lancedb.unit_rows(rows))
        self.assertFalse(lancedb.unit_rows(rows[:0]))


class BuildSettingsTest(unittest.TestCase):
    def test_partitions(self):
        self.assertEqual(lancedb.partitions({"partitions": 300, "vpl": 10}, 50000), 300)
        self.assertEqual(lancedb.partitions({"vpl": 160}, 50000), 312)
        self.assertEqual(lancedb.partitions({"vpl": 160}, 100), 1)
        self.assertIsNone(lancedb.partitions({}, 50000))

    def test_sample_rate_keeps_the_sample_within_the_rows_and_a_million(self):
        self.assertEqual(lancedb.sample_rate({}, None, 50000), 256)
        self.assertEqual(lancedb.sample_rate({}, 100, 50000), 256)
        self.assertEqual(lancedb.sample_rate({}, 312, 50000), 160)
        self.assertEqual(lancedb.sample_rate({}, 6250, 1_000_000), 160)
        self.assertEqual(lancedb.sample_rate({}, 62500, 10_000_000), 16)
        self.assertEqual(lancedb.sample_rate({}, 5_000_000, 10_000_000), 1)
        self.assertEqual(lancedb.sample_rate({"sample_rate": 64}, 6250, 1_000_000), 64)

    def test_index_kwargs(self):
        kind, rq = lancedb.index_kwargs({"bits": 5}, "dot", 312, 256)
        self.assertEqual(kind, "ivf_rq")
        self.assertEqual(rq, {"distance_type": "dot", "num_partitions": 312, "sample_rate": 256, "max_iterations": 50,
                              "num_bits": 5})
        kind, pq = lancedb.index_kwargs({"index": "IVF_PQ"}, "dot", None, 256)
        self.assertEqual((kind, pq["num_bits"], pq["num_sub_vectors"], pq["num_partitions"]), ("ivf_pq", 8, 96, None))
        self.assertEqual(lancedb.index_kwargs({}, "l2", 4, 256)[1]["num_bits"], 1)

        with self.assertRaises(ValueError):
            lancedb.index_kwargs({"index": "ivf_hnsw_sq"}, "dot", 4, 256)

    def test_unknown_build_settings_fail(self):
        lancedb.check_build({"bits": 1, "vpl": 160})

        with self.assertRaises(ValueError):
            lancedb.check_build({"nbits": 1})


class SearchSettingsTest(unittest.TestCase):
    def test_checks(self):
        lancedb.check_search({"nprobes": 8, "refine_factor": 4, "approx_mode": "fast"}, pylance=True)
        lancedb.check_search({"nprobes": 8, "refine_factor": 4, "open": "lancedb"}, pylance=False)

        with self.assertRaises(ValueError):
            lancedb.check_search({"nprobe": 8}, pylance=True)

        with self.assertRaises(ValueError):
            lancedb.check_search({"refine_factor": 4}, pylance=True)

        with self.assertRaises(ValueError):
            lancedb.check_search({"nprobes": 8, "open": "fast"}, pylance=True)

        for params in ({"nprobes": 8, "approx_mode": "fast"}, {"nprobes": 8, "query_parallelism": 4},
                       {"nprobes": 8, "open": "lance"}):
            with self.assertRaises(Unsupported):
                lancedb.check_search(params, pylance=False)

    def test_the_open_path(self):
        self.assertEqual(lancedb.open_mode({"nprobes": 8}, pylance=True), "lance")
        self.assertEqual(lancedb.open_mode({"nprobes": 8}, pylance=False), "lancedb")
        self.assertEqual(lancedb.open_mode({"nprobes": 8, "open": "lancedb"}, pylance=True), "lancedb")
        self.assertEqual(lancedb.open_mode({"nprobes": 8, "open": None}, pylance=True), "lance")

    def test_nearest(self):
        q = np.zeros(4, dtype=np.float32)
        out = lancedb.nearest(q, 10, {"nprobes": 64, "refine_factor": 2, "approx_mode": "accurate"})
        self.assertEqual((out["minimum_nprobes"], out["maximum_nprobes"], out["k"]), (64, 64, 10))
        self.assertEqual((out["refine_factor"], out["approx_mode"], out["query_parallelism"]), (2, "accurate", -1))
        self.assertIs(out["q"], q)
        plain = lancedb.nearest(q, 10, {"nprobes": 8, "refine_factor": None, "query_parallelism": 0})
        self.assertNotIn("refine_factor", plain)
        self.assertNotIn("approx_mode", plain)
        self.assertEqual(plain["query_parallelism"], 0)
        self.assertNotIn("refine_factor", lancedb.nearest(q, 10, {"nprobes": 8, "refine_factor": 0}))

    def test_pad(self):
        np.testing.assert_array_equal(lancedb.pad(np.array([5, 7]), 4), [5, 7, -1, -1])
        np.testing.assert_array_equal(lancedb.pad(np.arange(6), 4), [0, 1, 2, 3])


class PlanTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(lancedb.parse_value("40"), 40)
        self.assertAlmostEqual(lancedb.parse_value("1.97ms"), 1970.0)
        self.assertAlmostEqual(lancedb.parse_value("7.12µs"), 7.12)
        self.assertAlmostEqual(lancedb.parse_value("250ns"), 0.25)
        self.assertAlmostEqual(lancedb.parse_value("1.5s"), 1.5e6)
        self.assertAlmostEqual(lancedb.parse_value("25.95 K"), 25950.0)
        self.assertAlmostEqual(lancedb.parse_value("2.30 M"), 2.3e6)
        self.assertAlmostEqual(lancedb.parse_value("126.1 KB", "output_bytes"), 126.1 * 1024)
        self.assertEqual(lancedb.parse_value("0.0 B", "output_bytes"), 0.0)
        self.assertAlmostEqual(lancedb.parse_value("1.2 B", "rows_scanned"), 1.2e9)
        self.assertIsNone(lancedb.parse_value("100% (10/10)"))
        self.assertIsNone(lancedb.parse_value("Some(32)"))

    def test_stages_and_roles(self):
        stages = lancedb.parse_plan(ANALYZED)
        self.assertEqual([s["op"] for s in stages][:3], ["AnalyzeExec", "TracedExec", "ProjectionExec"])
        self.assertEqual(len(stages), 11)
        roles = {s["role"]: s for s in stages if s["role"]}
        self.assertEqual(set(roles), {"ids", "refine", "partitions", "ranking"})
        self.assertEqual(roles["ids"]["metrics"]["bytes_read"], 25950.0)
        self.assertEqual(roles["refine"]["metrics"]["iops"], 40)
        self.assertEqual(roles["partitions"]["metrics"]["parts_loaded"], 32)
        self.assertAlmostEqual(roles["partitions"]["metrics"]["index_comparisons"], 5700.0)
        self.assertEqual(roles["ranking"]["metrics"]["partitions_ranked"], 256)
        self.assertEqual(roles["ranking"]["depth"], 10)
        self.assertAlmostEqual(roles["ranking"]["elapsed_us"], 1734.0)
        self.assertNotIn("selectivity", stages[4]["metrics"])
        self.assertEqual(stages[0]["metrics"], {})

    def test_summary(self):
        summary = lancedb.plan_summary(lancedb.parse_plan(ANALYZED))
        self.assertAlmostEqual(summary["bytes_read"], 25950 + 122900 + 1.5e6 + 797200)
        self.assertEqual(summary["iops"], 6 + 40 + 215 + 5)
        self.assertEqual(summary["parts_loaded"], 32)
        self.assertAlmostEqual(summary["elapsed_us"], 3250.0)
        self.assertAlmostEqual(summary["by_role"]["refine"]["bytes_read"], 122900.0)
        self.assertEqual(summary["by_role"]["ranking"]["iops"], 5)

    def test_mean_summary(self):
        a = {"bytes_read": 10.0, "elapsed_us": None, "by_role": {"ids": {"iops": 2.0}}}
        b = {"bytes_read": 30.0, "elapsed_us": 4.0, "by_role": {"ids": {"iops": 4.0}, "refine": {"iops": 8.0}}}
        mean = lancedb.mean_summary([a, b])
        self.assertEqual(mean["bytes_read"], 20.0)
        self.assertEqual(mean["elapsed_us"], 4.0)
        self.assertEqual(mean["by_role"], {"ids": {"iops": 3.0}, "refine": {"iops": 8.0}})
        self.assertEqual(mean["plans"], 2)
        self.assertEqual(lancedb.mean_summary([]), {})


class FakeCache(lancedb.LanceDB):
    """A LanceDB adapter over a pretend library. A search leaves the model, `model` bytes, and `part` bytes for each
    partition it probes in the index cache, and the cache keeps the index loaded only from `keeps_at` bytes."""

    def __init__(self, keeps_at: int, model: int = 1000, part: int = 100):
        super().__init__()
        self.keeps_at, self.model, self.part = keeps_at, model, part
        self.searched: list[float] = []

    def new_session(self, capacity: int | None = None) -> dict:
        return {"capacity": capacity, "loaded": False, "bytes": 0}

    def open_with(self, path: Path, params: dict, session: dict | None = None) -> lancedb.Handle:
        return lancedb.Handle(path=path, meta={}, session=session)

    def search(self, handle: lancedb.Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        session = handle.session
        kept = session["capacity"] is None or session["capacity"] >= self.keeps_at
        loads = 0 if session["loaded"] and kept else 1
        session["loaded"] = True
        session["bytes"] = self.model + int(params["nprobes"]) * self.part
        self.searched.append(float(query[0]))

        if params.get("analyze"):
            handle.plan = [{"op": "ANNSubIndex", "depth": 0, "elapsed_us": 1.0, "role": "partitions",
                            "metrics": {"indices_loaded": loads}}]

        return lancedb.pad(np.empty(0), k)

    def cache_bytes(self, handle: lancedb.Handle) -> int:
        return handle.session["bytes"]


class C1CapacityTest(unittest.TestCase):
    def capacity(self, keeps_at: int) -> tuple[dict, FakeCache]:
        ctx = ServeContext(index=Path("."), dataset=None, rows=10, k=10, metric="cosine", threads=1, variant="",
                           params={"nprobes": 8}, queries={1: np.zeros(4), 3: np.ones(4)}, regime="c1",
                           warmup=[np.full(4, 7.0), np.full(4, 9.0)])
        fake = FakeCache(keeps_at)
        return fake.c1_capacity(ctx), fake

    def test_the_cache_starts_at_the_model_and_one_partition(self):
        found, fake = self.capacity(1100)
        self.assertEqual((found["start"], found["model_bytes"], found["partition_bytes"]), (1100, 1000, 100))
        self.assertEqual((found["capacity"], found["kept"], found["tries"]), (1100 + lancedb.C1_MARGIN * 100, True, 1))
        # Its searches run the warm-up queries, which no trial searches.
        self.assertEqual(set(fake.searched), {7.0, 9.0})

    def test_the_cache_grows_one_partition_at_a_time(self):
        found, _ = self.capacity(1450)
        self.assertEqual((found["capacity"], found["kept"], found["tries"]), (1500 + lancedb.C1_MARGIN * 100, True, 5))

    def test_growth_stops_after_its_tries(self):
        found, _ = self.capacity(10 ** 12)
        self.assertEqual((found["start"], found["kept"], found["tries"]), (1100, False, lancedb.C1_TRIES))
        doubled = lancedb.C1_TRIES - lancedb.C1_STEPS
        self.assertEqual(found["capacity"], 1100 + 100 * lancedb.C1_STEPS + 100 * (2 ** (doubled + 1) - 2))


class FilesTest(unittest.TestCase):
    def test_routing_and_bulk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = {
                "meta.json": "routing",
                "t.lance/_versions/latest_version_hint.json": "routing",
                "t.lance/_versions/18446744073709551614.manifest": "routing",
                "t.lance/_indices/a1/index.idx": "routing",
                "t.lance/_indices/a1/auxiliary.idx": "bulk",
                "t.lance/data/0101.lance": "bulk",
                "t.lance/_transactions/0-b2.txn": "bulk",
            }

            for name in layout:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(b"x" * 100)

            files = lancedb.LanceDB().files(root)
            kinds = {str(Path(f["path"]).relative_to(root)): f["ranges"][0]["kind"] for f in files}
            self.assertEqual(kinds, layout)
            # Lance keeps what it reads of the routing files in its own caches.
            loaded = {str(Path(f["path"]).relative_to(root)) for f in files if f["ranges"][0].get("loaded")}
            self.assertEqual(loaded, {name for name, kind in layout.items() if kind == "routing"})
            self.assertEqual({(f["bytes"], len(f["ranges"]), f["ranges"][0]["length"]) for f in files}, {(100, 1, 100)})


if __name__ == "__main__":
    unittest.main()
