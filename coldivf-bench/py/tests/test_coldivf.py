import os
import struct
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
from adapters import coldivf
from adapters.base import ServeContext, Unsupported
import data

# The first query line and a later one of a real `cold --one` run, trimmed to the fields the adapter reads.
FIRST = {
    "seq": 0, "query": 5, "recall": 1, "pages_resident_at_start": 0, "ctx_us": 28.6, "open_us": 1879.3, "read_us": 0,
    "prepare_us": 18.1, "gap_us": 0, "search_us": 2288.6, "total_us": 4186.0, "tables_us": 150.4, "route_us": 22.8,
    "submit_us": 83.4, "wave_us": 348, "cluster_wave_us": 431.4, "scan_us": 36.5, "rows_us": 1638.2,
    "rerank_us": 8.3, "post_embedding_us": 2287.6, "cluster_bytes": 983040, "cluster_ranges": 10,
    "row_bytes": 589824, "row_ranges": 30, "footprint_delta": 1949744.0, "p_share": 0.99, "ghz": 3.6,
    "open_p_share": 0.99, "open_ghz": 3.45, "energy_nj": 2523977, "pageins": 0, "disk_bytes": 1736704,
    "syscall_open_us": 771.8, "qos": "user-interactive", "index_crc": 722862947, "index_epoch": 0,
}
NEXT = {"seq": 1, "query": 6, "recall": 0.8, "open_us": 0, "prepare_us": 0, "search_us": 1817.0, "total_us": 1817.0}


def write_index(path: Path, sizes: list[int]) -> None:
    """A file with a ColdIVF version 1 header whose sections have `sizes`, laid out one after another."""
    head = bytearray(4096)
    head[:8] = coldivf.MAGIC
    struct.pack_into("<I", head, 8, 1)
    struct.pack_into("<Q", head, 24, 1234)
    offset = 0

    for i, size in enumerate(sizes):
        struct.pack_into("<QQII", head, 80 + 24 * i, offset, size, i + 1, 0)
        offset += size

    path.write_bytes(bytes(head) + b"\0" * (offset - 4096))


class HeaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.file = self.dir / "index.coldivf"
        write_index(self.file, [4096, 12288, 16384, 16384, 32768, 65536, 16384])

    def tearDown(self):
        self.tmp.cleanup()

    def test_sections_and_count(self):
        s = coldivf.sections(self.file)
        self.assertEqual(s["centers"], (16384, 16384))
        self.assertEqual(s["ids"], (147456, 16384))
        self.assertEqual(coldivf.header_count(self.file), 1234)

    def test_not_an_index(self):
        bad = self.dir / "bad.coldivf"
        bad.write_bytes(b"x" * 400)

        with self.assertRaises(ValueError):
            coldivf.sections(bad)

    def test_files_mark_routing_and_bulk(self):
        files = coldivf.ColdIVF().files(self.dir)
        main = files[0]
        kinds = [r["kind"] for r in main["ranges"]]
        self.assertEqual(kinds, ["routing"] * 4 + ["bulk"] * 3)
        self.assertEqual([Path(f["path"]).name for f in files[1:]], ["index.coldivf.delta", "index.coldivf.lock"])
        self.assertEqual(files[1]["ranges"], [])

    def test_index_file(self):
        self.assertEqual(coldivf.index_file(self.dir), self.file)
        self.assertEqual(coldivf.index_file(self.file), self.file)

        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                coldivf.index_file(empty)


class FlagsTest(unittest.TestCase):
    def test_flags(self):
        self.assertEqual(coldivf.flags({"nprobe": 32, "rerank": 39}, coldivf.SEARCH_FLAGS, "x"),
                         ["--nprobe", "32", "--rerank", "39"])
        self.assertEqual(coldivf.flags({"reach": None}, coldivf.SEARCH_FLAGS, "x"), [])
        self.assertEqual(coldivf.flags({"clustering": "two-level"}, coldivf.BUILD_FLAGS, "x"),
                         ["--clustering", "two-level"])

        with self.assertRaises(Unsupported):
            coldivf.flags({"nprobes": 3}, coldivf.SEARCH_FLAGS, "search setting")


class TrialTest(unittest.TestCase):
    def ctx(self, regime: str = "c0") -> ServeContext:
        return ServeContext(index=Path("."), dataset=None, rows=50000, k=10, metric="cosine", threads=1, variant="",
                            regime=regime)

    def test_c0_fields(self):
        adapter = coldivf.ColdIVF()
        adapter.tie_free = {5: True}
        out = adapter.trial_fields(self.ctx(), [5, 6], [FIRST, NEXT])
        self.assertAlmostEqual(out["open_us"], 1879.3 + 18.1)
        self.assertEqual(out["total_us"], 4186.0)
        self.assertEqual(out["next_us"], [1817.0])
        self.assertEqual(out["recall10"], 1)
        self.assertEqual(out["stages"]["cluster_bytes"], 983040)
        self.assertIsInstance(out["footprint_delta"], int)
        self.assertNotIn("status", out)

    def test_recall_needs_a_tie_free_truth(self):
        adapter = coldivf.ColdIVF()
        adapter.tie_free = {5: False}
        out = adapter.trial_fields(self.ctx(), [5, 6], [FIRST, NEXT])
        self.assertIsNone(out["recall10"])
        self.assertEqual(out["recall10_ids_only"], 1)

    def test_intent_counts_open_time_past_the_delay(self):
        adapter = coldivf.ColdIVF()
        ctx = self.ctx("intent")
        ctx.delay_ms = 1.0
        # The binary's gap overshoots the 1 ms asked for, and the open time past 1 ms counts all the same.
        out = adapter.trial_fields(ctx, [5], [FIRST | {"gap_us": 3600.0}])
        self.assertAlmostEqual(out["total_us"], (1879.3 + 18.1 - 1000.0) + 2288.6)
        # The vector came while the open still ran, and the search started the whole gap after it.
        self.assertAlmostEqual(out["extra"]["intent_search_late_us"], 3600.0)
        self.assertFalse(out["extra"]["intent_comparable"])
        ctx.delay_ms = 5.0
        out = adapter.trial_fields(ctx, [5], [FIRST | {"gap_us": 7600.0}])
        self.assertAlmostEqual(out["total_us"], 2288.6)
        self.assertAlmostEqual(out["extra"]["intent_search_late_us"], 1879.3 + 18.1 + 7600.0 - 5000.0)
        ctx.delay_ms = 0.0
        out = adapter.trial_fields(ctx, [5], [FIRST])
        self.assertAlmostEqual(out["total_us"], 1879.3 + 18.1 + 2288.6)
        self.assertTrue(out["extra"]["intent_comparable"])

    def test_what_the_binary_leaves_out_is_marked(self):
        out = coldivf.ColdIVF().trial_fields(self.ctx(), [5, 6], [FIRST, NEXT])
        self.assertIsNone(out["resident_rerun_us"])
        self.assertEqual((out["extra"]["dummy_searched"], out["extra"]["rerun_measured"]), (False, False))
        self.assertNotIn("intent_comparable", out["extra"])

    def test_the_binaries_carry_no_per_call_cost(self):
        self.assertEqual(coldivf.ColdIVF().run_overhead(None)["overhead_us"], 0.0)

    def test_cached_pages_reject_the_trial(self):
        out = coldivf.ColdIVF().trial_fields(self.ctx(), [5], [FIRST | {"pages_resident_at_start": 3}])
        self.assertEqual(out["status"], "rejected")


class WarmFieldsTest(unittest.TestCase):
    CTX = ServeContext(index=Path("."), dataset=None, rows=50000, k=10, metric="cosine", threads=1, variant="")

    def test_a_binary_of_one_pass(self):
        line = {"p50_us": 88.0, "p99_us": 160.0, "mean_us": 100.0, "queries": 500, "recall": 0.93,
                "share_at_0_9": 0.8, "scanned": 4700.0}
        out = coldivf.ColdIVF().warm_fields(self.CTX, line, {1: True, 10: True})
        self.assertEqual((out["recall10"], out["recall10_ids_only"], out["passes"]), (0.93, 0.93, 1))
        self.assertEqual(out["qps"], 10000.0)
        self.assertIsNone(out["p90_us"])
        self.assertIsNone(out["recall1"])

    def test_a_binary_with_passes_and_recall_levels(self):
        line = {"p50_us": 88.0, "p90_us": 120.0, "mean_us": 100.0, "queries": 500, "recall": 0.93, "recall1": 0.97,
                "recall10": 0.93, "recall100": 0.9, "recall10_p05": 0.6, "recall10_p10": 0.7, "ties": False,
                "passes": 3, "share_at_0_9": 0.8}
        out = coldivf.ColdIVF().warm_fields(self.CTX, line, {1: True, 10: True})
        self.assertEqual((out["recall1"], out["recall10_p05"], out["passes"], out["p90_us"]), (0.97, 0.6, 3, 120.0))
        self.assertEqual(out["extra"]["binary"]["recall100"], 0.9)

    def test_recall_by_ids_needs_a_split_without_ties(self):
        line = {"mean_us": 100.0, "recall": 0.93, "recall1": 0.97, "recall10_p05": 0.6, "ties": False,
                "share_at_0_9": 0.8}
        out = coldivf.ColdIVF().warm_fields(self.CTX, line, {1: False, 10: False})
        self.assertIsNone(out["recall10"])
        self.assertIsNone(out["recall1"])
        self.assertIsNone(out["recall10_p05"])
        self.assertEqual(out["recall10_ids_only"], 0.93)
        self.assertEqual(out["extra"]["recall_source"], "binary, ids only")

    def test_recall_the_binary_counts_with_ties_fills_the_fields(self):
        line = {"mean_us": 100.0, "recall": 0.93, "recall1": 0.97, "recall10": 0.94, "recall100": 0.9,
                "recall10_p05": 0.6, "recall10_p10": 0.7, "ties": True, "share_at_0_9": 0.8}
        out = coldivf.ColdIVF().warm_fields(self.CTX, line, {1: False, 10: False})
        self.assertEqual((out["recall10"], out["recall1"], out["recall10_p05"], out["recall10_p10"]),
                         (0.94, 0.97, 0.6, 0.7))
        self.assertEqual((out["recall10_ids_only"], out["recall100"]), (0.93, 0.9))
        # The binary's share counts ids alone, so it needs a split without ties.
        self.assertIsNone(out["share_ge_0_9"])
        self.assertEqual(out["extra"]["recall_source"], "binary, with ties")

    def test_offers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warm"
            path.write_bytes(b"xx --mode t1 drops the rows yy")
            self.assertTrue(coldivf.offers(path, b"--mode t1"))
            self.assertFalse(coldivf.offers(path, b"--split must"))
            self.assertFalse(coldivf.offers(Path(tmp) / "missing", b"x"))


class SplitDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root, n=600, nq=20)

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_of_the_full_base(self):
        root, name = coldivf.split_dataset(self.ds, 600, "test")
        self.assertEqual(name, "test.tiny-16-cosine")
        self.assertEqual(os.readlink(root / f"{name}.base.fbin"), str(self.ds.base_path))
        queries = np.fromfile(root / f"{name}.query.fbin", dtype=np.float32, offset=8).reshape(-1, 16)
        np.testing.assert_array_equal(queries, np.asarray(self.ds.queries)[1::2])
        truth = np.fromfile(root / f"{name}.gt.ibin", dtype=np.int32, offset=8).reshape(10, -1)
        np.testing.assert_array_equal(truth, np.asarray(self.ds.truth().ids)[1::2])
        # Minus the inner product of each truth row, as the binaries measure a cosine dataset of unit vectors.
        dists = np.asarray(data.fbin(root / f"{name}.gt.dist.fbin"))
        base, ids = np.asarray(self.ds.base, dtype=np.float64), truth.astype(np.int64)
        want = -np.einsum("qd,qkd->qk", queries.astype(np.float64), base[ids])
        np.testing.assert_allclose(dists, want, rtol=0, atol=1e-6)
        self.assertTrue(np.all(np.diff(dists, axis=1) >= -1e-6))
        # The Python truth checks each k-th distance against the base as it reads it, as the binaries do.
        split = data.Dataset(name, root, self.ds.derived)
        recall = data.Recall(split, split.truth())
        self.assertAlmostEqual(recall.kth(0, 10), float(dists[0, 9]), places=5)

    def test_near_ties_are_ties_between_rows_that_differ(self):
        qids = np.arange(20)
        # Query 3's 10th neighbor has an exact copy, which the binaries count as a tie too.
        self.assertEqual(coldivf.near_ties(data.Recall(self.ds, self.ds.truth()), qids), 0)
        base, queries = np.asarray(self.ds.base).copy(), np.asarray(self.ds.queries)
        tenth = helpers.exact(base, queries[5:6], 10)[0][0, 9]
        base[598] = base[tenth]
        base[598, 0] = np.nextafter(base[598, 0], np.float32(2))
        root = self.root / "near"
        data.write_fbin(root / "tiny-16-cosine.base.fbin", base)
        data.write_fbin(root / "tiny-16-cosine.query.fbin", queries)
        data.write_ibin(root / "tiny-16-cosine.gt.ibin", helpers.exact(base, queries, 100)[0].astype(np.int32))
        near = data.Dataset("tiny-16-cosine", root, root / "derived")
        self.assertEqual(coldivf.near_ties(data.Recall(near, near.truth()), qids), 1)

    def test_squared_distances_for_a_euclidean_dataset(self):
        ds = data.Dataset("tiny-16-euclidean", self.ds.root, self.ds.derived)
        ds._base = np.asarray(self.ds.base)
        queries, ids = np.asarray(self.ds.queries)[:2], np.array([[0, 5], [7, 1]])
        got = coldivf.truth_distances(ds, queries, ids)
        want = [[float(np.sum((ds._base[i] - queries[q]) ** 2, dtype=np.float64)) for i in row]
                for q, row in enumerate(ids)]
        np.testing.assert_allclose(got, want, rtol=1e-6)

        with self.assertRaises(ValueError):
            coldivf.truth_distances(ds, queries, np.array([[0, -1], [1, 2]]))

    def test_split_of_a_prefix_and_reuse(self):
        prefix = self.ds.derived / "tiny-16-cosine-300.gt.ibin"
        ids, _ = helpers.exact(np.asarray(self.ds.base)[:300], np.asarray(self.ds.queries), 100)
        data.write_ibin(prefix, ids.astype(np.int32))
        root, name = coldivf.split_dataset(self.ds, 300, "tune")
        target = self.ds.derived / f"{name}-300.gt.ibin"
        dists = data.dist_path(target)
        self.assertTrue(target.exists() and dists.exists())
        stamps = target.stat().st_mtime_ns, dists.stat().st_mtime_ns
        time.sleep(0.01)
        coldivf.split_dataset(self.ds, 300, "tune")
        self.assertEqual((target.stat().st_mtime_ns, dists.stat().st_mtime_ns), stamps)
        # New truth for the prefix makes both files again.
        time.sleep(0.01)
        data.write_ibin(prefix, ids.astype(np.int32))
        coldivf.split_dataset(self.ds, 300, "tune")
        self.assertGreater(dists.stat().st_mtime_ns, stamps[1])

    def test_all_is_the_dataset_itself(self):
        self.assertEqual(coldivf.split_dataset(self.ds, 600, "all"), (self.ds.root, self.ds.name))


if __name__ == "__main__":
    unittest.main()
