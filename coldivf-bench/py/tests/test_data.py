import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
import data


class TableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        rows = np.arange(12, dtype=np.float32).reshape(3, 4)
        data.write_fbin(self.dir / "a.fbin", rows)
        self.assertEqual(data.header(self.dir / "a.fbin"), (3, 4))
        np.testing.assert_array_equal(data.fbin(self.dir / "a.fbin"), rows)
        ids = np.arange(6, dtype=np.int32).reshape(2, 3)
        data.write_ibin(self.dir / "a.ibin", ids)
        np.testing.assert_array_equal(data.ibin(self.dir / "a.ibin"), ids)

    def test_empty_table(self):
        data.write_fbin(self.dir / "e.fbin", np.zeros((0, 5), dtype=np.float32))
        self.assertEqual(data.fbin(self.dir / "e.fbin").shape, (0, 5))

    def test_short_or_mismatched_files_fail(self):
        (self.dir / "short.fbin").write_bytes(b"\x01\x00")

        with self.assertRaises(ValueError):
            data.fbin(self.dir / "short.fbin")

        data.write_fbin(self.dir / "b.fbin", np.ones((2, 2), dtype=np.float32))

        with open(self.dir / "b.fbin", "ab") as f:
            f.write(b"\0\0\0\0")

        with self.assertRaises(ValueError):
            data.fbin(self.dir / "b.fbin")

    def test_write_leaves_no_temporary_file(self):
        data.write_ibin(self.dir / "c.ibin", np.zeros((1, 1), dtype=np.int32))
        self.assertEqual([p.name for p in self.dir.iterdir()], ["c.ibin"])


class ClustersTest(unittest.TestCase):
    def test_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "part"
            data.write_fbin(f"{stem}.centroids.fbin", np.zeros((3, 4), dtype=np.float32))
            data.write_ibin(f"{stem}.assign.ibin", np.array([[0], [2], [1]], dtype=np.int32))
            centroids, assign = data.clusters(stem)
            self.assertEqual(centroids.shape, (3, 4))
            np.testing.assert_array_equal(assign, [0, 2, 1])
            data.write_ibin(f"{stem}.assign.ibin", np.array([[3]], dtype=np.int32))

            with self.assertRaises(ValueError):
                data.clusters(stem)


class SplitTest(unittest.TestCase):
    def test_tune_and_test_alternate(self):
        tune, test = data.split_ids(1000, "tune"), data.split_ids(1000, "test")
        self.assertEqual(len(tune), 500)
        self.assertEqual(len(test), 500)
        self.assertTrue(np.all(tune % 2 == 0))
        self.assertTrue(np.all(test % 2 == 1))
        self.assertEqual(set(tune) | set(test), set(range(1000)))

    def test_limit_and_rest(self):
        self.assertEqual(len(data.split_ids(5000, "tune")), 500)
        self.assertEqual(data.split_ids(5000, "rest")[0], 1000)
        self.assertEqual(len(data.split_ids(5000, "all")), 5000)
        self.assertEqual(len(data.split_ids(40, "test")), 20)

    def test_unknown_split(self):
        with self.assertRaises(ValueError):
            data.split_ids(10, "train")


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root, n=500, nq=12)

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_and_paths(self):
        ds = data.Dataset.from_base(self.root / "tiny-16-cosine.base.fbin", derived=self.root / "derived")
        self.assertEqual(ds.name, "tiny-16-cosine")
        self.assertEqual((ds.count, ds.dim), (500, 16))
        self.assertEqual(ds.truth_path(), self.root / "tiny-16-cosine.gt.ibin")
        self.assertEqual(ds.truth_path(100), self.root / "derived" / "tiny-16-cosine-100.gt.ibin")
        self.assertEqual(ds.prefix(100).shape, (100, 16))

        with self.assertRaises(ValueError):
            data.Dataset.from_base(self.root / "x.fbin")

    def test_metric_of(self):
        self.assertEqual(data.metric_of("sift-128-euclidean"), "l2")
        self.assertEqual(data.metric_of("cohere-768-cosine"), "cosine")
        self.assertEqual(data.metric_of("x-ip"), "ip")

    def test_truth_record(self):
        truth = self.ds.truth()
        self.assertEqual(truth.record(), {"path": str(self.root / "tiny-16-cosine.gt.ibin"), "k": 100,
                                          "has_dist": False})

    def test_distances(self):
        q = np.array([1.0, 0.0])
        rows = np.array([[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_allclose(data.distances("l2", q, rows), [0.0, 5.0])
        np.testing.assert_allclose(data.distances("ip", q, rows), [-1.0, 0.0])
        np.testing.assert_allclose(data.distances("cosine", q, rows), [-1.0, 0.0])


class RecallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root, n=2000, nq=12, tie_query=3)
        self.truth = self.ds.truth()
        self.recall = data.Recall(self.ds, self.truth)
        self.qids = np.arange(12)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_results_score_one(self):
        found = np.asarray(self.truth.ids[:, :100])
        s = self.recall.summary(self.qids, found)
        self.assertEqual(s["recall10"], 1.0)
        self.assertEqual(s["recall10_ids_only"], 1.0)
        self.assertEqual(s["recall1"], 1.0)
        self.assertEqual(s["recall100"], 1.0)
        self.assertEqual(s["share_ge_0_9"], 1.0)

    def test_recall100_needs_100_results(self):
        found = np.asarray(self.truth.ids[:, :10])
        self.assertIsNone(self.recall.summary(self.qids, found)["recall100"])

    def test_a_miss(self):
        found = np.asarray(self.truth.ids[:, :10]).copy()
        found[0, 9] = self.truth.ids[0, 50]
        ties, plain = self.recall.per_query(self.qids, found)
        self.assertAlmostEqual(ties[0], 0.9)
        self.assertAlmostEqual(plain[0], 0.9)
        self.assertEqual(ties[1], 1.0)

    def test_a_duplicate_of_the_tenth_counts_as_a_hit(self):
        found = np.asarray(self.truth.ids[:, :10]).copy()
        # Query 3's 11th neighbor is a copy of its 10th, at the last id.
        self.assertEqual(self.truth.ids[3, 10], 1999)
        found[3, 9] = 1999
        ties, plain = self.recall.per_query(self.qids, found)
        self.assertEqual(ties[3], 1.0)
        self.assertAlmostEqual(plain[3], 0.9)

    def test_repeats_and_padding_are_not_hits(self):
        found = np.asarray(self.truth.ids[:, :10]).copy()
        found[0, 5:] = found[0, 0]
        found[1, 5:] = -1
        ties, _ = self.recall.per_query(self.qids, found)
        self.assertAlmostEqual(ties[0], 0.5)
        self.assertAlmostEqual(ties[1], 0.5)

    def test_tie_free(self):
        free = self.recall.tie_free(self.qids, 10)
        self.assertFalse(free[3])
        self.assertTrue(free[0])

    def test_distance_file_scores_and_distances_agree(self):
        ids = np.asarray(self.truth.ids)
        _, dists = helpers.exact(np.asarray(self.ds.base), np.asarray(self.ds.queries), 100)
        found = ids[:, :10].copy()
        found[3, 9] = 1999

        for values in (dists, -dists):
            data.write_fbin(self.root / "tiny-16-cosine.gt.dist.fbin", values.astype(np.float32))
            truth = self.ds.truth()
            self.assertTrue(truth.has_dist)
            self.assertEqual(truth.scores, values is not dists)
            recall = data.Recall(self.ds, truth)
            self.assertAlmostEqual(recall.kth(0, 10), dists[0, 9], places=5)
            ties, _ = recall.per_query(self.qids, found)
            self.assertEqual(ties[3], 1.0)

    def test_a_distance_file_of_another_base_fails(self):
        data.write_fbin(self.root / "tiny-16-cosine.gt.dist.fbin", np.full((12, 100), 0.5, dtype=np.float32))

        with self.assertRaises(ValueError):
            data.Recall(self.ds, self.ds.truth()).kth(0, 10)

    def test_a_near_tie_is_not_a_tie(self):
        # Moves a copy of query 0's 10th neighbor a little farther, by far less than 1e-5 but more than float32 noise.
        q = np.asarray(self.ds.queries[0], dtype=np.float64)
        tenth = np.asarray(self.ds.base[self.truth.ids[0, 9]], dtype=np.float64)
        moved = tenth - 3e-6 * q
        moved /= np.linalg.norm(moved)
        base = np.asarray(self.ds.base).copy()
        base[1998] = moved
        data.write_fbin(self.root / "tiny-16-cosine.base.fbin", base)
        ds = data.Dataset(self.ds.name, self.root, self.root / "derived")
        recall = data.Recall(ds, self.truth)
        gap = data.distances("cosine", q, moved)[0] - recall.kth(0, 10)
        self.assertGreater(gap, 1e-7)
        self.assertLess(gap, 1e-5)
        found = np.asarray(self.truth.ids[:, :10]).copy()
        found[0, 9] = 1998
        ties, _ = recall.per_query(np.arange(12), found)
        self.assertAlmostEqual(ties[0], 0.9)

    def test_tails(self):
        found = np.asarray(self.truth.ids[:, :10]).copy()
        found[:6, 5:] = -1
        s = self.recall.summary(self.qids, found)
        self.assertAlmostEqual(s["recall10"], 0.75)
        self.assertAlmostEqual(s["share_ge_0_9"], 0.5)
        self.assertAlmostEqual(s["recall10_p05"], 0.5)

    def test_tails_are_nearest_rank(self):
        found = np.asarray(self.truth.ids[:, :10]).copy()
        found[0, 5:] = -1
        s = self.recall.summary(self.qids, found)
        self.assertAlmostEqual(s["recall10_p05"], 0.5)
        self.assertAlmostEqual(s["recall10_p10"], 1.0)


if __name__ == "__main__":
    unittest.main()
