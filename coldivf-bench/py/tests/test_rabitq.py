import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters import rabitq_common as rc
from adapters.base import Unsupported
import data
from tests.inproc_runner import LibraryCase, has, recall10


class MetricAndCountTest(unittest.TestCase):
    def test_metric(self):
        self.assertEqual(rc.library_metric("cosine"), "ip")
        self.assertEqual(rc.library_metric("ip"), "ip")
        self.assertEqual(rc.library_metric("l2"), "l2")

        with self.assertRaises(Unsupported):
            rc.library_metric("hamming")

    def test_cluster_count(self):
        self.assertEqual(rc.cluster_count(2000, nlist=8), 8)
        self.assertEqual(rc.cluster_count(2000, vectors_per_list=100), 20)
        self.assertEqual(rc.cluster_count(1_000_000, vectors_per_list=160), 6250)

        for nlist, vpl in ((8, 100), (None, None), (0, None), (2001, None), (None, 4000)):
            with self.assertRaises(ValueError):
                rc.cluster_count(2000, nlist, vpl)

    def test_kmeans_method(self):
        self.assertEqual(rc.kmeans_method(None, 16), "rabitq")
        self.assertEqual(rc.kmeans_method(None, 33), "qg")
        self.assertEqual(rc.kmeans_method("rabitq", 6250), "rabitq")

        with self.assertRaises(ValueError):
            rc.kmeans_method("qg", 32)

        with self.assertRaises(ValueError):
            rc.kmeans_method("faiss", 100)

    def test_code_bytes(self):
        self.assertEqual(rc.code_bytes(10, 768, 1), 960)
        self.assertEqual(rc.code_bytes(10, 768, 5), 4800)
        self.assertEqual(rc.code_bytes(10, 768, 32), 10 * 768 * 4 + 960)
        # The rotated dimension is padded to a multiple of 32.
        self.assertEqual(rc.code_bytes(10, 100, 4), 10 * 128 * 4 // 8)


class PartitionTest(unittest.TestCase):
    def test_compact_drops_unused_clusters(self):
        centers = np.arange(12, dtype=np.float32).reshape(6, 2)
        part = rc.compact(rc.Partition(centers, np.array([0, 2, 2, 5], dtype=np.uint32), 1.5, {"a": 1}))
        np.testing.assert_array_equal(part.centers, centers[[0, 2, 5]])
        np.testing.assert_array_equal(part.assign, [0, 1, 1, 2])
        self.assertEqual(part.assign.dtype, np.uint32)
        self.assertEqual(part.info, {"a": 1, "dropped_clusters": 3})
        self.assertEqual(part.seconds, 1.5)

    def test_compact_keeps_a_full_partition(self):
        part = rc.Partition(np.zeros((2, 2), dtype=np.float32), np.array([1, 0], dtype=np.uint32))
        self.assertIs(rc.compact(part), part)

    def test_list_sizes(self):
        part = rc.Partition(np.zeros((4, 2), dtype=np.float32), np.array([0, 0, 0, 2], dtype=np.uint32))
        self.assertEqual(rc.list_sizes(part), {"clusters": 4, "list_min": 0, "list_max": 3, "empty_lists": 2})

    def test_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "shared"
            data.write_fbin(f"{stem}.centroids.fbin", np.eye(3, 4, dtype=np.float32))
            data.write_ibin(f"{stem}.assign.ibin", np.array([[2], [0], [1], [2]], dtype=np.int32))
            part = rc.imported(stem, 3, 4)
            np.testing.assert_array_equal(part.assign, [2, 0, 1])
            self.assertEqual((part.count, part.assign.dtype, part.seconds), (3, np.uint32, None))
            self.assertEqual(part.info["partition"], "imported")

            with self.assertRaisesRegex(ValueError, "covers 4 rows"):
                rc.imported(stem, 5, 4)

            with self.assertRaisesRegex(ValueError, "4 dimensions"):
                rc.imported(stem, 3, 8)


def write_partition(ds: data.Dataset, stem: Path, centers: int, unused: int = 0) -> None:
    """A shared partition of the base around `centers` of its rows, plus `unused` centers that no row is near."""
    base = np.asarray(ds.base)
    picks = base[np.linspace(0, len(base) - 1, centers).astype(int)]
    assign = np.argmax(base @ picks.T, axis=1)
    far = -np.ones((unused, base.shape[1]), dtype=np.float32) / np.sqrt(base.shape[1])
    data.write_fbin(f"{stem}.centroids.fbin", np.concatenate([picks, far]) if unused else picks)
    data.write_ibin(f"{stem}.assign.ibin", assign.reshape(-1, 1).astype(np.int32))


@unittest.skipUnless(has("rabitqlib"), "needs rabitqlib")
class IvfTest(LibraryCase):
    module = "adapters.rabitq_ivf"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.stem = cls.root / "shared"
        write_partition(cls.ds, cls.stem, 20)

    def test_raw_rows_find_the_neighbors(self):
        index, msg = self.built("raw", {"nlist": 20, "bits": 32})
        fields, info = msg["fields"], msg["info"]
        self.assertTrue(fields["exact_f32_stored"])
        self.assertEqual(fields["stored_vector_bytes"], rc.code_bytes(self.rows, 64, 32))
        self.assertGreater(fields["cluster_seconds"], 0)
        self.assertGreater(fields["encode_seconds"], 0)
        self.assertEqual(fields["extra"]["clusters"], 20)
        self.assertEqual(fields["extra"]["library"]["nbits"], 32)
        self.assertEqual((info["system"], info["batch_api"]), ("rabitq-ivf", True))
        self.assertEqual(info["regimes"], ["c0", "c1", "intent"])
        self.assertTrue((index / "ivf.rabitq").is_file())
        found = self.settings(index, [{"nprobe": 20}, {"nprobe": 2}])
        self.assertGreaterEqual(recall10(self.ds, found[0]), 0.95)
        self.assertGreaterEqual(recall10(self.ds, found[0]), recall10(self.ds, found[1]))
        self.check_cold(index, {"nprobe": 8})

    def test_codes_and_a_shared_partition(self):
        index, msg = self.built("shared-b5", {"bits": 5, "vectors_per_list": 100}, "--clusters", str(self.stem))
        fields = msg["fields"]
        self.assertIsNone(fields["cluster_seconds"])
        self.assertFalse(fields["exact_f32_stored"])
        self.assertEqual(fields["extra"]["partition"], "imported")
        self.assertEqual(fields["extra"]["clusters"], 20)
        found = self.settings(index, [{"nprobe": 20}])
        self.assertGreaterEqual(recall10(self.ds, found[0]), 0.8)
        found = self.settings(index, [{"nprobe": 10}], threads=2)
        self.assertGreater(found[0]["fields"]["qps"], 0)
        self.assertTrue(found[0]["fields"]["batch_api"])

    def test_vectors_per_list_sets_the_count(self):
        _, msg = self.built("vpl", {"vectors_per_list": 200, "bits": 1})
        self.assertEqual(msg["fields"]["extra"]["clusters"], 10)
        self.assertEqual(msg["fields"]["extra"]["kmeans"], "rabitq")

    def test_bad_settings_are_errors(self):
        msg, _ = self.runner.build(self.root / "bad", {"bits": 10, "nlist": 4})
        self.assertEqual(msg["event"], "error")
        self.assertIn("bits is 10", msg["message"])
        msg, _ = self.runner.build(self.root / "mismatch", {"bits": 1, "nlist": 7}, "--clusters", str(self.stem))
        self.assertIn("has 20", msg["message"])
        index, _ = self.built("small", {"nlist": 4, "bits": 1})
        got, _ = self.runner.warm(index, [{"nprobes": 3}])
        self.check_error(got, "unknown rabitq-ivf search settings")

    def test_overhead(self):
        got, err = self.runner.run("overhead", *self.runner.common())
        self.assertEqual(got[0]["event"], "overhead", err[-2000:])
        self.assertGreater(got[0]["fields"]["overhead_us"], 0)


@unittest.skipUnless(has("rabitqlib"), "needs rabitqlib")
class HnswTest(LibraryCase):
    module = "adapters.rabitq_hnsw"

    def test_graph(self):
        index, msg = self.built("b5", {"bits": 5, "M": 16, "ef_construction": 100})
        fields = msg["fields"]
        self.assertFalse(fields["exact_f32_stored"])
        self.assertEqual(fields["extra"]["clusters"], 16)
        self.assertEqual(fields["extra"]["library"]["num_points"], self.rows)
        self.assertTrue((index / "hnsw.rabitq").is_file())
        found = self.settings(index, [{"ef": 100}, {"ef": 10}])
        self.assertGreaterEqual(recall10(self.ds, found[0]), 0.8)
        self.check_cold(index, {"ef": 40})

    def test_unused_shared_clusters_are_dropped(self):
        stem = self.root / "unused"
        write_partition(self.ds, stem, 12, unused=3)
        _, msg = self.built("shared", {"bits": 4}, "--clusters", str(stem))
        extra = msg["fields"]["extra"]
        self.assertEqual((extra["clusters"], extra["dropped_clusters"]), (12, 3))

    def test_bits_must_fit(self):
        msg, _ = self.runner.build(self.root / "bad", {"bits": 32})
        self.assertIn("bits is 32", msg["message"])


class SymphonyQGMetricTest(unittest.TestCase):
    def test_cosine_runs_as_l2(self):
        from adapters import symphonyqg

        self.assertEqual(symphonyqg.graph_metric("cosine"), "l2")
        self.assertEqual(symphonyqg.graph_metric("l2"), "l2")
        self.assertEqual(symphonyqg.graph_metric("ip"), "ip")

        with self.assertRaises(Unsupported):
            symphonyqg.graph_metric("hamming")


@unittest.skipUnless(has("rabitqlib"), "needs rabitqlib")
class SymphonyQGTest(LibraryCase):
    module = "adapters.symphonyqg"

    def test_raw_and_coded_rows(self):
        index, msg = self.built("raw", {"degree": 32, "ef_construction": 100})
        self.assertTrue(msg["fields"]["exact_f32_stored"])
        self.assertEqual(msg["fields"]["stored_vector_bytes"], self.rows * 64 * 4)
        self.assertEqual(msg["fields"]["extra"]["library"]["metric"], "l2")
        found = self.settings(index, [{"ef": 100}, {"ef": 10}])
        self.assertGreaterEqual(recall10(self.ds, found[0]), 0.8)
        self.check_cold(index, {"ef": 40})
        coded, msg = self.built("q4", {"degree": 32, "ef_construction": 100, "quantization_bits": 4})
        self.assertFalse(msg["fields"]["exact_f32_stored"])
        self.assertEqual(msg["fields"]["extra"]["library"]["quantization_bits"], 4)
        found = self.settings(coded, [{"ef": 100}], threads=2)
        self.assertGreater(found[0]["fields"]["qps"], 0)
        got, _ = self.runner.warm(index, [{"ef": 5}])
        self.check_error(got, "ef must be at least k")

    def test_degree_must_be_a_multiple_of_32(self):
        msg, _ = self.runner.build(self.root / "bad", {"degree": 48})
        self.assertIn("multiple of 32", msg["message"])


if __name__ == "__main__":
    unittest.main()
