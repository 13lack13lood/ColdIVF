import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
from adapters import qdrant_shard as qs
from adapters.base import BuildContext, ServeContext, Unsupported

HAS_QDRANT = importlib.util.find_spec("qdrant_edge") is not None


def identity(folder: Path) -> tuple:
    files = [p for p in folder.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files), len(files), max(p.stat().st_mtime_ns for p in files)


class SettingsTest(unittest.TestCase):
    def test_dot_product_for_unit_vectors(self):
        self.assertEqual(qs.library_distance("cosine", True), "Dot")
        self.assertEqual(qs.library_distance("l2", True), "Dot")
        self.assertEqual(qs.library_distance("cosine", False), "Cosine")
        self.assertEqual(qs.library_distance("l2", False), "Euclid")
        self.assertEqual(qs.library_distance("ip", False), "Dot")

    def test_defaults_and_refusals(self):
        s = qs.settings_of({}, "cosine", True)
        self.assertEqual(s, {"codes": "sq8", "inline": False, "m": 16, "ef_construct": 100, "quantile": None,
                             "distance": "Dot"})
        self.assertTrue(qs.settings_of({"codes": "bq2", "inline": True}, "cosine", True)["inline"])

        with self.assertRaises(Unsupported):
            qs.settings_of({"codes": "pq"}, "cosine", True)

        with self.assertRaises(Unsupported):
            qs.settings_of({"ef": 100}, "cosine", True)

    def test_bulk_is_what_a_search_reads_from_disk(self):
        ram = {"inline": False}
        disk = {"inline": True}
        self.assertEqual(qs.file_kind("segments/a/vector_storage/matrix.dat", ram), "bulk")
        self.assertEqual(qs.file_kind("segments/a/vector_storage/quantized.data", ram), "routing")
        self.assertEqual(qs.file_kind("segments/a/vector_index/links_compressed.bin", ram), "routing")
        self.assertEqual(qs.file_kind("segments/a/vector_storage/quantized.data", disk), "bulk")
        self.assertEqual(qs.file_kind("segments/a/vector_index/links_comp_vec.bin", disk), "bulk")
        self.assertEqual(qs.file_kind("segments/a/id_tracker.mappings", disk), "routing")

    def test_codes_kept_in_ram_are_loaded(self):
        adapter = qs.QdrantEdge()
        ram, disk = {"settings": {"inline": False}}, {"settings": {"inline": True}}
        self.assertTrue(adapter.loaded("segments/a/vector_storage/quantized.data", ram))
        self.assertFalse(adapter.loaded("segments/a/vector_storage/quantized.data", disk))
        self.assertFalse(adapter.loaded("segments/a/vector_index/links_compressed.bin", ram))
        self.assertFalse(adapter.loaded("segments/a/vector_storage/matrix.dat", ram))


@unittest.skipUnless(HAS_QDRANT, "qdrant-edge-py isn't installed")
class LibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.ds = helpers.make_dataset(root / "data", n=500, nq=20)
        cls.adapter = qs.QdrantEdge()
        cls.adapter.load()
        cls.index = root / "index"
        cls.index.mkdir()
        ctx = BuildContext(config={"codes": "sq8"}, data_path=cls.ds.base_path, rows=500, out=cls.index,
                           clusters=None, threads=2, seed=1, metric="cosine", variant="", dataset=cls.ds)
        cls.fields = cls.adapter.build(ctx)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def serve(self, params: dict) -> ServeContext:
        return ServeContext(index=self.index, dataset=self.ds, rows=500, k=10, metric="cosine", threads=1, variant="",
                            params=params)

    def test_build_leaves_one_indexed_segment_and_no_log(self):
        extra = self.fields["extra"]
        self.assertEqual((extra["points"], extra["indexed_vectors"]), (500, 500))
        self.assertTrue(extra["wal_removed"])
        self.assertIn("edge_config.json", extra["stashed_files"])
        self.assertTrue(self.fields["exact_f32_stored"])

    def test_a_session_leaves_the_index_as_built(self):
        before = identity(self.index)
        ctx = self.serve({"hnsw_ef": 64, "oversampling": 2})
        handle = self.adapter.open(ctx, self.index)
        self.assertNotEqual(identity(self.index), before)
        found = self.adapter.search(handle, np.asarray(self.ds.queries[1], dtype=np.float32), 10, ctx.params)
        self.assertEqual(found.tolist(), np.asarray(self.ds.truth(500).ids)[1, :10].tolist())
        self.adapter.close(handle)
        self.assertEqual(identity(self.index), before)

    def test_a_worker_process_opens_a_private_copy(self):
        before = identity(self.index)
        self.adapter.private = True

        try:
            handle = self.adapter.open(self.serve({}), self.index)
            self.assertNotEqual(handle.folder, self.adapter.store(self.index).data)
            self.assertEqual(identity(self.index), before)
            self.adapter.close(handle)
            self.assertFalse(handle.folder.exists())
        finally:
            self.adapter.private = False

    def test_files_cover_the_store(self):
        files = self.adapter.files(self.index)
        kinds = {Path(f["path"]).name: f["ranges"][0]["kind"] for f in files if f["ranges"]}
        self.assertEqual(kinds["matrix.dat"], "bulk")
        self.assertEqual(kinds["quantized.data"], "routing")
        listed = {f["path"] for f in files}
        on_disk = {str(p) for p in self.index.rglob("*") if p.is_file()}
        self.assertEqual(listed, on_disk)

    def test_unknown_search_settings_are_unsupported(self):
        with self.assertRaises(Unsupported):
            self.adapter.search_params({"ef": 64})


if __name__ == "__main__":
    unittest.main()
