import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
from adapters import astervec_lsm as al
from adapters.base import BuildContext, ServeContext, Unsupported

HAS_ASTERVEC = importlib.util.find_spec("astervec") is not None


def identity(folder: Path) -> tuple:
    files = [p for p in folder.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files), len(files), max(p.stat().st_mtime_ns for p in files)


class SettingsTest(unittest.TestCase):
    def test_metric(self):
        self.assertEqual(al.library_metric("cosine", False), "Cosine")
        self.assertEqual(al.library_metric("ip", True), "Cosine")
        self.assertEqual(al.library_metric("l2", True), "L2")

        with self.assertRaises(Unsupported):
            al.library_metric("ip", False)

    def test_caches(self):
        self.assertEqual(al.cache_settings({}), {})
        self.assertEqual(al.cache_settings({"caches": "minimal", "ef": 32}),
                         {"block_cache_bytes": 0, "edge_cache_size": 0, "paged_max_cached_pages": 0})
        self.assertEqual(al.cache_settings({"caches": "minimal", "edge_cache_size": 50}),
                         {"block_cache_bytes": 0, "edge_cache_size": 50, "paged_max_cached_pages": 0})
        self.assertEqual(al.cache_settings({"paged_max_cached_pages": 16}), {"paged_max_cached_pages": 16})

        with self.assertRaises(Unsupported):
            al.cache_settings({"caches": "none"})

    def test_bulk_is_the_vector_file_and_the_edge_tables(self):
        self.assertEqual(al.file_kind("vectors.bin"), "bulk")
        self.assertEqual(al.file_kind("db/000008.sst"), "bulk")
        self.assertEqual(al.file_kind("db/lsm_vec_db.meta"), "routing")
        self.assertEqual(al.file_kind("db/MANIFEST-000012"), "routing")
        self.assertEqual(al.file_kind("vectors.bin.deleted"), "routing")


@unittest.skipUnless(HAS_ASTERVEC, "aster-vec isn't installed")
class LibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.ds = helpers.make_dataset(root / "data", n=500, nq=20)
        cls.adapter = al.AsterVec()
        cls.adapter.load()
        cls.index = root / "index"
        cls.index.mkdir()
        ctx = BuildContext(config={}, data_path=cls.ds.base_path, rows=500, out=cls.index, clusters=None,
                           threads=2, seed=3, metric="cosine", variant="", dataset=cls.ds)
        cls.fields = cls.adapter.build(ctx)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def serve(self, params: dict) -> ServeContext:
        return ServeContext(index=self.index, dataset=self.ds, rows=500, k=10, metric="cosine", threads=1, variant="",
                            params=params)

    def test_build_records_its_settings(self):
        extra = self.fields["extra"]
        self.assertEqual(extra["settings"]["seed"], 3)
        self.assertEqual(extra["settings"]["metric"], "Cosine")
        self.assertIn("db/lsm_vec_db.meta", extra["stashed_files"])
        self.assertGreater(extra["edge_table_bytes"], 0)

    def test_sessions_with_either_cache_leave_the_index_as_built(self):
        before = identity(self.index)
        ctx = self.serve({"ef": 64})
        handle = self.adapter.open(ctx, self.index)
        query = np.asarray(self.ds.queries[1], dtype=np.float32)
        truth = np.asarray(self.ds.truth(500).ids)[1, :10].tolist()
        self.assertEqual(self.adapter.search(handle, query, 10, ctx.params).tolist(), truth)
        minimal = {"ef": 64, "caches": "minimal"}
        self.adapter.configure(handle, minimal)
        self.assertEqual(handle.caches["paged_max_cached_pages"], 0)
        self.assertEqual(self.adapter.search(handle, query, 10, minimal).tolist(), truth)
        self.adapter.close(handle)
        self.assertEqual(identity(self.index), before)

    def test_a_worker_process_opens_a_private_copy(self):
        before = identity(self.index)
        main = self.adapter.open(self.serve({}), self.index)
        self.adapter.private = True

        try:
            # The store is open in place, and its lock file would refuse a second open of the same files.
            worker = self.adapter.open(self.serve({}), self.index)
            query = np.asarray(self.ds.queries[2], dtype=np.float32)
            self.assertEqual(self.adapter.search(worker, query, 10, {}).tolist(),
                             self.adapter.search(main, query, 10, {}).tolist())
            self.adapter.close(worker)
        finally:
            self.adapter.private = False
            self.adapter.close(main)

        self.assertEqual(identity(self.index), before)

    def test_files_cover_the_store(self):
        files = self.adapter.files(self.index)
        listed = {f["path"] for f in files}
        self.assertEqual(listed, {str(p) for p in self.index.rglob("*") if p.is_file()})
        bulk = {Path(f["path"]).name for f in files if f["ranges"] and f["ranges"][0]["kind"] == "bulk"}
        self.assertIn("vectors.bin", bulk)
        self.assertTrue(all(name == "vectors.bin" or name.endswith(".sst") for name in bulk))

    def test_unknown_settings_are_unsupported(self):
        with self.assertRaises(Unsupported):
            self.adapter.open(self.serve({"ef_search": 64}), self.index)


if __name__ == "__main__":
    unittest.main()
