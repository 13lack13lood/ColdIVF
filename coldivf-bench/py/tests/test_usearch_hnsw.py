import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
from adapters import graphs, usearch_hnsw as uh
from adapters.base import BuildContext, ServeContext, Unsupported

HAS_USEARCH = importlib.util.find_spec("usearch") is not None


def write_index(path: Path, rows: int, vector_bytes: int, levels: list[int], magic: bytes = uh.MAGIC) -> int:
    """A file laid out as usearch saves one, with M 16, and returns its size."""
    conn, base = 16, 32
    nodes = b"".join(b"\0" * (8 + 2 + 4 + 4 * base + level * (4 + 4 * conn)) for level in levels)
    body = struct.pack("<II", rows, vector_bytes) + b"\1" * (rows * vector_bytes)
    body += magic.ljust(uh.DENSE_HEAD_BYTES, b"\0")
    body += struct.pack("<QQQQQ", len(levels), conn, base, max(levels), 0)
    body += np.asarray(levels, dtype="<i2").tobytes() + nodes
    path.write_bytes(body)
    return len(body)


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.file = self.dir / uh.FILE
        self.size = write_index(self.file, rows=5, vector_bytes=768, levels=[0, 1, 0, 0, 2])

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout_reads_the_headers(self):
        lay = uh.layout(self.file)
        self.assertEqual((lay.rows, lay.vector_bytes, lay.nodes, lay.connectivity, lay.connectivity_base),
                         (5, 768, 5, 16, 32))
        self.assertEqual(lay.vectors_end, 8 + 5 * 768)
        self.assertEqual(lay.size, self.size)

    def test_files_mark_the_vectors_bulk_and_the_graph_routing(self):
        graphs.write_meta(self.dir, {"system": "usearch"})
        files = uh.USearch().files(self.dir)
        ranges = files[0]["ranges"]
        self.assertEqual([r["kind"] for r in ranges], ["routing", "bulk", "routing"])
        self.assertEqual(ranges[1], {"offset": 8, "length": 5 * 768, "kind": "bulk"})
        self.assertEqual(sum(r["length"] for r in ranges), self.size)
        self.assertEqual(ranges[2]["offset"] + ranges[2]["length"], self.size)
        self.assertEqual(Path(files[1]["path"]).name, graphs.META)
        self.assertEqual(files[1]["ranges"], [])
        self.assertEqual(uh.USearch().files(self.file)[0]["path"], str(self.file))

    def test_other_files_are_refused(self):
        bad = self.dir / "bad.usearch"
        write_index(bad, rows=2, vector_bytes=8, levels=[0, 0], magic=b"notours")

        with self.assertRaises(ValueError):
            uh.layout(bad)

        short = self.dir / "short.usearch"
        short.write_bytes(self.file.read_bytes()[: uh.layout(self.file).graph_start + uh.GRAPH_HEAD_BYTES + 3])

        with self.assertRaises(ValueError):
            uh.layout(short)


class MetricTest(unittest.TestCase):
    def test_inner_product_wherever_it_ranks_the_same(self):
        self.assertEqual(uh.library_metric("i8", "cosine", True), "cos")
        self.assertEqual(uh.library_metric("i8", "cosine", False), "cos")
        self.assertEqual(uh.library_metric("i8", "l2", True), "cos")
        self.assertEqual(uh.library_metric("f16", "cosine", True), "ip")
        self.assertEqual(uh.library_metric("f32", "l2", True), "ip")
        self.assertEqual(uh.library_metric("f16", "cosine", False), "cos")
        self.assertEqual(uh.library_metric("f32", "l2", False), "l2sq")
        self.assertEqual(uh.library_metric("f32", "ip", False), "ip")

        for metric in ("l2", "ip"):
            with self.assertRaises(Unsupported):
                uh.library_metric("i8", metric, False)


@unittest.skipUnless(HAS_USEARCH, "usearch isn't installed")
class LibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.ds = helpers.make_dataset(root / "data", n=400, nq=20)
        cls.adapter = uh.USearch()
        cls.adapter.load()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, config: dict, rows: int = 400) -> Path:
        out = Path(tempfile.mkdtemp(dir=self.tmp.name))
        ctx = BuildContext(config=config, data_path=self.ds.base_path, rows=rows, out=out, clusters=None, threads=2,
                           seed=1, metric="cosine", variant="", dataset=self.ds)
        fields = self.adapter.build(ctx)
        self.assertEqual(fields["stored_vector_bytes"], rows * 16 * uh.VECTOR_BYTES[config.get("dtype", "i8")])
        return out

    def serve(self, index: Path, params: dict) -> ServeContext:
        return ServeContext(index=index, dataset=self.ds, rows=400, k=10, metric="cosine", threads=1, variant="",
                            params=params)

    def test_the_saved_file_matches_its_layout(self):
        for dtype in uh.DTYPES:
            index = self.build({"dtype": dtype})
            lay = uh.layout(index / uh.FILE)
            node0 = 8 + 2 + 4 + 4 * lay.connectivity_base
            levels = np.fromfile(index / uh.FILE, dtype="<i2", count=lay.nodes,
                                 offset=lay.graph_start + uh.GRAPH_HEAD_BYTES)
            nodes = lay.nodes * node0 + int(levels.astype(np.int64).sum()) * (4 + 4 * lay.connectivity)
            self.assertEqual(lay.size, lay.graph_start + uh.GRAPH_HEAD_BYTES + 2 * lay.nodes + nodes)
            self.assertEqual(graphs.read_meta(index)["settings"]["metric"], "cos" if dtype == "i8" else "ip")

    def test_search_finds_the_neighbors_and_pads(self):
        index = self.build({"dtype": "f32"})
        ctx = self.serve(index, {"expansion_search": 200})
        handle = self.adapter.open(ctx, index)
        query = np.asarray(self.ds.queries[1], dtype=np.float32)
        found = self.adapter.search(handle, query, 10, ctx.params)
        self.assertEqual(found.tolist(), np.asarray(self.ds.truth(400).ids)[1, :10].tolist())
        self.assertGreater(self.adapter.stats(handle)["dist_evals"], 0)
        batch = self.adapter.search_batch(handle, np.asarray(self.ds.queries[:3]), 10, ctx.params, 2)
        self.assertEqual(batch.shape, (3, 10))
        self.adapter.close(handle)

        tiny = self.build({"dtype": "f16"}, rows=5)
        handle = self.adapter.open(self.serve(tiny, {}), tiny)
        self.assertEqual(sorted(self.adapter.search(handle, query, 8, {})[:5].tolist()), list(range(5)))
        self.assertEqual(self.adapter.search(handle, query, 8, {})[5:].tolist(), [-1, -1, -1])
        self.assertEqual((self.adapter.search_batch(handle, np.stack([query]), 8, {}, 1)[0, 5:] == -1).sum(), 3)
        self.adapter.close(handle)

    def test_unknown_settings_are_unsupported(self):
        index = self.build({"dtype": "i8"})

        with self.assertRaises(Unsupported):
            self.adapter.open(self.serve(index, {"ef": 64}), index)

        with self.assertRaises(Unsupported):
            self.build({"dtype": "b1"})

        with self.assertRaises(Unsupported):
            self.build({"M": 16})


if __name__ == "__main__":
    unittest.main()
