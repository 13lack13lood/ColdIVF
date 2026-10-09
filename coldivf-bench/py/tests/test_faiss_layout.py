import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters import faiss_layout as fl
from tests.faiss_files import codes_of, find, graph, header, ilar, ivf_rabitq, pretransform, refine, sq8, vec

HAS_FAISS = importlib.util.find_spec("faiss") is not None


class SyntheticTest(unittest.TestCase):
    """Files written byte for byte in faiss's format, so the reader is checked without faiss."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.d, self.cs = 8, 6
        self.centroids = np.arange(3 * self.d, dtype="<f4").reshape(3, self.d) + 0.5
        self.lists = [np.array([0, 3, 4]), np.array([], dtype=np.int64), np.array([1, 2])]
        self.n = 5

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, raw: bytes) -> Path:
        path = self.dir / name
        path.write_bytes(raw)
        return path

    def ivf_file(self, sparse: bool = False) -> tuple[Path, bytes, np.ndarray]:
        rows = (np.arange(self.n * self.d) % 255).astype(np.uint8).reshape(self.n, self.d)
        inner = ivf_rabitq(self.d, self.centroids, ilar(self.cs, self.lists, sparse), self.n, self.cs)
        raw = refine(self.d, self.n, pretransform(self.d, self.n, inner), sq8(self.d, rows))
        return self.write("index.faiss", raw), raw, rows

    def test_rotated_ivf_with_refine_rows(self):
        path, raw, rows = self.ivf_file()
        layout = fl.read(path)
        self.assertEqual(layout.records, ["IxRF", "IxPT", "Iwrq", "IxF2", "IxSQ"])
        coarse, refine_rows = layout.part("coarse"), layout.part("refine")
        self.assertEqual(coarse.offset, find(raw, self.centroids))
        self.assertEqual(coarse.length, self.centroids.nbytes)
        self.assertEqual(refine_rows.offset, find(raw, rows))
        self.assertEqual(layout.row_bytes, {"coarse": 4 * self.d, "refine": self.d})
        self.assertEqual(layout.part("rotation").length, 4 * self.d * self.d)
        lists = layout.lists
        self.assertEqual(lists.sizes.tolist(), [3, 0, 2])

        for i, ids in enumerate(self.lists):
            if len(ids):
                self.assertEqual(raw[lists.codes[i]: lists.codes[i] + len(ids) * self.cs], codes_of(ids, self.cs))
                self.assertEqual(raw[lists.ids[i]: lists.ids[i] + 8 * len(ids)], ids.astype("<i8").tobytes())

        self.assertEqual(layout.part("lists").length, 5 * (self.cs + 8))
        self.assertEqual(lists.sizes_offset, layout.part("list_sizes").offset)
        json.dumps(layout.record())

    def test_ranges_cover_the_file(self):
        path, raw, _ = self.ivf_file()
        layout = fl.read(path)
        ranges = layout.ranges()
        self.assertEqual(ranges[0]["offset"], 0)
        self.assertEqual(sum(r["length"] for r in ranges), len(raw))

        for a, b in zip(ranges, ranges[1:]):
            self.assertEqual(a["offset"] + a["length"], b["offset"])
            self.assertNotEqual(a["kind"], b["kind"])

        bulk = [(r["offset"], r["length"]) for r in ranges if r["kind"] == "bulk"]
        self.assertEqual(bulk, [(p.offset, p.length) for p in (layout.part("lists"), layout.part("refine"))])

    def test_sparse_list_sizes(self):
        path, _, _ = self.ivf_file(sparse=True)
        layout = fl.read(path)
        self.assertEqual(layout.lists.sizes.tolist(), [3, 0, 2])
        self.assertEqual(layout.lists.sizes_type, "sprs")

    def test_spans_of_lists_in_the_index_file(self):
        path, _, _ = self.ivf_file()
        lists = fl.read(path).lists
        spans = lists.spans(np.array([2, -1, 1, 0, 7]))
        self.assertEqual(spans, [(int(lists.codes[2]), 2 * (self.cs + 8)), (int(lists.codes[0]), 3 * (self.cs + 8))])

    def test_ondisk_lists(self):
        table = np.array([[3, 4, 0], [0, 0, 0], [2, 2, 4 * (self.cs + 8)]], dtype="<u8")
        slots = np.zeros(0, dtype="<u8")
        name = np.frombuffer(b"/old/place/lists.ivfdata", dtype=np.uint8)
        # The table counts lists, not 8-byte fields, since each list is one 24-byte item.
        lists = (b"ilod" + struct.pack("<QQ", 3, self.cs) + struct.pack("<Q", 3) + table.tobytes() + vec(slots)
                 + vec(name) + struct.pack("<Q", 1000))
        inner = ivf_rabitq(self.d, self.centroids, lists, self.n, self.cs)
        path = self.write("ivf.index", pretransform(self.d, self.n, inner))
        layout = fl.read(path)
        self.assertEqual(layout.lists.file, "/old/place/lists.ivfdata")
        self.assertEqual(fl.ondisk_file(path, layout), self.dir / "lists.ivfdata")
        self.assertEqual(layout.lists.ids.tolist(), [4 * self.cs, 0, 4 * (self.cs + 8) + 2 * self.cs])
        self.assertEqual(layout.lists.spans(np.array([0])), [(0, 3 * self.cs), (4 * self.cs, 24)])
        self.assertEqual({r["kind"] for r in layout.ranges()}, {"routing"})

    def test_graph_with_codes(self):
        body, arrays = graph(self.n, 4)
        codes = np.full((self.n, self.d), 7, dtype=np.uint8)
        raw = b"IHNs" + header(self.d, self.n) + body + sq8(self.d, codes)
        layout = fl.read(self.write("hnsw.faiss", raw))
        parts = layout.named("graph")
        self.assertEqual([p.offset for p in parts], [find(raw, arrays[k]) for k in ("levels", "offsets", "neighbors")])
        self.assertEqual(layout.part("storage").offset, len(raw) - codes.nbytes)
        self.assertEqual([r["kind"] for r in layout.ranges()], ["routing", "bulk"])

    def test_graph_over_rabitq_codes_with_refine_rows(self):
        body, _ = graph(self.n, 4)
        codes = np.full((self.n, 10), 3, dtype=np.uint8)
        storage = (b"Ixrr" + header(self.d, self.n) + struct.pack("<QQiQ", self.d, 10, 1, 4) + vec(codes)
                   + vec(np.zeros(self.d, dtype="<f4")) + b"\x04")
        rows = np.full((self.n, self.d), 9, dtype=np.uint8)
        base = b"IHNr" + header(self.d, self.n) + body + storage + b"\x01"
        layout = fl.read(self.write("hnsw.faiss", refine(self.d, self.n, base, sq8(self.d, rows))))
        self.assertEqual(layout.records, ["IxRF", "IHNr", "Ixrr", "IxSQ"])
        self.assertEqual(layout.part("storage").length, codes.nbytes)
        self.assertEqual(layout.part("refine").length, rows.nbytes)

    def test_bad_files(self):
        path, raw, _ = self.ivf_file()

        with self.assertRaises(ValueError):
            fl.read(self.write("trailing.faiss", raw + b"\0"))

        with self.assertRaises(ValueError):
            fl.read(self.write("short.faiss", raw[:-10]))

        with self.assertRaises(ValueError):
            fl.read(self.write("unknown.faiss", b"IxZZ" + raw[4:]))

        huge = b"IxF2" + header(self.d, 1) + struct.pack("<Q", 1 << 60)

        with self.assertRaises(ValueError):
            fl.read(self.write("huge.faiss", huge))

    def test_merged_ranges(self):
        layout = fl.Layout(path="x", size=100, parts=[fl.Part("lists", 10, 20), fl.Part("refine", 30, 10),
                                                       fl.Part("coarse", 50, 10)])
        self.assertEqual(layout.ranges(), [
            {"offset": 0, "length": 10, "kind": "routing"},
            {"offset": 10, "length": 30, "kind": "bulk"},
            {"offset": 40, "length": 60, "kind": "routing"},
        ])

    def test_loaded_parts_stand_apart(self):
        layout = fl.Layout(path="x", size=100, parts=[fl.Part("rotation", 4, 20), fl.Part("coarse", 24, 10),
                                                       fl.Part("lists", 40, 20), fl.Part("refine", 60, 10)])
        self.assertEqual(layout.ranges(frozenset({"rotation", "refine"})), [
            {"offset": 0, "length": 4, "kind": "routing"},
            {"offset": 4, "length": 20, "kind": "routing", "loaded": True},
            {"offset": 24, "length": 16, "kind": "routing"},
            {"offset": 40, "length": 20, "kind": "bulk"},
            {"offset": 60, "length": 10, "kind": "bulk", "loaded": True},
            {"offset": 70, "length": 30, "kind": "routing"},
        ])
        self.assertEqual(layout.ranges(), [
            {"offset": 0, "length": 40, "kind": "routing"},
            {"offset": 40, "length": 30, "kind": "bulk"},
            {"offset": 70, "length": 30, "kind": "routing"},
        ])

    def test_array_lists(self):
        lists = fl.array_lists(100, np.array([2, 0, 1]), 4)
        self.assertEqual(lists.codes.tolist(), [100, 124, 124])
        self.assertEqual(lists.ids.tolist(), [108, 124, 128])


@unittest.skipUnless(HAS_FAISS, "faiss isn't installed")
class FaissFileTest(unittest.TestCase):
    """Files that faiss itself writes, with every part checked against the arrays faiss holds."""

    @classmethod
    def setUpClass(cls):
        import faiss

        cls.faiss = faiss
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        rng = np.random.default_rng(0)
        cls.d = 32
        x = rng.standard_normal((2000, cls.d)).astype(np.float32)
        cls.x = x / np.linalg.norm(x, axis=1, keepdims=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def raw(self, path: Path, offset: int, length: int) -> bytes:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def ivf(self, spec: str) -> object:
        faiss = self.faiss
        index = faiss.index_factory(self.d, spec)
        index.train(self.x)
        index.add(self.x)
        return index

    def check_lists(self, path: Path, layout: fl.Layout, index: object) -> None:
        faiss = self.faiss
        ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        source = fl.ondisk_file(path, layout) or path

        for i in range(ivf.nlist):
            n = int(ivf.invlists.list_size(i))
            self.assertEqual(n, layout.lists.sizes[i])

            if n:
                codes = faiss.rev_swig_ptr(ivf.invlists.get_codes(i), n * ivf.code_size).tobytes()
                ids = faiss.rev_swig_ptr(ivf.invlists.get_ids(i), n).tobytes()
                self.assertEqual(self.raw(source, int(layout.lists.codes[i]), len(codes)), codes)
                self.assertEqual(self.raw(source, int(layout.lists.ids[i]), len(ids)), ids)

    def test_mapped_ivf_with_refine(self):
        faiss = self.faiss

        for bits in ("", "4"):
            index = self.ivf(f"RR{self.d},IVF16,RaBitQ{bits}")
            rows = faiss.IndexScalarQuantizer(self.d, faiss.ScalarQuantizer.QT_8bit)
            rows.train(self.x)
            rows.add(self.x)
            path = self.dir / f"ivf{bits}.faiss"
            faiss.write_index(faiss.IndexRefine(index, rows), str(path))
            layout = fl.read(path)
            self.check_lists(path, layout, index)
            ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
            coarse = layout.part("coarse")
            self.assertEqual(self.raw(path, coarse.offset, coarse.length), ivf.quantizer.reconstruct_n(0, 16).tobytes())
            part = layout.part("refine")
            self.assertEqual(self.raw(path, part.offset, part.length), faiss.vector_to_array(rows.codes).tobytes())

    def test_graph_quantizer(self):
        index = self.ivf(f"RR{self.d},IVF16_HNSW32,RaBitQ")
        path = self.dir / "graphq.faiss"
        self.faiss.write_index(index, str(path))
        layout = fl.read(path)
        self.check_lists(path, layout, index)
        self.assertEqual(len(layout.named("coarse_graph")), 3)

    def test_ondisk_lists(self):
        faiss = self.faiss
        from faiss.contrib.ondisk import merge_ondisk

        index = self.ivf(f"RR{self.d},IVF16,RaBitQ")
        shard = self.dir / "shard.faiss"
        faiss.write_index(index, str(shard))
        trained = faiss.deserialize_index(faiss.serialize_index(index))
        trained.reset()
        merge_ondisk(trained, [str(shard)], str(self.dir / "lists.ivfdata"))
        path = self.dir / "ivf.index"
        faiss.write_index(trained, str(path))
        layout = fl.read(path)
        self.assertEqual(fl.ondisk_file(path, layout), self.dir / "lists.ivfdata")
        self.check_lists(path, layout, trained)

    def test_graphs(self):
        faiss = self.faiss

        for spec in ("HNSW16,SQ8", "HNSW16,Flat", "HNSW16,RaBitQ4,Refine(SQ8)"):
            index = faiss.index_factory(self.d, spec)
            index.train(self.x)
            index.add(self.x)
            path = self.dir / (spec.replace(",", "-").replace("(", "").replace(")", "") + ".faiss")
            faiss.write_index(index, str(path))
            layout = fl.read(path)
            graph_parts = layout.named("graph")
            self.assertEqual(len(graph_parts), 3)

            if spec.startswith("HNSW16,RaBitQ"):
                self.assertIsNotNone(layout.part("refine"))
                continue

            hnsw = index.hnsw
            want = [faiss.vector_to_array(a).tobytes() for a in (hnsw.levels, hnsw.offsets, hnsw.neighbors)]
            self.assertEqual([self.raw(path, p.offset, p.length) for p in graph_parts], want)
            storage = faiss.downcast_index(index.storage)
            part = layout.part("storage")
            self.assertEqual(self.raw(path, part.offset, part.length), faiss.vector_to_array(storage.codes).tobytes())

    def test_fastscan_is_unknown(self):
        index = self.ivf(f"RR{self.d},IVF16,RaBitQfs")
        path = self.dir / "fastscan.faiss"
        self.faiss.write_index(index, str(path))

        with self.assertRaises(ValueError):
            fl.read(path)


if __name__ == "__main__":
    unittest.main()
