import json
import unittest

from adapters import sqlite_vec as sv
from tests.inproc_runner import LibraryCase, has, recall10


class SqlTest(unittest.TestCase):
    def test_code_list(self):
        self.assertEqual(sv.code_list("f32"), [])
        self.assertEqual(sv.code_list("int8"), ["int8"])
        self.assertEqual(sv.code_list(["bit", "f32", "int8"]), ["int8", "bit"])

        with self.assertRaisesRegex(ValueError, "unknown codes"):
            sv.code_list(["int4"])

    def test_column_bytes(self):
        self.assertEqual([sv.column_bytes(c, 768) for c in sv.CODES], [3072, 768, 96])
        self.assertEqual(sv.column_bytes("bit", 12), 2)

    def test_create_and_insert(self):
        self.assertEqual(sv.create_sql(8, []), "create virtual table vectors using vec0(embedding float[8])")
        self.assertEqual(sv.create_sql(8, ["int8", "bit"], 512),
                         "create virtual table vectors using vec0(embedding float[8], embedding_int8 int8[8], "
                         "embedding_bit bit[8], chunk_size=512)")
        self.assertEqual(sv.insert_sql(["bit"]), "insert into vectors(rowid, embedding, embedding_bit) values "
                                                 "(?, ?, vec_quantize_binary(?))")

    def test_float32_rows_in_an_auxiliary_column(self):
        self.assertEqual(sv.create_sql(8, ["int8"], rescore="aux"),
                         "create virtual table vectors using vec0(embedding_int8 int8[8], +embedding blob)")
        self.assertEqual(sv.search_settings({}, ["bit"], "aux")["codes"], "bit")

        with self.assertRaisesRegex(ValueError, "no exact scan"):
            sv.search_settings({"codes": "f32"}, ["bit"], "aux")

    def test_queries(self):
        exact = sv.query_sql("f32")
        self.assertIn("embedding match ? and k = ?", exact)
        self.assertEqual(exact.count("?"), len(sv.query_args("f32", b"v", 10, 4)))
        coarse = sv.query_sql("int8")
        self.assertIn("embedding_int8 match vec_quantize_int8(?, 'unit') and k = ?", coarse)
        self.assertIn("order by vec_distance_l2(embedding, ?) limit ?", coarse)
        self.assertEqual(sv.query_args("int8", b"v", 10, 4), (b"v", 40, b"v", 10))
        self.assertEqual(coarse.count("?"), 4)
        self.assertIn("embedding_bit match vec_quantize_binary(?)", sv.query_sql("bit"))

    def test_scan_counts(self):
        self.assertEqual(sv.scan_counts("f32", 1000, 768, 10, 1), {"dist_evals": 1000, "bytes_scanned": 3072000})
        self.assertEqual(sv.scan_counts("bit", 1000, 768, 10, 8),
                         {"dist_evals": 1080, "bytes_scanned": 96000 + 80 * 3072})
        self.assertEqual(sv.scan_counts("int8", 20, 8, 10, 8)["dist_evals"], 40)

    def test_search_settings(self):
        self.assertEqual(sv.search_settings({}, [])["codes"], "f32")
        self.assertEqual(sv.search_settings({}, ["int8", "bit"])["codes"], "int8")
        self.assertEqual(sv.search_settings({"codes": "f32"}, ["bit"])["codes"], "f32")

        with self.assertRaisesRegex(ValueError, "no bit codes"):
            sv.search_settings({"codes": "bit"}, ["int8"])

        with self.assertRaisesRegex(ValueError, "oversample"):
            sv.search_settings({"oversample": 0}, ["int8"])

        with self.assertRaisesRegex(ValueError, "unknown"):
            sv.search_settings({"oversampling": 2}, ["int8"])


@unittest.skipUnless(has("sqlite_vec"), "needs sqlite-vec")
class SqliteVecTest(LibraryCase):
    module = "adapters.sqlite_vec"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.index = cls.root / "both"
        msg, err = cls.runner.build(cls.index, {"codes": ["int8", "bit"], "page_size": 16384})
        assert msg.get("event") == "build", err[-2000:]
        cls.msg = msg

    def test_build(self):
        fields = self.msg["fields"]
        self.assertTrue(fields["exact_f32_stored"])
        self.assertEqual(fields["stored_vector_bytes"], self.rows * (256 + 64 + 8))
        meta = json.loads((self.index / "meta.json").read_text())
        self.assertEqual((meta["codes"], meta["page_size"], meta["rows"]), (["int8", "bit"], 16384, self.rows))
        self.assertTrue((self.index / "vec0.db").is_file())
        self.assertEqual(self.msg["info"]["qps_mode"], "processes")

    def test_scans(self):
        found = self.settings(self.index, [{"codes": "f32"}, {"codes": "int8", "oversample": 8},
                                           {"codes": "bit", "oversample": 8}, {}])
        self.assertEqual(recall10(self.ds, found[0]), 1.0)
        self.assertGreaterEqual(recall10(self.ds, found[1]), 0.95)
        self.assertGreaterEqual(recall10(self.ds, found[2]), 0.5)
        self.assertEqual(found[0]["fields"]["dist_evals"], self.rows)
        self.assertEqual(found[1]["fields"]["dist_evals"], self.rows + 80)
        # Without codes a search scans the build's first coarse codes.
        self.assertEqual(found[3]["fields"]["bytes_scanned"], self.rows * 64 + 10 * 256)

    def test_cold_and_rates(self):
        self.check_cold(self.index, {"codes": "bit", "oversample": 4, "mmap_size": 0})
        found = self.settings(self.index, [{"codes": "f32"}], threads=2)
        self.assertEqual(found[0]["fields"]["processes"], 2)
        self.assertGreater(found[0]["fields"]["qps"], 0)

    def test_float32_rows_in_an_auxiliary_column(self):
        index, msg = self.built("aux", {"codes": ["int8", "bit"], "rescore": "aux"})
        self.assertEqual(msg["fields"]["extra"]["rescore"], "aux")
        found = self.settings(index, [{"codes": "int8", "oversample": 8}, {"codes": "bit", "oversample": 8}])
        self.assertGreaterEqual(recall10(self.ds, found[0]), 0.95)
        self.assertGreaterEqual(recall10(self.ds, found[1]), 0.5)
        self.check_cold(index, {"codes": "int8", "oversample": 4})
        got, _ = self.runner.warm(index, [{"codes": "f32"}])
        self.check_error(got, "no exact scan")
        msg, _ = self.runner.build(self.root / "aux-alone", {"codes": "f32", "rescore": "aux"})
        self.assertIn("needs int8 or bit codes", msg["message"])

    def test_errors(self):
        exact, _ = self.built("exact", {"codes": "f32"})
        got, _ = self.runner.warm(exact, [{"codes": "int8"}])
        self.check_error(got, "no int8 codes")
        msg, _ = self.runner.build(self.root / "ip", {}, "--metric", "ip")
        self.assertEqual(msg["event"], "unsupported")
        msg, _ = self.runner.build(self.root / "odd-pages", {"page_size": 10000})
        self.assertIn("not the 10000 asked for", msg["message"])

    def test_overhead(self):
        got, err = self.runner.run("overhead", *self.runner.common())
        self.assertEqual(got[0]["event"], "overhead", err[-2000:])
        self.assertGreater(got[0]["fields"]["overhead_us"], 0)


if __name__ == "__main__":
    unittest.main()
