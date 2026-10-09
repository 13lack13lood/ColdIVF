import unittest

import numpy as np

from adapters import turbovec as tv
from tests.inproc_runner import LibraryCase, has, recall10


class LogicTest(unittest.TestCase):
    def test_sample_rows(self):
        rows = tv.sample_rows(1000, 50, seed=3)
        self.assertEqual(len(rows), 50)
        self.assertEqual(len(set(rows.tolist())), 50)
        self.assertTrue(np.all(np.diff(rows) > 0))
        self.assertTrue(np.all((rows >= 0) & (rows < 1000)))
        np.testing.assert_array_equal(rows, tv.sample_rows(1000, 50, seed=3))
        self.assertFalse(np.array_equal(rows, tv.sample_rows(1000, 50, seed=4)))
        np.testing.assert_array_equal(tv.sample_rows(5, 9, seed=1), np.arange(5))

    def test_code_bytes(self):
        self.assertEqual(tv.code_bytes(10, 768, 2), 10 * 192)
        self.assertEqual(tv.code_bytes(10, 768, 4), 10 * 384)
        # At 3 bits a byte holds two codes, as at 4 bits.
        self.assertEqual(tv.code_bytes(10, 768, 3), 10 * 384)


@unittest.skipUnless(has("turbovec"), "needs turbovec")
class TurboVecTest(LibraryCase):
    module = "adapters.turbovec"

    def test_four_bits(self):
        index, msg = self.built("b4", {"bits": 4})
        fields, info = msg["fields"], msg["info"]
        self.assertFalse(fields["exact_f32_stored"])
        self.assertEqual(fields["stored_vector_bytes"], tv.code_bytes(self.rows, 64, 4))
        self.assertEqual(fields["extra"]["calibration"], "uncalibrated")
        self.assertEqual(fields["extra"]["vectors"], self.rows)
        self.assertEqual((info["system"], info["batch_api"]), ("turbovec", True))
        self.assertTrue((index / "index.tv").is_file())
        found = self.settings(index, [{}])
        # 64 dimensions are few for TurboQuant, so this only shows that the ids name the right rows. Random ids would
        # find ~0.005 of the neighbors.
        self.assertGreaterEqual(recall10(self.ds, found[0]), 0.6)
        found = self.settings(index, [{}], threads=2)
        self.assertGreater(found[0]["fields"]["qps"], 0)
        self.check_cold(index, {})
        got, _ = self.runner.warm(index, [{"rerank": 2}])
        self.check_error(got, "unknown turbovec search settings")

    def test_calibrated_two_bits(self):
        index, msg = self.built("b2", {"bits": 2, "calibrate": 256})
        self.assertEqual(msg["fields"]["extra"]["calibration"], "calibrated")
        found = self.settings(index, [{}])
        self.assertGreaterEqual(recall10(self.ds, found[0]), 0.25)

    def test_l2_and_bad_bits(self):
        msg, _ = self.runner.build(self.root / "l2", {"bits": 4}, "--metric", "l2")
        self.assertEqual(msg["event"], "unsupported")
        msg, _ = self.runner.build(self.root / "b8", {"bits": 8})
        self.assertIn("bits is 8", msg["message"])

    def test_overhead(self):
        got, err = self.runner.run("overhead", *self.runner.common())
        self.assertEqual(got[0]["event"], "overhead", err[-2000:])
        self.assertGreater(got[0]["fields"]["overhead_us"], 0)


if __name__ == "__main__":
    unittest.main()
