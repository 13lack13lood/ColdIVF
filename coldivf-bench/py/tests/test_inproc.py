import unittest

import numpy as np

from adapters import inproc


class SettingsTest(unittest.TestCase):
    DEFAULTS = {"bits": inproc.REQUIRED, "nlist": None, "niter": 25}

    def test_given_keys_win_over_defaults(self):
        got = inproc.settings({"bits": 5, "niter": 10}, self.DEFAULTS, "test settings")
        self.assertEqual(got, {"bits": 5, "nlist": None, "niter": 10})

    def test_unknown_keys_are_errors(self):
        with self.assertRaisesRegex(ValueError, r"unknown test settings \['nprobes'\]"):
            inproc.settings({"bits": 1, "nprobes": 3}, self.DEFAULTS, "test settings")

    def test_required_keys_must_be_given(self):
        with self.assertRaisesRegex(ValueError, r"need \['bits'\]"):
            inproc.settings({}, self.DEFAULTS, "test settings")

        with self.assertRaisesRegex(ValueError, r"need \['bits'\]"):
            inproc.settings(None, self.DEFAULTS, "test settings")

    def test_the_input_is_not_changed(self):
        given = {"bits": 3}
        inproc.settings(given, self.DEFAULTS, "test settings")
        self.assertEqual(given, {"bits": 3})


class IdsTest(unittest.TestCase):
    def test_empty_slots_become_minus_one(self):
        row = np.array([7, 2**32 - 1, 3, 2**32 - 1], dtype=np.uint32)
        np.testing.assert_array_equal(inproc.as_ids(row, 4, 2**32 - 1), [7, -1, 3, -1])

    def test_short_rows_are_padded_and_long_rows_cut(self):
        np.testing.assert_array_equal(inproc.as_ids([5, 9], 4), [5, 9, -1, -1])
        np.testing.assert_array_equal(inproc.as_ids(np.arange(6), 3), [0, 1, 2])
        np.testing.assert_array_equal(inproc.as_ids([], 2), [-1, -1])

    def test_ids_are_int64(self):
        self.assertEqual(inproc.as_ids(np.array([1], dtype=np.uint32), 2).dtype, np.int64)

    def test_a_large_id_survives_without_an_empty_marker(self):
        np.testing.assert_array_equal(inproc.as_ids(np.array([2**32 - 1], dtype=np.int64), 1), [2**32 - 1])


class UnitTest(unittest.TestCase):
    def test_unit_rows_pass(self):
        rows = np.random.default_rng(0).standard_normal((500, 16)).astype(np.float32)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        inproc.check_unit(rows)
        inproc.check_unit(rows[:0])

    def test_a_long_row_fails(self):
        rows = np.random.default_rng(0).standard_normal((500, 16)).astype(np.float32)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        rows[-1] *= 1.01

        # The last row is always among the rows checked.
        with self.assertRaisesRegex(ValueError, "row 499 has length 1.01"):
            inproc.check_unit(rows, samples=10)


if __name__ == "__main__":
    unittest.main()
