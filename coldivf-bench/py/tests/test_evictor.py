import os
import tempfile
import unittest
from pathlib import Path

import evictor

PAGE = evictor.PAGE


def read(path: Path, start: int, length: int) -> None:
    with open(path, "rb") as f:
        f.seek(start)
        f.read(length)


@unittest.skipUnless(evictor.MACOS, "eviction through msync is written for macOS")
class HeldMapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "f.bin"
        self.path.write_bytes(os.urandom(64 * PAGE + 100))

    def tearDown(self):
        self.tmp.cleanup()

    def test_evict_and_check(self):
        held = evictor.HeldMap(self.path)
        self.assertEqual(held.pages, 65)
        held.sync()
        held.evict()
        self.assertEqual(held.resident(), (0, 65))
        read(self.path, 3 * PAGE, 2 * PAGE)
        # Another process may read the first two pages of a newly mapped file back, so the check starts past them.
        cached, total = held.resident(2 * PAGE, 10 * PAGE)
        self.assertEqual((cached, total), (2, 8))
        self.assertEqual(held.resident(3 * PAGE, 5 * PAGE)[0], 2)
        held.close()

    def test_spans(self):
        held = evictor.HeldMap(self.path)
        self.assertEqual(held._span(10, 20, inward=False), (0, PAGE))
        self.assertEqual(held._span(10, 2 * PAGE + 5, inward=True), (PAGE, 2 * PAGE))
        # The last page holds the end of the file, so a range that reaches the end owns it.
        self.assertEqual(held._span(64 * PAGE, None, inward=True), (64 * PAGE, 65 * PAGE))
        held.close()

    def test_inward_eviction_keeps_shared_pages(self):
        held = evictor.HeldMap(self.path)
        held.sync()
        read(self.path, 0, 8 * PAGE)
        held.evict(PAGE + 1, 6 * PAGE + 1, inward=True)
        self.assertEqual(held.resident(0, 8 * PAGE)[0], 4)
        held.close()

    def test_changes(self):
        held = evictor.HeldMap(self.path)
        self.assertFalse(held.changed())

        with open(self.path, "ab") as f:
            f.write(b"more")

        self.assertTrue(held.changed())
        held.close()

    def test_empty_file(self):
        empty = self.dir / "empty"
        empty.write_bytes(b"")
        held = evictor.HeldMap(empty)
        self.assertEqual(held.resident(), (0, 0))
        held.evict()
        held.close()


@unittest.skipUnless(evictor.MACOS, "eviction through msync is written for macOS")
class EvictorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "sub").mkdir()
        self.a = self.dir / "a.bin"
        self.b = self.dir / "sub" / "b.bin"
        self.a.write_bytes(os.urandom(8 * PAGE))
        self.b.write_bytes(os.urandom(4 * PAGE))

    def tearDown(self):
        self.tmp.cleanup()

    def test_make_cold_empties_every_file(self):
        with evictor.Evictor([self.dir]) as ev:
            self.assertEqual(len(ev.maps), 2)
            read(self.a, 0, 8 * PAGE)
            res = ev.make_cold(wait=0.05)
            self.assertEqual(res.resident_pages, 0)
            self.assertEqual(res.total_pages, 12)
            self.assertEqual(res.files, 2)
            read(self.b, 0, PAGE)
            after = ev.residency()
            self.assertEqual(after.touched, 1)
            self.assertEqual(after.resident_bytes, PAGE)

    def test_refresh_follows_the_folder(self):
        with evictor.Evictor([self.dir]) as ev:
            c = self.dir / "c.bin"
            c.write_bytes(b"x" * 10)
            self.assertEqual(ev.refresh(), [str(c)])
            self.assertEqual(ev.refresh(), [])
            c.unlink()
            self.assertEqual(ev.refresh(), [str(c)])
            self.assertNotIn(str(c), ev.maps)

    def test_watched_file_that_appears_later(self):
        log = self.dir / "a.bin.delta"

        with evictor.Evictor([self.a, log]) as ev:
            self.assertEqual(len(ev.maps), 1)
            log.write_bytes(b"x" * 100)
            ev.refresh()
            self.assertIn(str(log), ev.maps)

    def test_ranges(self):
        with evictor.Evictor([self.a]) as ev:
            ev.sync()
            read(self.a, 0, 8 * PAGE)
            ranges = [evictor.Range(self.a, 2 * PAGE, 3 * PAGE)]
            ev.evict_ranges(ranges)
            self.assertEqual(ev.resident_in(ranges), (0, 3))
            self.assertEqual(ev.residency().resident_pages, 5)

            with self.assertRaises(KeyError):
                ev.resident_in([evictor.Range(self.b, 0, 10)])

    def test_residency_of(self):
        read(self.a, 0, PAGE)
        res = evictor.residency_of([self.a])
        self.assertGreaterEqual(res.resident_pages, 1)

    def files(self) -> list[dict]:
        """a.bin with its pages 2 to 5 loaded, and b.bin, which ends in a partial page, loaded whole."""
        return [{"path": str(self.a), "bytes": 8 * PAGE,
                 "ranges": [{"offset": 0, "length": 2 * PAGE, "kind": "routing"},
                            {"offset": 2 * PAGE, "length": 3 * PAGE, "kind": "routing", "loaded": True},
                            {"offset": 5 * PAGE, "length": 3 * PAGE, "kind": "bulk"}]},
                {"path": str(self.b), "bytes": 3 * PAGE + 10,
                 "ranges": [{"offset": 0, "length": 3 * PAGE + 10, "kind": "bulk", "loaded": True}]},
                {"path": str(self.dir / "missing"), "bytes": 0, "ranges": []}]

    def test_loaded_ranges(self):
        ranges = evictor.loaded_ranges(self.files())
        self.assertEqual([(r.path.name, r.offset, r.length) for r in ranges],
                         [("a.bin", 2 * PAGE, 3 * PAGE), ("b.bin", 0, 3 * PAGE + 10)])

    def test_loaded_ranges_that_touch_share_their_pages(self):
        # A header of 4 KiB and the rest, both loaded, share the first page, which neither holds whole on its own.
        files = [{"path": str(self.a), "ranges": [{"offset": 4096, "length": 8 * PAGE - 4096, "kind": "bulk",
                                                   "loaded": True},
                                                  {"offset": 0, "length": 4096, "kind": "routing", "loaded": True}]}]
        ranges = evictor.loaded_ranges(files)
        self.assertEqual([(r.offset, r.length, r.kind) for r in ranges], [(0, 8 * PAGE, "routing")])

        with evictor.Evictor([self.a]) as ev:
            ev.sync()
            read(self.a, 0, 8 * PAGE)
            self.assertEqual(ev.resident_bytes_in(ranges), 8 * PAGE)

    def test_resident_bytes_in_count_whole_pages_up_to_the_end_of_the_file(self):
        self.b.write_bytes(os.urandom(3 * PAGE + 10))

        with evictor.Evictor([self.a, self.b]) as ev:
            ev.sync()
            ev.evict()
            read(self.a, 0, 8 * PAGE)
            read(self.b, 0, 3 * PAGE + 10)
            loaded = evictor.loaded_ranges(self.files())
            self.assertEqual(ev.resident_bytes_in(loaded), 3 * PAGE + 3 * PAGE + 10)
            self.assertEqual(ev.resident_bytes_in([evictor.Range(self.dir / "missing", 0, PAGE)]), 0)
            # A range inside one page holds no whole page.
            self.assertEqual(ev.resident_bytes_in([evictor.Range(self.a, 10, 100)]), 0)

    def test_split_and_drop_leave_the_other_ranges(self):
        self.b.write_bytes(os.urandom(3 * PAGE + 10))
        files = self.files()

        with evictor.Evictor([self.a, self.b]) as ev:
            ev.sync()

        read(self.a, 0, 8 * PAGE)
        read(self.b, 0, 3 * PAGE + 10)
        self.assertEqual(evictor.resident_split(files), (5 * PAGE, 6 * PAGE + 10))
        self.assertEqual(evictor.drop_loaded(files), 6 * PAGE + 10)
        self.assertEqual(evictor.resident_split(files)[0], 5 * PAGE)

        # Another process may read the first two pages of a newly mapped file back, so those aren't checked.
        with evictor.Evictor([self.a, self.b]) as ev:
            a, b = ev.maps[str(self.a)], ev.maps[str(self.b)]
            self.assertEqual((a.resident(2 * PAGE, 5 * PAGE)[0], b.resident(2 * PAGE)[0]), (0, 0))

        self.assertEqual(evictor.drop_loaded([{"path": str(self.a), "ranges": []}]), 0)


if __name__ == "__main__":
    unittest.main()
