import io
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np

from adapters import paimon
import rusage


def header(dim: int = 768, nlist: int = 312, magic: bytes = paimon.MAGIC) -> bytes:
    """A 64-byte header laid out like the library's, with the fields this adapter reads."""
    head = bytearray(paimon.HEADER_BYTES)
    head[:4] = magic
    struct.pack_into("<5I", head, 4, 1, dim, dim, nlist, 1)
    return bytes(head)


class SettingsTest(unittest.TestCase):
    def test_metric(self):
        self.assertEqual(paimon.paimon_metric("l2", True), "l2")
        self.assertEqual(paimon.paimon_metric("ip", False), "inner_product")
        self.assertEqual(paimon.paimon_metric("cosine", True), "inner_product")
        self.assertEqual(paimon.paimon_metric("cosine", False), "cosine")

    def test_nlist(self):
        self.assertEqual(paimon.nlist_of({"nlist": 100, "vpl": 5}, 50000), 100)
        self.assertEqual(paimon.nlist_of({"vpl": 160}, 50000), 312)
        self.assertIsNone(paimon.nlist_of({}, 50000))

    def test_options(self):
        out = paimon.build_options({"bits": 8, "vpl": 160}, 768, 50000, "inner_product")
        self.assertEqual(out, {"index.type": "ivf_rq", "dimension": "768", "rq.bits": "8", "metric": "inner_product",
                               "ivf.coarse-assignment": "exact", "nlist": "312"})
        auto = paimon.build_options({"options": {"ivf.train.max-points-per-centroid": 16}}, 768, 50000, "l2")
        self.assertEqual((auto["nlist"], auto["expected-vector-count"], auto["rq.bits"]), ("auto", "50000", "4"))
        self.assertEqual(auto["ivf.train.max-points-per-centroid"], "16")
        self.assertTrue(all(isinstance(v, str) for v in auto.values()))

    def test_training_rows(self):
        self.assertIsNone(paimon.training_rows(1000, None))
        self.assertIsNone(paimon.training_rows(1000, 5000))
        ids = paimon.training_rows(1000, 10)
        self.assertEqual(len(ids), 10)
        self.assertEqual((ids[0], ids[-1]), (0, 999))
        self.assertTrue(np.all(np.diff(ids) > 0))

    def test_checks(self):
        paimon.check_build({"bits": 4, "vpl": 160})
        paimon.check_search({"nprobe": 8, "read_threads": 16})

        with self.assertRaises(ValueError):
            paimon.check_build({"nbits": 4})

        with self.assertRaises(ValueError):
            paimon.check_search({"nprobes": 8})

        with self.assertRaises(ValueError):
            paimon.check_search({"read_threads": 8})

    def test_unknown_variant_fails_before_the_import(self):
        adapter = paimon.Paimon()
        adapter.variant = "mmap"

        with self.assertRaises(ValueError):
            adapter.load()


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_header(self):
        self.assertEqual(paimon.header_fields(header(768, 312)), {"version": 1, "dim": 768, "nlist": 312})

        with self.assertRaises(ValueError):
            paimon.header_fields(header(magic=b"XXXX"))

        with self.assertRaises(ValueError):
            paimon.header_fields(header()[:40])

    def test_routing_formula(self):
        # The header, 312 centroids of 768 float32 values, and a 16-byte offset entry per list.
        self.assertEqual(paimon.routing_formula(768, 312), 64 + 958464 + 4992)

    def test_measured_routing(self):
        self.assertEqual(paimon.measured_routing([[(0, 64)], [(64, 963456)], [(970000, 5)]]), 963520)

        with self.assertRaises(ValueError):
            paimon.measured_routing([[(0, 64)], [(128, 100)]])

    def test_files(self):
        size = paimon.routing_formula(16, 4) + 1000
        (self.root / paimon.INDEX_FILE).write_bytes(header(16, 4) + b"\0" * (size - 64))
        files = paimon.Paimon().files(self.root)
        self.assertEqual(len(files), 1)
        # The reader keeps the routing span in memory once it has read it.
        self.assertEqual(files[0]["ranges"], [{"offset": 0, "length": size - 1000, "kind": "routing", "loaded": True},
                                              {"offset": size - 1000, "length": 1000, "kind": "bulk"}])
        (self.root / paimon.META).write_text(json.dumps({"routing_end": 100}))
        files = paimon.Paimon().files(self.root)
        self.assertEqual([r["length"] for r in files[0]["ranges"]], [100, size - 100])
        self.assertEqual(Path(files[1]["path"]).name, paimon.META)
        self.assertEqual(files[1]["ranges"][0]["kind"], "routing")

    def test_the_read_pool_runs_at_the_class_a_thread_starts_with(self):
        adapter = paimon.Paimon()
        pool = adapter.pool_for(2)

        try:
            classes = {pool.submit(rusage.qos_name).result() for _ in range(4)}
        finally:
            pool.shutdown()

        # Only the timed thread asks for the user-interactive class, as in every other system.
        self.assertEqual(classes, {"default"} if rusage.MACOS else {""})


class InputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "file.bin"
        self.bytes = np.random.default_rng(3).integers(0, 256, 1 << 20, dtype=np.uint8).tobytes()
        self.path.write_bytes(self.bytes)
        self.pool = ThreadPoolExecutor(4)
        self.ranges = [(0, 64), (4096, 10000), (len(self.bytes) - 7, 7), (123, 0), (500000, 70000)]

    def tearDown(self):
        self.pool.shutdown()
        self.tmp.cleanup()

    def inputs(self) -> list[paimon.Input]:
        out = [paimon.SerialInput(self.path), paimon.PoolInput(self.path, self.pool)]

        if sys.platform == "darwin":
            out.append(paimon.AdviseInput(self.path))

        return out

    def test_every_input_returns_the_ranges(self):
        for source in self.inputs():
            with self.subTest(type(source).__name__):
                chunks = source.pread_many(self.ranges)
                self.assertEqual([bytes(c) for c in chunks], [self.bytes[o : o + n] for o, n in self.ranges])
                source.pread_many([(8, 8)])
                rounds = source.log.between(0, source.log.count)
                self.assertEqual([(r[0], r[1]) for r in rounds], [(5, 64 + 10000 + 7 + 70000), (1, 8)])
                self.assertTrue(all(r[2] >= 0 for r in rounds))
                source.close()

    def test_a_range_past_the_end_fails(self):
        for source in self.inputs():
            with self.subTest(type(source).__name__):
                with self.assertRaises(OSError), redirect_stderr(io.StringIO()) as err:
                    source.pread_many([(len(self.bytes) - 4, 8), (0, 4)])

                self.assertIn("OSError", err.getvalue())

                source.close()

    def test_short_reads_are_completed(self):
        real = os.pread

        def short(fd: int, n: int, offset: int) -> bytes:
            return real(fd, min(n, 1000), offset)

        source = paimon.SerialInput(self.path)

        with mock.patch.object(paimon.os, "pread", side_effect=short):
            chunk = paimon.pread_full(source.fd, 100, 4500)

        self.assertEqual(chunk, self.bytes[100:4600])
        source.close()

    @unittest.skipUnless(sys.platform == "darwin", "F_RDADVISE is a macOS call")
    def test_advise_splits_lengths_past_the_int_limit(self):
        calls = []

        with mock.patch.object(paimon.fcntl, "fcntl", side_effect=lambda fd, cmd, arg: calls.append(arg)):
            paimon.advise(3, 10, paimon.MAX_ADVISE + 5)

        self.assertEqual([paimon.RADVISORY.unpack(a) for a in calls], [(10, paimon.MAX_ADVISE),
                                                                         (10 + paimon.MAX_ADVISE, 5)])
        self.assertEqual(paimon.RADVISORY.size, 16)


class RoundLogTest(unittest.TestCase):
    def test_numbering_survives_dropped_rounds(self):
        log = paimon.RoundLog(keep=3)

        for i in range(5):
            log.add([(0, i + 1)], float(i))

        self.assertEqual(log.count, 5)
        self.assertEqual([r[1] for r in log.between(0, 5)], [3, 4, 5])
        self.assertEqual([r[1] for r in log.between(3, 4)], [4])
        self.assertEqual(log.between(5, 5), [])

    def test_concurrent_rounds_all_count(self):
        log = paimon.RoundLog()

        def work() -> None:
            for _ in range(500):
                log.add([(0, 1), (5, 2)], 1.0)

        threads = [threading.Thread(target=work) for _ in range(8)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(log.count, 4000)
        self.assertEqual(sum(r[1] for r in log.between(0, log.count)), 4000 * 3)

    def test_round_stats(self):
        out = paimon.round_stats([(1, 64, 3.0), (1, 1000, 20.5), (8, 6000, 100.0)])
        self.assertEqual((out["bytes_requested"], out["bytes_scanned"]), (7064, 7064))
        self.assertEqual(out["stages"], {"read_us": 123.5, "rounds": 3, "round_ranges": [1, 1, 8],
                                         "round_bytes": [64, 1000, 6000], "round_us": [3.0, 20.5, 100.0]})
        self.assertEqual(paimon.round_stats([])["stages"]["rounds"], 0)


class PoolTest(unittest.TestCase):
    def test_prestart_starts_every_thread(self):
        adapter = paimon.Paimon()
        adapter.prestart(6)
        names = [t.name for t in threading.enumerate() if t.name.startswith("pread")]
        self.assertEqual(len(names), 6)
        self.assertIs(adapter.pool_for(6), adapter.pool)
        old = adapter.pool
        self.assertIsNot(adapter.pool_for(3), old)
        adapter.pool.shutdown()


if __name__ == "__main__":
    unittest.main()
