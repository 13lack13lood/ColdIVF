import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
from adapters import diskann
from adapters.base import BuildContext, ServeContext, Unsupported
import data
import evictor

# A report of the build tool, trimmed to the fields the adapter reads.
REPORT = {
    "event": "build", "seconds": 12.5, "cpu_seconds": 40.2, "peak_footprint": 812345678, "bytes_read": 9000,
    "bytes_written": 260000000,
    "stages": [["PqConstruction", 3.25], ["InmemIndexBuild", 8.0], ["DiskLayout", 1.1]],
    "files": {"index_disk.index": 204804096, "index_pq_pivots.bin": 790000},
    "meta": {"rows": 50000, "dim": 768, "pq_bytes": 96, "metric": "l2"},
}


class SettingsTest(unittest.TestCase):
    def test_build_defaults_fill_in(self):
        config = diskann.build_config({})
        self.assertEqual((config["degree"], config["build_list"], config["pq_bytes"]), (64, 100, 96))
        self.assertEqual(config["metric"], "l2")
        self.assertIsNone(config["alpha"])
        self.assertEqual(diskann.build_flags(config), ["--degree", "64", "--list", "100", "--pq-bytes", "96",
                                                       "--saturate", "false", "--metric", "l2", "--ram-gb", "7.0"])

    def test_a_build_setting_overrides_its_default(self):
        config = diskann.build_config({"degree": 32, "alpha": 1.3, "saturate": True})
        flags = diskann.build_flags(config)
        self.assertEqual(flags[flags.index("--degree") + 1], "32")
        self.assertEqual(flags[flags.index("--alpha") + 1], "1.3")
        self.assertEqual(flags[flags.index("--saturate") + 1], "true")

    def test_a_build_on_pq_codes_passes_its_bytes(self):
        flags = diskann.build_flags(diskann.build_config({"build_pq_bytes": 768}))
        self.assertEqual(flags[flags.index("--build-pq-bytes") + 1], "768")
        self.assertNotIn("--build-pq-bytes", diskann.build_flags(diskann.build_config({})))

    def test_unknown_build_settings_are_refused(self):
        with self.assertRaises(Unsupported):
            diskann.build_config({"degre": 64})

    def test_search_settings(self):
        self.assertEqual(diskann.search_params({"list": 50, "beam": 4}, 10), (50, 4))
        self.assertEqual(diskann.search_params({"list": 10, "beam": 2, "cache_nodes": 5}, 10), (10, 2))

        for bad in ({"list": 50}, {"beam": 4}, {"list": 9, "beam": 4}, {"list": 50, "beam": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                diskann.search_params(bad, 10)

        with self.assertRaises(Unsupported):
            diskann.search_params({"list": 50, "beam": 4, "nprobe": 8}, 10)

    def test_only_c1_caches_nodes_by_default(self):
        self.assertEqual(diskann.cache_nodes({}, "c1"), diskann.C1_CACHE_NODES)
        self.assertEqual(diskann.cache_nodes({}, "c0"), 0)
        self.assertEqual(diskann.cache_nodes({}, "intent"), 0)
        self.assertEqual(diskann.cache_nodes({"cache_nodes": 7}, "c1"), 7)
        self.assertEqual(diskann.cache_nodes({"cache_nodes": 7}, "c0"), 7)

    def test_variants_name_readers(self):
        self.assertEqual(diskann.reader_name(""), "pool")

        for name in ("pool", "advise", "stock"):
            self.assertEqual(diskann.reader_name(name), name)

        with self.assertRaises(Unsupported):
            diskann.reader_name("mmap")

    def test_the_stock_reader_runs_only_as_the_control(self):
        adapter = diskann.DiskANN()
        adapter.variant = "stock"
        adapter.allow(1_000_000, 1)

        for rows, threads in ((1_000_001, 1), (1000, 2)):
            with self.assertRaises(Unsupported):
                adapter.allow(rows, threads)

        self.assertNotIn("c1", adapter.supported_regimes())
        adapter.variant = "pool"
        adapter.allow(20_000_000, 12)
        self.assertIn("c1", adapter.supported_regimes())


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        sizes = {"index_pq_pivots.bin": 100, "index_pq_compressed.bin": 960, "index_disk.index": 5 * 4096,
                 "meta.json": 30}

        for name, size in sizes.items():
            (self.dir / name).write_bytes(b"\0" * size)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_header_and_the_pq_data_are_routing(self):
        files = {Path(f["path"]).name: f for f in diskann.files_of(self.dir)}
        self.assertEqual(set(files), {"index_pq_pivots.bin", "index_pq_compressed.bin", "index_disk.index",
                                      "meta.json"})

        for name in ("index_pq_pivots.bin", "index_pq_compressed.bin", "meta.json"):
            self.assertEqual([r["kind"] for r in files[name]["ranges"]], ["routing"])
            self.assertEqual(files[name]["ranges"][0]["length"], files[name]["bytes"])
            # An open copies them into memory.
            self.assertTrue(files[name]["ranges"][0]["loaded"])

        disk = files["index_disk.index"]
        self.assertEqual(disk["ranges"], [{"offset": 0, "length": 4096, "kind": "routing"},
                                          {"offset": 4096, "length": 4 * 4096, "kind": "bulk"}])

    def test_the_stock_reader_loads_the_index_file_too(self):
        files = {Path(f["path"]).name: f for f in diskann.files_of(self.dir, "stock")}
        self.assertEqual(files["index_disk.index"]["ranges"],
                         [{"offset": 0, "length": 4096, "kind": "routing", "loaded": True},
                          {"offset": 4096, "length": 4 * 4096, "kind": "bulk", "loaded": True}])
        self.assertEqual(diskann.files_of(self.dir, "advise"), diskann.files_of(self.dir))

    def test_a_file_stands_for_its_folder(self):
        self.assertEqual(diskann.files_of(self.dir / "index_disk.index"), diskann.files_of(self.dir))

    def test_a_folder_without_the_index_file(self):
        (self.dir / "index_disk.index").unlink()

        with self.assertRaises(FileNotFoundError):
            diskann.files_of(self.dir)


@unittest.skipUnless(evictor.MACOS, "eviction through msync is written for macOS")
class WarmModeTest(unittest.TestCase):
    """The warm modes on an index folder of cached files: "whole" reads the index file, and "pq" drops its pages and
    keeps the PQ files the open holds."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        sizes = {"index_pq_pivots.bin": 100, "index_pq_compressed.bin": 16 * evictor.PAGE,
                 "index_disk.index": 64 * evictor.PAGE, "meta.json": 30}

        for name, size in sizes.items():
            (self.dir / name).write_bytes(os.urandom(size))
            (self.dir / name).read_bytes()

        self.handle = diskann.Handle(ptr=None, dim=0, points=0, threads=1, opened={}, folder=self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def ctx(self, mode: str) -> ServeContext:
        return ServeContext(index=self.dir, dataset=None, rows=1, k=10, metric="l2", threads=1, variant="",
                            mode=mode)

    def cached(self, name: str) -> int:
        """The cached pages of the file past its first two, which another process may read back into the cache
        after a new mapping."""
        held = evictor.HeldMap(self.dir / name)

        try:
            return held.resident(2 * evictor.PAGE)[0]
        finally:
            held.close()

    def test_mode_pq_drops_the_cached_sectors_and_keeps_the_pq_files(self):
        diskann.DiskANN().warm_up(self.handle, self.ctx("pq"))
        self.assertEqual(self.cached("index_disk.index"), 0)
        self.assertEqual(self.cached("index_pq_compressed.bin"), 14)

    def test_mode_whole_reads_the_index_file(self):
        adapter = diskann.DiskANN()
        adapter.warm_up(self.handle, self.ctx("pq"))
        adapter.warm_up(self.handle, self.ctx("whole"))
        self.assertEqual(self.cached("index_disk.index"), 62)

    def test_mode_pq_needs_the_pool_reader(self):
        for variant in ("advise", "stock"):
            adapter = diskann.DiskANN()
            adapter.variant = variant

            with self.assertRaises(Unsupported, msg=variant):
                adapter.warm_up(self.handle, self.ctx("pq"))

        self.assertGreater(self.cached("index_disk.index"), 0)


class ReportTest(unittest.TestCase):
    def test_build_fields(self):
        fields = diskann.build_fields(REPORT, 50000)
        self.assertEqual(fields["seconds"], 12.5)
        self.assertEqual(fields["peak_footprint"], 812345678)
        self.assertEqual(fields["encode_seconds"], 3.25)
        self.assertIsNone(fields["cluster_seconds"])
        self.assertEqual(fields["stored_vector_bytes"], 50000 * (768 * 4 + 96))
        self.assertTrue(fields["exact_f32_stored"])
        self.assertEqual(fields["extra"]["strategy"], "one shot")
        self.assertEqual(fields["extra"]["stages"]["InmemIndexBuild"], 8.0)

    def test_a_sharded_build_says_so(self):
        report = dict(REPORT, stages=[["PartitionData", 1.0], ["BuildIndicesOnShards", 5.0]])
        self.assertEqual(diskann.build_fields(report, 10)["extra"]["strategy"], "sharded")

    def test_the_report_is_the_last_json_line(self):
        out = b"some noise\n{\"event\": \"early\"}\n{not json\n" + json.dumps(REPORT).encode() + b"\ntrailing\n"
        self.assertEqual(diskann.last_json(out)["seconds"], 12.5)

        with self.assertRaises(RuntimeError):
            diskann.last_json(b"no report here\n")

    def test_the_report_follows_the_dots_a_sharded_build_prints(self):
        # DiskANN's merge of shards prints a dot every 499,999 nodes to stdout with no newline, so the report shares
        # their line.
        out = b"...................." + json.dumps(REPORT).encode() + b"\n"
        self.assertEqual(diskann.last_json(out)["seconds"], 12.5)
        self.assertEqual(diskann.last_json(b"..{not json {\"event\": \"build\"}\n"), {"event": "build"})

        with self.assertRaises(RuntimeError):
            diskann.last_json(b"....{not json\n")

    def test_search_stats(self):
        info = diskann.SearchInfo(hops=12, ios=48, cmps=3000, results=10, reads=48, read_calls=12,
                                  bytes_requested=48 * 4096, reader_ns=2_500_000, io_us=2400.0, pq_us=80.0,
                                  total_us=3100.0)
        stats = diskann.search_stats(info, {"open_pq_us": 900.0})
        self.assertEqual(stats["bytes_requested"], 48 * 4096)
        self.assertEqual(stats["bytes_scanned"], 48 * 4096)
        opened = diskann.search_stats(info, {}, open_bytes=5000)
        self.assertEqual(opened["bytes_requested"], 5000 + 48 * 4096)
        self.assertEqual(opened["stages"]["open_bytes_requested"], 5000)
        self.assertEqual(stats["dist_evals"], 3000)
        self.assertEqual(stats["stages"]["open_pq_us"], 900.0)
        self.assertEqual(stats["stages"]["reader_us"], 2500.0)
        self.assertEqual((stats["stages"]["hops"], stats["stages"]["sector_reads"]), (12, 48))


class MetricTest(unittest.TestCase):
    def test_unit_rows(self):
        rows = np.eye(3, dtype=np.float32)
        self.assertTrue(diskann.unit_rows(rows))
        self.assertFalse(diskann.unit_rows(rows * 1.01))

    def test_a_metric_unlike_the_datasets_needs_unit_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = helpers.make_dataset(root, n=200, nq=10, tie_query=None)
            adapter = diskann.DiskANN()
            ctx = BuildContext(config={}, data_path=ds.base_path, rows=200, out=root, clusters=None, threads=1,
                               seed=1, metric="cosine", variant="", dataset=ds)
            adapter.check_metric(ctx, "l2")
            adapter.check_metric(ctx, "cosine")
            scaled = np.asarray(ds.base) * 3
            data.write_fbin(ds.base_path, scaled)
            loud = data.Dataset(ds.name, root, root / "derived")
            ctx.dataset = loud

            with self.assertRaises(ValueError):
                adapter.check_metric(ctx, "l2")

            adapter.check_metric(ctx, "cosine")


def events(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


@unittest.skipUnless(diskann.LIB.exists() and diskann.BIN.exists(), "needs the library and the build tool")
class LiveTest(unittest.TestCase):
    """Runs the adapter's subcommands on a small index, the way the driver does."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ds = helpers.make_dataset(cls.root, n=3000, dim=16, nq=40, tie_query=None)
        cls.index = cls.root / "index"
        config = json.dumps({"degree": 16, "build_list": 50, "pq_bytes": 8})
        cls.build, _ = cls.run_adapter("build", *cls.common(), "--out", str(cls.index), "--config", config,
                                       "--threads", "2")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def common(cls, variant: str = "") -> list[str]:
        return ["--data", str(cls.ds.base_path), "--rows", "3000", "--variant", variant]

    @classmethod
    def run_adapter(cls, *args: str, stdin: str = "") -> tuple[list[dict], str]:
        out = subprocess.run([helpers.python(), "-m", "adapters.diskann", *args], input=stdin, capture_output=True,
                             text=True, cwd=helpers.PY, env=helpers.env(), timeout=600)
        return events(out.stdout), out.stderr

    def test_build(self):
        self.assertEqual([m["event"] for m in self.build], ["build"])
        fields = self.build[0]["fields"]
        self.assertGreater(fields["seconds"], 0)
        self.assertGreater(fields["peak_footprint"], 0)
        self.assertEqual(fields["stored_vector_bytes"], 3000 * (16 * 4 + 8))
        self.assertEqual(fields["extra"]["meta"]["rows"], 3000)
        self.assertTrue((self.index / "index_disk.index").exists())

    def test_every_reader_finds_the_same_ids(self):
        found = {}

        for variant in ("pool", "advise", "stock"):
            with tempfile.TemporaryDirectory() as ids:
                got, err = self.run_adapter("warm", *self.common(variant), "--index", str(self.index), "--grid",
                                            json.dumps([{"list": 50, "beam": 4}]), "--split", "all", "--ids-dir",
                                            ids, "--passes", "1", "--warmup", "5")
                settings = [m for m in got if m["event"] == "setting"]
                self.assertEqual(len(settings), 1, err)
                found[variant] = np.load(settings[0]["ids_file"])

        np.testing.assert_array_equal(found["pool"], found["advise"])
        np.testing.assert_array_equal(found["pool"], found["stock"])
        truth = np.asarray(self.ds.truth(3000).ids)[:, :10]
        hits = sum(len(set(f) & set(t)) for f, t in zip(found["pool"], truth))
        self.assertGreater(hits / truth.size, 0.9)

    def test_warm_mode_pq_reads_the_sectors_from_the_disk(self):
        found, fields = {}, {}

        for mode in ("whole", "pq"):
            with tempfile.TemporaryDirectory() as ids:
                got, err = self.run_adapter("warm", *self.common(), "--index", str(self.index), "--grid",
                                            json.dumps([{"list": 50, "beam": 4}]), "--split", "all", "--mode", mode,
                                            "--ids-dir", ids, "--passes", "1", "--warmup", "5")
                settings = [m for m in got if m["event"] == "setting"]
                self.assertEqual(len(settings), 1, err)
                found[mode], fields[mode] = np.load(settings[0]["ids_file"]), settings[0]["fields"]

        np.testing.assert_array_equal(found["whole"], found["pq"])
        # Mode whole leaves the index file cached, and mode pq leaves its sectors on the disk.
        disk = (self.index / diskann.DISK_FILE).stat().st_size
        self.assertGreaterEqual(fields["whole"]["resident_file_bytes"], disk)
        self.assertLess(fields["pq"]["resident_file_bytes"], disk // 2)

    def test_qps_through_the_batch_call(self):
        with tempfile.TemporaryDirectory() as ids:
            got, err = self.run_adapter("warm", *self.common(), "--threads", "2", "--index", str(self.index),
                                        "--grid", json.dumps([{"list": 50, "beam": 4}]), "--ids-dir", ids,
                                        "--warmup", "2", "--qps-seconds", "0.2", "--qps-repeats", "1")
            fields = [m for m in got if m["event"] == "setting"][0]["fields"]
            self.assertTrue(fields["batch_api"], err)
            self.assertGreater(fields["qps"], 0)

    def go(self, variant: str, regime: str, commands: list[dict], extra: list[str] = ()) -> list[dict]:
        args = ["cold", *self.common(variant), "--index", str(self.index), "--params",
                json.dumps({"list": 50, "beam": 4}), "--queries", "1,2,3", "--regime", regime, "--cpu-state",
                "none", *extra]
        got, err = self.run_adapter(*args, stdin="".join(json.dumps(c) + "\n" for c in commands))
        self.assertTrue(got, err)
        return got

    def test_a_c0_trial(self):
        for variant in ("pool", "advise", "stock"):
            got = self.go(variant, "c0", [{"cmd": "go", "trial": 0, "queries": [1, 2, 3]}],
                          ["--dummy", str(self.index)])
            self.assertEqual([m["event"] for m in got], ["ready", "trial"], got)
            fields = got[1]["fields"]
            self.assertEqual(len(fields["ids"]), 10)
            self.assertGreater(fields["open_us"], 0)
            self.assertEqual(len(fields["next_us"]), 2)
            self.assertIn("open_pq_us", fields["stages"])
            # The first query is charged with the open, which reads both PQ files whole and the header.
            pq = sum((self.index / name).stat().st_size for name in diskann.PQ_FILES)
            self.assertGreaterEqual(fields["stages"]["open_bytes_requested"], pq + 4096)
            self.assertGreater(fields["bytes_requested"], fields["stages"]["open_bytes_requested"])
            self.assertEqual(fields["stages"]["sector_reads"], fields["stages"]["nodes_read"])
            # The bytes it read of each file, which a reader that skips the page cache hides from mincore.
            read = {Path(p).name: n for p, n in fields["extra"]["files_read"].items()}
            self.assertEqual({k: v for k, v in read.items() if k != diskann.DISK_FILE},
                             {name: (self.index / name).stat().st_size for name in diskann.PQ_FILES})
            self.assertEqual(read[diskann.DISK_FILE] + pq, fields["bytes_requested"], variant)
            self.assertGreater(fields["resident_rerun_us"], 0)
            self.assertEqual(fields["extra"].get("rerun_prefilled", False), variant == "pool")

    def test_a_setting_with_other_cached_nodes_opens_again(self):
        with tempfile.TemporaryDirectory() as ids:
            grid = [{"list": 50, "beam": 4}, {"list": 50, "beam": 4, "cache_nodes": 200}]
            got, err = self.run_adapter("warm", *self.common(), "--index", str(self.index), "--grid",
                                        json.dumps(grid), "--split", "all", "--ids-dir", ids, "--passes", "1",
                                        "--warmup", "5")
            settings = [m for m in got if m["event"] == "setting"]
            self.assertEqual(len(settings), 2, err)
            first, second = (np.load(s["ids_file"]) for s in settings)

        # Cached nodes change no answer, and fewer sectors come from the file.
        np.testing.assert_array_equal(first, second)
        self.assertLess(settings[1]["fields"]["bytes_scanned"], settings[0]["fields"]["bytes_scanned"])

    def test_c1_keeps_one_process(self):
        commands = [{"cmd": "go", "trial": t, "queries": [q]} for t, q in enumerate([1, 2])] + [{"cmd": "stop"}]
        got = self.go("pool", "c1", commands)
        self.assertEqual([m["event"] for m in got], ["ready", "trial", "trial"])
        self.assertIsNone(got[1]["fields"]["open_us"])
        self.assertEqual(got[1]["fields"]["stages"]["open_bytes_requested"], 0)

    def test_the_stock_reader_has_no_c1(self):
        got = self.go("stock", "c1", [{"cmd": "stop"}])
        self.assertEqual(got[0]["event"], "unsupported")

    def test_an_unknown_search_setting_is_unsupported(self):
        args = ["cold", *self.common(), "--index", str(self.index), "--params",
                json.dumps({"list": 50, "beam": 4, "nprobe": 3}), "--queries", "1", "--cpu-state", "none"]
        got, _ = self.run_adapter(*args, stdin="")
        self.assertEqual(got[0]["event"], "unsupported")

    def test_files(self):
        got, _ = self.run_adapter("files", "--index", str(self.index))
        names = {Path(f["path"]).name for f in got[0]["files"]}
        self.assertIn("index_disk.index", names)


if __name__ == "__main__":
    unittest.main()
