import json
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import helpers
from adapters import coldivf
from adapters.base import BuildContext, ServeContext, Unsupported
import data

# The first query line and a later one of a real `cold --one` run, trimmed to the fields the adapter reads.
FIRST = {
    "seq": 0, "query": 5, "recall": 0.9, "recall10": 1, "ids": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
    "pages_resident_at_start": 0, "pages_checked": 3, "ctx_us": 28.6, "open_us": 1879.3, "read_us": 0,
    "prepare_us": 18.1, "gap_us": 0, "late_us": 1897.4, "search_us": 2288.6, "total_us": 4186.0,
    "from_vector_us": 4190.0, "tables_us": 150.4, "route_us": 22.8, "submit_us": 83.4, "wave_us": 348,
    "cluster_wave_us": 431.4, "scan_us": 36.5, "rows_us": 1638.2, "rerank_us": 8.3, "post_embedding_us": 2287.6,
    "cluster_bytes": 983040, "cluster_ranges": 10, "row_bytes": 589824, "row_ranges": 30, "header_bytes": 16384,
    "table_bytes": 147456, "footprint_delta": 1949744.0, "footprint_before": 9289728, "peak_footprint": 12697600,
    "p_share": 0.99, "ghz": 3.6, "open_p_share": 0.99,
    "open_ghz": 3.45, "energy_nj": 2523977, "pageins": 0, "disk_bytes": 1736704, "resident_rerun_us": 730.2,
    "rerun_disk_bytes": 0, "dummy": None, "dummy_us": None, "warmup_query": 501, "syscall_open_us": 771.8,
    "qos": "user-interactive", "index_crc": 722862947, "index_epoch": 0,
}
NEXT = {"seq": 1, "query": 6, "recall": 0.8, "recall10": 0.8, "open_us": 0, "prepare_us": 0, "search_us": 1817.0,
        "total_us": 1817.0}

# A stand-in for the cold binary. It writes its arguments to $FAKE_COLD_LOG, then prints a line for each query of a
# trial, and with --c1 serves trials from its input until it reads stop. Its comment holds the text that shows the
# adapter a cold binary of this harness.
FAKE_COLD = """#!{python}
# that the manifest or the dataset gives it
import json, os, sys
args = sys.argv[1:]
if not args:
    sys.exit(2)
with open(os.environ["FAKE_COLD_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
def value(words, name):
    return words[words.index(name) + 1]
def trial(words):
    for seq, q in enumerate(value(words, "--queries").split(",")):
        print(json.dumps({{"seq": seq, "query": int(q), "trial": int(value(words, "--trial")), "open_us": 1.0,
                          "prepare_us": 2.0, "search_us": 100.0 + seq, "total_us": 100.0 + seq,
                          "from_vector_us": 100.0 + seq, "pages_resident_at_start": 0, "pages_checked": 30,
                          "ids": [int(q)] * 10, "dummy": None}}), flush=True)
if "--c1" in args:
    print("warming", flush=True)
    print(json.dumps({{"event": "ready", "warmup_us": 5.0}}), flush=True)
    for line in sys.stdin:
        if line.strip() == "stop":
            break
        with open(os.environ["FAKE_COLD_LOG"], "a") as log:
            log.write(json.dumps(line.split()) + "\\n")
        trial(line.split())
else:
    trial(args)
"""


def write_index(path: Path, sizes: list[int]) -> None:
    """A file with a ColdIVF version 1 header whose sections have `sizes`, laid out one after another."""
    head = bytearray(4096)
    head[:8] = coldivf.MAGIC
    struct.pack_into("<I", head, 8, 1)
    struct.pack_into("<Q", head, 24, 1234)
    offset = 0

    for i, size in enumerate(sizes):
        struct.pack_into("<QQII", head, 80 + 24 * i, offset, size, i + 1, 0)
        offset += size

    path.write_bytes(bytes(head) + b"\0" * (offset - 4096))


class HeaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.file = self.dir / "index.coldivf"
        write_index(self.file, [4096, 12288, 16384, 16384, 32768, 65536, 16384])

    def tearDown(self):
        self.tmp.cleanup()

    def test_sections_and_count(self):
        s = coldivf.sections(self.file)
        self.assertEqual(s["centers"], (16384, 16384))
        self.assertEqual(s["ids"], (147456, 16384))
        self.assertEqual(coldivf.header_count(self.file), 1234)

    def test_not_an_index(self):
        bad = self.dir / "bad.coldivf"
        bad.write_bytes(b"x" * 400)

        with self.assertRaises(ValueError):
            coldivf.sections(bad)

    def test_files_mark_routing_and_bulk(self):
        files = coldivf.ColdIVF().files(self.dir)
        main = files[0]
        kinds = [r["kind"] for r in main["ranges"]]
        self.assertEqual(kinds, ["routing"] * 4 + ["bulk"] * 3)
        self.assertEqual([Path(f["path"]).name for f in files[1:]], ["index.coldivf.delta", "index.coldivf.lock"])
        self.assertEqual(files[1]["ranges"], [])

    def test_index_file(self):
        self.assertEqual(coldivf.index_file(self.dir), self.file)
        self.assertEqual(coldivf.index_file(self.file), self.file)

        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                coldivf.index_file(empty)


class FlagsTest(unittest.TestCase):
    def test_flags(self):
        self.assertEqual(coldivf.flags({"nprobe": 32, "rerank": 39}, coldivf.SEARCH_FLAGS, "x"),
                         ["--nprobe", "32", "--rerank", "39"])
        self.assertEqual(coldivf.flags({"reach": None}, coldivf.SEARCH_FLAGS, "x"), [])
        self.assertEqual(coldivf.flags({"clustering": "two-level"}, coldivf.BUILD_FLAGS, "x"),
                         ["--clustering", "two-level"])

        with self.assertRaises(Unsupported):
            coldivf.flags({"nprobes": 3}, coldivf.SEARCH_FLAGS, "search setting")


# A stand-in for build_index. It writes its arguments to $FAKE_BUILD_LOG and a results line to its --results file. Its
# comment holds the name of the option that saves the clusters, which shows the adapter a build_index that takes it.
FAKE_BUILD = """#!{python}
# export-clusters
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_BUILD_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
with open(args[args.index("--results") + 1], "a") as f:
    f.write(json.dumps({{"seconds": 3.0, "cpu_seconds": 5.0, "threads": 4, "seed": 1, "peak_footprint": 10,
                        "disk_read": 0, "disk_written": 0, "cluster_seconds": 1.0, "encode_seconds": 2.0,
                        "export_seconds": 0.5, "nlist": 10, "nparents": 2}}) + "\\n")
"""


class BuildCommandTest(unittest.TestCase):
    """The adapter's build, run against a stand-in for build_index."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=200, nq=20)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.write_binary(FAKE_BUILD)
        self.log = self.root / "log.jsonl"
        self.env = mock.patch.dict(os.environ, {"FAKE_BUILD_LOG": str(self.log)})
        self.env.start()
        self.bins = mock.patch.object(coldivf, "BIN", self.bin)
        self.bins.start()
        self.out = self.root / "index" / "tl640-n200-s1"
        self.out.mkdir(parents=True)

    def tearDown(self):
        self.bins.stop()
        self.env.stop()
        self.tmp.cleanup()

    def write_binary(self, text: str) -> None:
        binary = self.bin / "build_index"
        binary.write_text(text.format(python=sys.executable))
        binary.chmod(0o755)

    def ctx(self, config: dict, out: Path | None = None) -> BuildContext:
        return BuildContext(config=config, data_path=self.ds.base_path, rows=200, out=out or self.out, clusters=None,
                            threads=4, seed=1, metric="cosine", variant="", dataset=self.ds)

    def args(self) -> list[str]:
        return json.loads(self.log.read_text().splitlines()[-1])

    def test_export_clusters_saves_the_partition_beside_the_index_folder(self):
        fields = coldivf.ColdIVF().run_build(self.ctx({"clustering": "two-level", "export_clusters": True}))
        args, stem = self.args(), self.root / "index" / "tl640-n200-s1.clusters" / "partition"
        self.assertEqual(args[args.index("--export-clusters") + 1], str(stem))
        self.assertEqual(args[args.index("--clustering") + 1], "two-level")
        self.assertNotIn("true", args)
        self.assertTrue(stem.parent.is_dir())
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(fields["extra"]["export_clusters"], str(stem))
        self.assertEqual((fields["cluster_seconds"], fields["encode_seconds"]), (1.0, 2.0))
        self.assertEqual((fields["extra"]["export_seconds"], fields["extra"]["nlist"]), (0.5, 10))

    def test_a_dummy_of_the_same_config_saves_its_own_partition(self):
        dummy = self.root / "index" / "tl640-n200-s1-dummy50"
        dummy.mkdir()
        coldivf.ColdIVF().run_build(self.ctx({"export_clusters": True}, out=dummy))
        args = self.args()
        self.assertEqual(args[args.index("--export-clusters") + 1],
                         str(self.root / "index" / "tl640-n200-s1-dummy50.clusters" / "partition"))

    def test_a_build_without_export_clusters_saves_none(self):
        for config in ({}, {"export_clusters": False}):
            fields = coldivf.ColdIVF().run_build(self.ctx(dict(config, vectors_per_list=20)))
            args = self.args()
            self.assertNotIn("--export-clusters", args)
            self.assertEqual(args[args.index("--vectors-per-list") + 1], "20")
            self.assertIsNone(fields["extra"]["export_clusters"])

        self.assertFalse((self.root / "index" / "tl640-n200-s1.clusters").exists())

    def test_export_clusters_must_be_a_flag_the_binary_takes(self):
        with self.assertRaises(ValueError):
            coldivf.ColdIVF().run_build(self.ctx({"export_clusters": "data/x"}))

        self.write_binary(FAKE_BUILD.replace("# export-clusters", "# an older build_index"))

        with self.assertRaises(Unsupported):
            coldivf.ColdIVF().run_build(self.ctx({"export_clusters": True}))

        self.assertFalse(self.log.exists())


class TrialTest(unittest.TestCase):
    def ctx(self, regime: str = "c0") -> ServeContext:
        return ServeContext(index=Path("."), dataset=None, rows=50000, k=10, metric="cosine", threads=1, variant="",
                            regime=regime)

    def test_c0_fields(self):
        out = coldivf.ColdIVF().trial_fields(self.ctx(), [5, 6], [FIRST, NEXT])
        self.assertAlmostEqual(out["open_us"], 1879.3 + 18.1)
        self.assertEqual(out["total_us"], 4186.0)
        self.assertEqual(out["next_us"], [1817.0])
        self.assertEqual(out["ids"], FIRST["ids"])
        self.assertEqual(out["stages"]["cluster_bytes"], 983040)
        self.assertEqual(out["stages"]["table_bytes"], 147456)
        self.assertIsInstance(out["footprint_delta"], int)
        self.assertEqual((out["footprint_before"], out["lifetime_max_footprint"]), (9289728, 12697600))
        self.assertNotIn("status", out)

    def test_recall_counts_ties_as_the_binary_does(self):
        out = coldivf.ColdIVF().trial_fields(self.ctx(), [5, 6], [FIRST, NEXT])
        self.assertEqual((out["recall10"], out["recall10_ids_only"]), (1, 0.9))
        self.assertEqual(out["extra"]["next_recall10"], [0.8])
        self.assertEqual(out["extra"]["recall_source"], "binary, with ties")

    def test_intent_times_from_the_vector(self):
        ctx = self.ctx("intent")
        ctx.delay_ms = 5.0
        line = FIRST | {"gap_us": 3000.0, "late_us": 4.0, "from_vector_us": 2292.6}
        out = coldivf.ColdIVF().trial_fields(ctx, [5], [line])
        # The binary's search starts when the vector arrives, so the time from then is its whole latency.
        self.assertEqual(out["total_us"], 2292.6)
        self.assertAlmostEqual(out["open_us"], 1879.3 + 18.1)
        self.assertEqual(out["stages"]["late_us"], 4.0)
        self.assertNotIn("intent_comparable", out["extra"])

    def test_c1_times_the_search_alone(self):
        line = FIRST | {"open_us": None, "prepare_us": None, "read_us": None, "total_us": 2288.6}
        out = coldivf.ColdIVF().trial_fields(self.ctx("c1"), [5], [line])
        self.assertIn("c1", coldivf.ColdIVF().supported_regimes())
        self.assertIsNone(out["open_us"])
        self.assertEqual((out["total_us"], out["search_us"]), (2288.6, 2288.6))
        # The process's footprint before the search holds the cluster tables it keeps.
        self.assertEqual(out["footprint_before"], 9289728)
        self.assertNotIn("status", out)

    def test_the_dummy_and_the_rerun_are_reported(self):
        out = coldivf.ColdIVF().trial_fields(self.ctx(), [5, 6], [FIRST | {"dummy": "d.coldivf", "dummy_us": 9.0}])
        self.assertEqual(out["resident_rerun_us"], 730.2)
        self.assertEqual((out["extra"]["dummy_searched"], out["extra"]["dummy_us"]), (True, 9.0))
        self.assertEqual(out["extra"]["rerun_disk_bytes"], 0)
        self.assertFalse(coldivf.ColdIVF().trial_fields(self.ctx(), [5], [FIRST])["extra"]["dummy_searched"])

    def test_the_binaries_carry_no_per_call_cost(self):
        self.assertEqual(coldivf.ColdIVF().run_overhead(None)["overhead_us"], 0.0)

    def test_cached_pages_reject_the_trial(self):
        out = coldivf.ColdIVF().trial_fields(self.ctx(), [5], [FIRST | {"pages_resident_at_start": 1}])
        self.assertEqual(out["status"], "rejected")
        # C1 allows 0.5% of the bulk pages to stay cached, as the driver does.
        line = FIRST | {"pages_checked": 1000}
        c1 = coldivf.ColdIVF().trial_fields(self.ctx("c1"), [5], [line | {"pages_resident_at_start": 5}])
        self.assertNotIn("status", c1)
        c1 = coldivf.ColdIVF().trial_fields(self.ctx("c1"), [5], [line | {"pages_resident_at_start": 6}])
        self.assertEqual(c1["status"], "rejected")


class ColdCommandTest(unittest.TestCase):
    """The adapter's cold processes, run against a stand-in for the binary."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=200, nq=20)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        cold = self.bin / "cold"
        cold.write_text(FAKE_COLD.format(python=sys.executable))
        cold.chmod(0o755)
        self.log = self.root / "log.jsonl"
        self.env = mock.patch.dict(os.environ, {"FAKE_COLD_LOG": str(self.log)})
        self.env.start()
        self.bins = mock.patch.object(coldivf, "BIN", self.bin)
        self.bins.start()
        for folder in ("index", "dummy"):
            (self.root / folder).mkdir()
            write_index(self.root / folder / "i.coldivf", [4096, 12288, 16384])

    def tearDown(self):
        self.bins.stop()
        self.env.stop()
        self.tmp.cleanup()

    def ctx(self, regime: str, **more) -> ServeContext:
        ctx = ServeContext(index=self.root / "index", dataset=self.ds, rows=200, k=10, metric="cosine", threads=1,
                           variant="", regime=regime, params={"nprobe": 32, "rerank": 39}, warmup_ids=[17, 19],
                           cpu_state="idle")
        for key, value in more.items():
            setattr(ctx, key, value)
        return ctx

    def logged(self) -> list[list[str]]:
        return [json.loads(x) for x in self.log.read_text().splitlines()]

    @staticmethod
    def value(args: list[str], name: str) -> str:
        return args[args.index(name) + 1]

    def test_a_c0_trial_searches_its_own_queries_after_the_dummy(self):
        adapter, ctx = coldivf.ColdIVF(), self.ctx("c0", dummy=self.root / "dummy")
        adapter.cold_setup(ctx)
        out = adapter.cold_trial(ctx, {"trial": 3, "queries": [5, 6, 7]})
        args = self.logged()[-1]
        self.assertEqual((self.value(args, "--trial"), self.value(args, "--queries")), ("3", "5,6,7"))
        self.assertEqual((self.value(args, "--warmup-query"), self.value(args, "--gap-ms")), ("17", "0.0"))
        self.assertEqual(self.value(args, "--dummy"), str(self.root / "dummy" / "i.coldivf"))
        self.assertEqual((self.value(args, "--dataset"), self.value(args, "--variant")), (self.ds.name, "idle"))
        self.assertEqual((self.value(args, "--nprobe"), self.value(args, "--rerank")), ("32", "39"))
        self.assertNotIn("--c1", args)
        self.assertEqual((out["total_us"], out["next_us"], out["ids"]), (100.0, [101.0, 102.0], [5] * 10))

    def test_an_intent_trial_gets_its_delay(self):
        adapter, ctx = coldivf.ColdIVF(), self.ctx("intent", delay_ms=2.5)
        adapter.cold_setup(ctx)
        adapter.cold_trial(ctx, {"trial": 0, "queries": [4]})
        args = self.logged()[-1]
        self.assertEqual(self.value(args, "--gap-ms"), "2.5")
        self.assertNotIn("--dummy", args)

    def test_a_c1_cell_serves_every_trial_from_one_process(self):
        adapter, ctx = coldivf.ColdIVF(), self.ctx("c1")
        adapter.cold_setup(ctx)
        server = adapter.server
        first = adapter.cold_trial(ctx, {"trial": 0, "queries": [1, 2], "self_evict": False})
        second = adapter.cold_trial(ctx, {"trial": 1, "queries": [3]})
        self.assertEqual((first["ids"], first["next_us"], second["ids"]), ([1] * 10, [101.0], [3] * 10))
        self.assertIsNone(first["open_us"])
        adapter.cold_teardown(ctx)
        self.assertEqual(server.returncode, 0)
        self.assertNotIn(server, adapter.children)
        start, *trials = self.logged()
        self.assertIn("--c1", start)
        self.assertEqual(self.value(start, "--warmup-query"), "17")
        self.assertEqual(trials, [["--trial", "0", "--queries", "1,2"], ["--trial", "1", "--queries", "3"]])

    def test_a_cold_binary_without_lists_of_queries_is_refused(self):
        (self.bin / "cold").write_text("#!/bin/sh\nexit 2\n")
        with self.assertRaises(RuntimeError):
            coldivf.ColdIVF().cold_setup(self.ctx("c0"))


class WarmRunTest(unittest.TestCase):
    """The adapter's warm runs against a stand-in for the warm binary, which writes one result line per run. Its
    comment holds the text that shows the adapter a warm binary with passes, threads and modes."""

    FAKE_WARM = """#!{python}
# --mode t1
import json, sys
args = sys.argv[1:]
with open(args[args.index("--results") + 1], "a") as f:
    f.write(json.dumps({{"p50_us": 400.0, "p90_us": 450.0, "p99_us": 500.0, "mean_us": 410.0, "queries": 10,
                         "recall": 0.9, "recall10": 0.9, "ties": True, "passes": 3}}) + "\\n")
"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=200, nq=20)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        warm = self.bin / "warm"
        warm.write_text(self.FAKE_WARM.format(python=sys.executable))
        warm.chmod(0o755)
        self.bins = mock.patch.object(coldivf, "BIN", self.bin)
        self.bins.start()
        (self.root / "index").mkdir()
        write_index(self.root / "index" / "i.coldivf", [4096, 12288, 16384])

    def tearDown(self):
        self.bins.stop()
        self.tmp.cleanup()

    def run_grid(self, counts: list[dict]) -> list[dict]:
        """The fields of a two-setting warm run while the machine's counters read `counts`, two per setting."""
        ctx = ServeContext(index=self.root / "index", dataset=self.ds, rows=200, k=10, metric="cosine", threads=1,
                           variant="", split="test")
        grid = [{"nprobe": 8, "rerank": 20}, {"nprobe": 16, "rerank": 20}]

        with mock.patch.object(coldivf.rusage, "vm_counts", side_effect=counts):
            out = list(coldivf.ColdIVF().warm_run(ctx, grid, np.arange(1, 20, 2), self.root / "ids", 3, 0, 1.0, 1))

        return [r["fields"] for r in out]

    def test_swap_ins_and_compressions_during_each_run_are_reported(self):
        quiet, swapped = {"swapins": 10, "compressions": 50}, {"swapins": 13, "compressions": 50}
        compressed = {"swapins": 13, "compressions": 57}
        first, second = self.run_grid([quiet, swapped, swapped, compressed])
        self.assertEqual((first["swapins_delta"], first["compressions_delta"]), (3, 0))
        self.assertEqual((second["swapins_delta"], second["compressions_delta"]), (0, 7))
        self.assertEqual(first["extra"]["vm_window"], "the warm binary's whole run, its warm-up included")

    def test_a_quiet_machine_moves_no_counter(self):
        still = {"swapins": 10, "compressions": 50}
        for fields in self.run_grid([still] * 4):
            self.assertEqual((fields["swapins_delta"], fields["compressions_delta"]), (0, 0))
            self.assertEqual((fields["p50_us"], fields["recall10"]), (400.0, 0.9))


class WarmFieldsTest(unittest.TestCase):
    CTX = ServeContext(index=Path("."), dataset=None, rows=50000, k=10, metric="cosine", threads=1, variant="")

    def test_a_binary_of_one_pass(self):
        line = {"p50_us": 88.0, "p99_us": 160.0, "mean_us": 100.0, "queries": 500, "recall": 0.93,
                "share_at_0_9": 0.8, "scanned": 4700.0}
        out = coldivf.ColdIVF().warm_fields(self.CTX, line, {1: True, 10: True})
        self.assertEqual((out["recall10"], out["recall10_ids_only"], out["passes"]), (0.93, 0.93, 1))
        self.assertEqual(out["qps"], 10000.0)
        self.assertIsNone(out["p90_us"])
        self.assertIsNone(out["recall1"])

    def test_a_binary_with_passes_and_recall_levels(self):
        line = {"p50_us": 88.0, "p90_us": 120.0, "mean_us": 100.0, "queries": 500, "recall": 0.93, "recall1": 0.97,
                "recall10": 0.93, "recall100": 0.9, "recall10_p05": 0.6, "recall10_p10": 0.7, "ties": False,
                "passes": 3, "share_at_0_9": 0.8}
        out = coldivf.ColdIVF().warm_fields(self.CTX, line, {1: True, 10: True})
        self.assertEqual((out["recall1"], out["recall10_p05"], out["passes"], out["p90_us"]), (0.97, 0.6, 3, 120.0))
        self.assertEqual(out["extra"]["binary"]["recall100"], 0.9)

    def test_recall_by_ids_needs_a_split_without_ties(self):
        line = {"mean_us": 100.0, "recall": 0.93, "recall1": 0.97, "recall10_p05": 0.6, "ties": False,
                "share_at_0_9": 0.8}
        out = coldivf.ColdIVF().warm_fields(self.CTX, line, {1: False, 10: False})
        self.assertIsNone(out["recall10"])
        self.assertIsNone(out["recall1"])
        self.assertIsNone(out["recall10_p05"])
        self.assertEqual(out["recall10_ids_only"], 0.93)
        self.assertEqual(out["extra"]["recall_source"], "binary, ids only")

    def test_recall_the_binary_counts_with_ties_fills_the_fields(self):
        line = {"mean_us": 100.0, "recall": 0.93, "recall1": 0.97, "recall10": 0.94, "recall100": 0.9,
                "recall10_p05": 0.6, "recall10_p10": 0.7, "ties": True, "share_at_0_9": 0.8}
        out = coldivf.ColdIVF().warm_fields(self.CTX, line, {1: False, 10: False})
        self.assertEqual((out["recall10"], out["recall1"], out["recall10_p05"], out["recall10_p10"]),
                         (0.94, 0.97, 0.6, 0.7))
        self.assertEqual((out["recall10_ids_only"], out["recall100"]), (0.93, 0.9))
        # The binary's share counts ids alone, so it needs a split without ties.
        self.assertIsNone(out["share_ge_0_9"])
        self.assertEqual(out["extra"]["recall_source"], "binary, with ties")

    def test_offers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warm"
            path.write_bytes(b"xx --mode t1 drops the rows yy")
            self.assertTrue(coldivf.offers(path, b"--mode t1"))
            self.assertFalse(coldivf.offers(path, b"--split must"))
            self.assertFalse(coldivf.offers(Path(tmp) / "missing", b"x"))


class SplitDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root, n=600, nq=20)

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_of_the_full_base(self):
        root, name = coldivf.split_dataset(self.ds, 600, "test")
        self.assertEqual(name, "test.tiny-16-cosine")
        self.assertEqual(os.readlink(root / f"{name}.base.fbin"), str(self.ds.base_path))
        queries = np.fromfile(root / f"{name}.query.fbin", dtype=np.float32, offset=8).reshape(-1, 16)
        np.testing.assert_array_equal(queries, np.asarray(self.ds.queries)[1::2])
        truth = np.fromfile(root / f"{name}.gt.ibin", dtype=np.int32, offset=8).reshape(10, -1)
        np.testing.assert_array_equal(truth, np.asarray(self.ds.truth().ids)[1::2])
        # Minus the inner product of each truth row, as the binaries measure a cosine dataset of unit vectors.
        dists = np.asarray(data.fbin(root / f"{name}.gt.dist.fbin"))
        base, ids = np.asarray(self.ds.base, dtype=np.float64), truth.astype(np.int64)
        want = -np.einsum("qd,qkd->qk", queries.astype(np.float64), base[ids])
        np.testing.assert_allclose(dists, want, rtol=0, atol=1e-6)
        self.assertTrue(np.all(np.diff(dists, axis=1) >= -1e-6))
        # The Python truth checks each k-th distance against the base as it reads it, as the binaries do.
        split = data.Dataset(name, root, self.ds.derived)
        recall = data.Recall(split, split.truth())
        self.assertAlmostEqual(recall.kth(0, 10), float(dists[0, 9]), places=5)

    def test_near_ties_are_ties_between_rows_that_differ(self):
        qids = np.arange(20)
        # Query 3's 10th neighbor has an exact copy, which the binaries count as a tie too.
        self.assertEqual(coldivf.near_ties(data.Recall(self.ds, self.ds.truth()), qids), 0)
        base, queries = np.asarray(self.ds.base).copy(), np.asarray(self.ds.queries)
        tenth = helpers.exact(base, queries[5:6], 10)[0][0, 9]
        base[598] = base[tenth]
        base[598, 0] = np.nextafter(base[598, 0], np.float32(2))
        root = self.root / "near"
        data.write_fbin(root / "tiny-16-cosine.base.fbin", base)
        data.write_fbin(root / "tiny-16-cosine.query.fbin", queries)
        data.write_ibin(root / "tiny-16-cosine.gt.ibin", helpers.exact(base, queries, 100)[0].astype(np.int32))
        near = data.Dataset("tiny-16-cosine", root, root / "derived")
        self.assertEqual(coldivf.near_ties(data.Recall(near, near.truth()), qids), 1)

    def test_squared_distances_for_a_euclidean_dataset(self):
        ds = data.Dataset("tiny-16-euclidean", self.ds.root, self.ds.derived)
        ds._base = np.asarray(self.ds.base)
        queries, ids = np.asarray(self.ds.queries)[:2], np.array([[0, 5], [7, 1]])
        got = coldivf.truth_distances(ds, queries, ids)
        want = [[float(np.sum((ds._base[i] - queries[q]) ** 2, dtype=np.float64)) for i in row]
                for q, row in enumerate(ids)]
        np.testing.assert_allclose(got, want, rtol=1e-6)

        with self.assertRaises(ValueError):
            coldivf.truth_distances(ds, queries, np.array([[0, -1], [1, 2]]))

    def test_split_of_a_prefix_and_reuse(self):
        prefix = self.ds.derived / "tiny-16-cosine-300.gt.ibin"
        ids, _ = helpers.exact(np.asarray(self.ds.base)[:300], np.asarray(self.ds.queries), 100)
        data.write_ibin(prefix, ids.astype(np.int32))
        root, name = coldivf.split_dataset(self.ds, 300, "tune")
        target = self.ds.derived / f"{name}-300.gt.ibin"
        dists = data.dist_path(target)
        self.assertTrue(target.exists() and dists.exists())
        stamps = target.stat().st_mtime_ns, dists.stat().st_mtime_ns
        time.sleep(0.01)
        coldivf.split_dataset(self.ds, 300, "tune")
        self.assertEqual((target.stat().st_mtime_ns, dists.stat().st_mtime_ns), stamps)
        # New truth for the prefix makes both files again.
        time.sleep(0.01)
        data.write_ibin(prefix, ids.astype(np.int32))
        coldivf.split_dataset(self.ds, 300, "tune")
        self.assertGreater(dists.stat().st_mtime_ns, stamps[1])

    def test_all_is_the_dataset_itself(self):
        self.assertEqual(coldivf.split_dataset(self.ds, 600, "all"), (self.ds.root, self.ds.name))


if __name__ == "__main__":
    unittest.main()
