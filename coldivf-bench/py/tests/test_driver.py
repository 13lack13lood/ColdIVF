import io
import os
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from tests import helpers
import data
import driver
import evictor
import schema

SUITE = """
phase = "test"
dataset = "tiny-16-cosine"
n = 1500
index_root = "{root}/indexes"
results_root = "{root}/results"
seed = 3
max_attempts = 2
wait_s = 0.05

[gate]
mode = "off"
invalid_vm = []

[[system]]
name = "fake"
adapter = "tests.fake_adapter"
variant = "{variant}"

  [[system.build]]
  config_id = "flat"
  threads = 2

  [[system.warm]]
  config_id = "flat"
  split = "tune"
  passes = 2
  warmup = 4
  grid = {{ unused = [1, 2] }}

  [[system.cold]]
  config_id = "flat"
  regime = "c0"
  trials = 3
  next = 2
  dummy_n = 300

  [[system.cold]]
  config_id = "flat"
  regime = "intent"
  trials = 2
  next = 0
  delay_ms = 2

  [[system.cold]]
  config_id = "flat"
  regime = "c1"
  trials = 3
  group = "live"
"""


def write_suite(root: Path, text: str = SUITE, variant: str = "") -> Path:
    path = root / "suite.toml"
    path.write_text(text.format(root=root, variant=variant))
    return path


class SuiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load(self):
        suite = driver.load_suite(write_suite(self.root))
        self.assertEqual((suite.phase, suite.n, suite.max_attempts), ("test", 1500, 2))
        system = suite.systems[0]
        self.assertEqual(system.module, "tests.fake_adapter")
        self.assertEqual([w.settings for w in suite.warm], [[{"unused": 1}, {"unused": 2}]])
        self.assertEqual(len(suite.builds), 2)
        dummy = suite.cells[0].dummy
        self.assertEqual((dummy.n, dummy.config_id), (300, "flat"))
        self.assertEqual(suite.cells[2].next, 0)
        self.assertEqual(suite.index_root, self.root / "indexes")

    def test_unknown_keys_fail(self):
        with self.assertRaises(driver.SuiteError):
            driver.load_suite(write_suite(self.root, SUITE.replace("trials = 3\n  next", "trails = 3\n  next")))

    def test_c0_and_c1_on_one_index_in_one_group_fail(self):
        with self.assertRaises(driver.SuiteError):
            driver.load_suite(write_suite(self.root, SUITE.replace('group = "live"', 'group = "main"')))

    def test_grid_order(self):
        grid = driver.expand_grid({"nprobe": [1, 2], "rerank": [5, 6]}, [{"nprobe": 9}])
        self.assertEqual(grid, [{"nprobe": 1, "rerank": 5}, {"nprobe": 1, "rerank": 6}, {"nprobe": 2, "rerank": 5},
                                {"nprobe": 2, "rerank": 6}, {"nprobe": 9}])

    def test_swap_rule(self):
        suite = driver.load_suite(write_suite(self.root, SUITE.replace("invalid_vm = []\n", "")))
        d = driver.Driver(suite)
        self.assertEqual(d.vm_moved({"swapins": 2, "compressions": 0}), "swapins 2")
        self.assertEqual(d.vm_moved({"swapins": 0, "compressions": None}), "")
        suite.gate["invalid_vm"] = []
        self.assertEqual(d.vm_moved({"swapins": 2, "compressions": 5}), "")

    def test_unknown_build(self):
        with self.assertRaises(driver.SuiteError):
            driver.load_suite(write_suite(self.root, SUITE.replace('regime = "intent"',
                                                                   'regime = "intent"\n  seed = 9')))


class GateTest(unittest.TestCase):
    def test_parse_top(self):
        text = ("Processes: 1\nCPU usage: 10.0% user, 5.0% sys, 85.00% idle\nx\n"
                "CPU usage: 1.5% user, 2.5% sys, 96.0% idle")
        self.assertEqual(driver.parse_top(text), [85.0, 96.0])

    def test_process_checks(self):
        rows = [(1, 0, "/sbin/launchd"), (10, 1, "caffeinate -di"), (20, 1, "/usr/bin/cargo build"),
                (30, 1, "python driver.py"), (31, 30, "rustc --crate-name x")]
        self.assertTrue(driver.caffeinated(rows))
        self.assertFalse(driver.caffeinated([(10, 1, "caffeinate -d")]))
        self.assertEqual(driver.ancestors(rows, 31), {31, 30})
        self.assertEqual(driver.busy(rows, ["cargo", "rustc"], {30, 31}), ["20 /usr/bin/cargo build"])

    def test_cputime(self):
        self.assertEqual(driver._cputime("0:01.50"), 1.5)
        self.assertEqual(driver._cputime("345:27.44"), 345 * 60 + 27.44)
        self.assertEqual(driver._cputime("1-02:03:04"), 86400 + 2 * 3600 + 3 * 60 + 4)
        delta = driver.agents_delta({"a": 1.0}, {"a": 1.25}, 2.0)
        self.assertEqual(delta, {"window_s": 2.0, "cpu_ms": {"a": 250.0}})

    def test_job_gate_files(self):
        with tempfile.TemporaryDirectory() as locks:
            Path(locks, "SOLO").write_text(f"{os.getpgrp()} 16.0\n")
            code = ("import fcntl, sys, time\nf = open(sys.argv[1], 'w')\nfcntl.flock(f, fcntl.LOCK_EX)\n"
                    "print('held', flush=True)\ntime.sleep(30)")
            other = subprocess.Popen([helpers.python(), "-c", code, str(Path(locks, "slot0"))],
                                     stdout=subprocess.PIPE, text=True)
            mine = open(Path(locks, "slot1"), "w")

            try:
                self.assertEqual(other.stdout.readline().strip(), "held")

                with mock.patch.dict(os.environ, {"COLDIVF_BENCH_LOCKS": locks}):
                    rows = driver.processes()
                    solo, cap, others = driver.gate_jobs(rows, driver.ancestors(rows, os.getpid()),
                                                         driver.locks_folder({}))

                self.assertTrue(solo)
                self.assertEqual(cap, 16.0)
                self.assertEqual(len(others), 1)
                self.assertTrue(others[0].startswith(f"{other.pid} "))
            finally:
                mine.close()
                other.kill()
                other.wait()
                other.stdout.close()

    def test_gate_off_still_records(self):
        cfg = driver.DEFAULT_GATE | {"mode": "off", "require_caffeinate": False, "require_ac": False,
                                     "busy_names": []}
        gate = driver.check_gate(cfg)
        self.assertIsNone(gate["cpu_idle_pct"])
        self.assertIn("power", gate)

    def test_an_enforced_gate_must_see_the_job_gate(self):
        cfg = driver.DEFAULT_GATE | {"mode": "enforce", "require_caffeinate": False, "require_ac": False,
                                     "busy_names": []}
        env = {k: v for k, v in os.environ.items() if k != "COLDIVF_BENCH_LOCKS"}

        with mock.patch.object(driver, "cpu_idle", return_value=99.0), mock.patch.dict(os.environ, env, clear=True):
            gate = driver.check_gate(cfg)
            self.assertFalse(gate["passed"])
            self.assertTrue(any("can't see the job gate" in p for p in gate["problems"]), gate["problems"])
            self.assertEqual((gate["solo"], gate["locks"], gate["locks_seen"]), (False, None, False))
            self.assertTrue(driver.check_gate(cfg | {"mode": "warn"})["passed"])
            gate = driver.check_gate(cfg | {"locks": "/nonexistent/locks"})
            self.assertTrue(any("isn't there" in p for p in gate["problems"]), gate["problems"])

            with tempfile.TemporaryDirectory() as locks:
                gate = driver.check_gate(cfg | {"locks": locks})

        self.assertTrue(gate["passed"], gate["problems"])
        self.assertEqual((gate["solo"], gate["locks_seen"]), (False, True))


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_remove_stale_clones(self):
        dead = self.root / "clones.noindex" / "99999999-0-abc"
        live = self.root / "clones.noindex" / f"{os.getpid()}-0-abc"

        for p in (dead, live):
            p.mkdir(parents=True)
            (p / "f").write_bytes(b"x")

        self.assertEqual(driver.remove_stale(self.root), [str(dead)])
        self.assertTrue(live.exists())

    def test_remove_stale(self):
        dead = self.root / "x.coldivf.99999999.0.spill"
        live = self.root / f"x.coldivf.{os.getpid()}.1.groups"
        other = self.root / "x.coldivf"

        for p in (dead, live, other):
            p.write_bytes(b"x")

        self.assertEqual(driver.remove_stale(self.root), [str(dead)])
        self.assertTrue(live.exists() and other.exists())

    def test_the_disk_sampler_sees_files_that_live_a_moment(self):
        sampler = driver.DiskSampler(self.root, 0.005)
        brief = self.root / "x.coldivf.1.0.groups"
        brief.write_bytes(b"x" * 65536)
        time.sleep(0.1)
        brief.unlink()
        peak, seen = sampler.stop()
        self.assertIn(str(brief), seen)
        self.assertGreaterEqual(peak, 65536)

    def test_touched_files_add_reads_that_skip_the_cache(self):
        res = evictor.Residency(pages={"/i/a": 2, "/i/b": 0, "/i/c": 0}, totals={"/i/a": 4, "/i/b": 4, "/i/c": 4})
        self.assertEqual(driver.touched_files(res, None, []), 1)
        self.assertEqual(driver.touched_files(res, {"/i/b": 4096, "/i/c": 0, "/elsewhere": 9}, []), 2)
        self.assertEqual(driver.touched_files(res, {"/i/a": 5, "/i/d": 1}, ["/i/d"]), 2)

    def test_moved_files(self):
        files = [{"path": "/r/idx/a", "bytes": 1, "ranges": [{"offset": 0, "length": 1, "kind": "bulk",
                                                              "loaded": True}]}]
        moved = driver.moved_files(files, Path("/r/idx"), Path("/c/9/idx"))
        self.assertEqual(moved[0]["path"], "/c/9/idx/a")
        self.assertEqual(moved[0]["ranges"], files[0]["ranges"])
        self.assertEqual(files[0]["path"], "/r/idx/a")

    def test_identity_is_kept_beside_the_index(self):
        index = self.root / "idx"
        index.mkdir()
        (index / "a").write_bytes(b"abc")
        first = driver.identity(index)
        self.assertEqual((first["bytes"], first["files"]), (3, 1))
        self.assertTrue((self.root / "idx.identity.json").exists())
        self.assertEqual(driver.identity(index)["checksum"], first["checksum"])
        (index / "a").write_bytes(b"abd")
        self.assertNotEqual(driver.identity(index)["checksum"], first["checksum"])


class EndToEndTest(unittest.TestCase):
    """Runs whole batches with the test adapter, with a build, a warm grid, and C0, intent and C1 trials."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=1500, nq=30)
        self.env = mock.patch.dict(os.environ, {"COLDIVF_DATA": str(self.root / "data")})
        self.env.start()
        self.quiet = mock.patch.object(driver.Driver, "log", lambda self, msg: None)
        self.quiet.start()

    def tearDown(self):
        self.quiet.stop()
        self.env.stop()
        self.tmp.cleanup()

    def batch(self, variant: str = "", **kwargs) -> driver.Driver:
        suite = driver.load_suite(write_suite(self.root, variant=variant))
        d = driver.Driver(suite, **kwargs)
        self.assertEqual(d.run(), 0)
        return d

    def lines(self, name: str = "fake") -> list[dict]:
        return schema.read(self.root / "results" / "test" / f"{name}.jsonl")

    def test_whole_protocol_and_resume(self):
        self.batch()
        lines = self.lines()

        for line in lines + self.lines("batch"):
            self.assertEqual(schema.validate(line), [], line.get("error"))

        kinds = [(x["kind"], x.get("regime")) for x in lines]
        self.assertEqual(kinds.count(("build", None)), 2)
        self.assertEqual(kinds.count(("warm", None)), 2)
        self.assertEqual(kinds.count(("cold", "c0")), 3)
        self.assertEqual(kinds.count(("cold", "intent")), 2)
        self.assertEqual(kinds.count(("cold", "c1")), 3)
        bad = [(x["kind"], x["status"], x["error"]) for x in lines if x["status"] != "ok"]
        self.assertEqual(bad, [])

        builds = [x for x in lines if x["kind"] == "build"]
        self.assertTrue(all(x["index_bytes"] > 0 and x["peak_disk_bytes"] > 0 for x in builds))
        self.assertTrue(all(x["index_identity"]["checksum"] for x in builds))
        # The dummy's line names its own n.
        self.assertEqual(sorted(x["n"] for x in builds), [300, 1500])

        for warm in (x for x in lines if x["kind"] == "warm"):
            self.assertEqual(warm["recall10"], 1.0)
            self.assertEqual(warm["queries"], 15)
            self.assertTrue(Path(warm["ids_file"]).exists())
            self.assertEqual(warm["gt"]["k"], 100)
            self.assertGreater(warm["overhead_us"], 0)

        cold = [x for x in lines if x["kind"] == "cold"]

        for x in cold:
            self.assertEqual(x["recall10"], 1.0)
            self.assertEqual(x["resident_pages_at_go"], 0)
            self.assertGreater(x["files_touched"], 0)
            self.assertGreater(x["total_us"], 0)
            self.assertEqual(x["query_id"] % 2, 1)
            self.assertEqual(x["extra"]["resident_loaded_bytes_after"], 0)
            self.assertFalse(x["gate"]["solo"])

        self.assertIsNone(cold[0]["predecessor"])
        self.assertTrue(all(x["predecessor"] for x in cold[1:] if x["regime"] != "c1"))
        self.assertEqual({len(x["next_us"]) for x in cold if x["regime"] == "c0"}, {2})
        self.assertEqual(len({x["run_key"] for x in lines}), len(lines))
        summary = self.lines("batch")[-1]
        self.assertEqual(summary["counts"], {"ok": len(lines)})
        self.assertGreater(summary["extra"]["open_us_first"], 0)
        self.assertGreater(summary["open_us_batch"], 0)
        self.assertEqual([v["overhead_us"] > 0 for v in summary["extra"]["overhead"].values()], [True])
        self.assertEqual({v["worker_qos"] for v in summary["extra"]["systems"].values()}, {"default"})

        again = self.batch()
        self.assertEqual(len(self.lines()), len(lines))
        self.assertEqual(again.resumed, 2 + 2 + 3 + 2 + 3)

    def test_warmup_queries_stay_out_of_every_trial(self):
        suite = driver.load_suite(write_suite(self.root))
        d = driver.Driver(suite)

        for cell in suite.cells:
            used = {q for t in range(cell.trials) for q in d.trial_queries(cell, t)}
            warm = d.warmup_queries(cell)
            self.assertEqual(len(warm), driver.WARMUP_QUERIES)
            self.assertFalse(used & set(warm), cell.label)
            # A test cell warms up on tune queries, which are the even ones.
            self.assertTrue(all(q % 2 == 0 for q in warm), cell.label)

        # A cell of every query warms up on the end of its split, past the queries its trials search.
        cell = suite.cells[0]
        cell.split, cell.trials, cell.next = "all", 25, 0
        self.assertEqual(d.warmup_queries(cell), [29, 28, 27, 26, 25])
        self.assertIn("--warmup-queries", d.cold_args(cell, [1]))

        # Nor does any trial of another cell on the index in the group search them. With 30 queries in blocks of 3, the
        # cell of 3 trials takes the queries 0 to 8, and the cell of 9 trials the first of the blocks 4 to 8 and 0 to 3.
        other = suite.cells[1]
        cell.trials, cell.next = 3, 2
        other.split, other.trials, other.next = "all", 9, 0
        self.assertEqual(sorted({q for t in range(9) for q in d.trial_queries(other, t)}),
                         [0, 3, 6, 9, 12, 15, 18, 21, 24])
        self.assertEqual(d.warmup_queries(cell), [29, 28, 27, 26, 25, 23, 22, 20])

    def test_cells_on_one_index_search_queries_of_their_own_in_each_round(self):
        suite = driver.load_suite(write_suite(self.root))
        d = driver.Driver(suite)
        c0, intent, c1 = suite.cells
        # The test split's queries are the odd ones. C0 and intent share the index in the main group, in blocks of 3
        # positions. C0 takes the blocks 0, 1 and 2, and intent, with one query a trial, starts 2 blocks later.
        self.assertEqual([d.trial_queries(c0, t) for t in range(3)], [[1, 3, 5], [7, 9, 11], [13, 15, 17]])
        self.assertEqual([d.trial_queries(intent, t) for t in range(2)], [[13], [19]])
        # C1 runs in a group of its own, so it takes the positions one after another.
        self.assertEqual([d.trial_queries(c1, t) for t in range(3)], [[1], [3], [5]])

        # Cells of as many trials search the same queries over them. In each round they search different ones, and
        # neither searches one that the other searched in the round before, also once the trials outnumber the 5
        # blocks that fit in the split.
        for trials, queries in ((4, 12), (12, 15)):
            intent.trials, intent.next, c0.trials = trials, 2, trials
            a, b = ([set(d.trial_queries(c, t)) for t in range(trials)] for c in (c0, intent))
            self.assertEqual(set().union(*a), set().union(*b))
            self.assertEqual(len(set().union(*a)), queries)

            for t in range(trials):
                self.assertFalse(a[t] & b[t], (trials, t))
                self.assertFalse(t and (a[t] & b[t - 1] or b[t] & a[t - 1]), (trials, t))

        # A cell of another split takes that split's positions from the start, as a cell alone does.
        intent.split = "tune"
        self.assertEqual(d.trial_queries(intent, 0), [0, 2, 4])

    def test_a_build_of_its_own_n_is_scored_against_its_own_truth(self):
        rows = self.ds.prefix(1000)
        ids, _ = helpers.exact(np.asarray(rows), np.asarray(self.ds.queries), 100)
        derived = self.root / "derived"
        data.write_ibin(derived / "tiny-16-cosine-1000.gt.ibin", ids.astype(np.int32))
        text = SUITE.replace('  [[system.warm]]\n  config_id = "flat"',
                             '  [[system.build]]\n  config_id = "small"\n  n = 1000\n  threads = 2\n\n'
                             '  [[system.warm]]\n  config_id = "small"\n  split = "tune"\n  passes = 1\n'
                             '  warmup = 4\n  grid = {{ unused = [1] }}\n\n'
                             '  [[system.cold]]\n  config_id = "small"\n  trials = 2\n  next = 0\n\n'
                             '  [[system.warm]]\n  config_id = "flat"', 1)

        with mock.patch.object(data, "derived_root", return_value=derived):
            suite = driver.load_suite(write_suite(self.root, text))
            self.assertEqual(driver.Driver(suite, only={"build", "warm", "cold"}).run(), 0)

        small = [x for x in self.lines() if x["config_id"] == "small"]
        self.assertEqual(sorted((x["kind"], x["n"]) for x in small),
                         [("build", 1000), ("cold", 1000), ("cold", 1000), ("warm", 1000)])

        for x in small:
            if x["kind"] != "build":
                self.assertEqual(Path(x["gt"]["path"]).name, "tiny-16-cosine-1000.gt.ibin")
                self.assertEqual(x["recall10"], 1.0, x["kind"])

    def test_cached_pages_reject_a_trial(self):
        real = evictor.Evictor.make_cold
        calls = []

        def cached_once(ev, wait=0.5, sync=True):
            res = real(ev, wait, sync)

            if not calls:
                res.pages[next(iter(res.pages))] += 1

            calls.append(1)
            return res

        with mock.patch.object(evictor.Evictor, "make_cold", cached_once):
            self.batch(only={"build", "cold"}, systems={"fake"})

        cold = [x for x in self.lines() if x["kind"] == "cold" and x["regime"] != "c1"]
        rejected = [x for x in cold if x["status"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["resident_pages_at_go"], 1)
        retry = [x for x in cold if x["run_key"] == rejected[0]["run_key"] and x["status"] == "ok"]
        self.assertEqual([x["attempt"] for x in retry], [2])
        self.assertEqual(sum(1 for x in cold if x["status"] == "ok"), 5)

    def test_loaded_files_count_apart_after_a_trial(self):
        self.batch("loaded", only={"build", "cold"})
        cold = [x for x in self.lines() if x["kind"] == "cold"]
        self.assertEqual({x["status"] for x in cold}, {"ok"})

        # The fake says it copies every file at open, so whatever its trial left cached is the second copy.
        for x in cold:
            self.assertEqual(x["resident_file_bytes_after"], 0, x["regime"])
            self.assertGreater(x["extra"]["resident_loaded_bytes_after"], 0, x["regime"])
            self.assertGreater(x["files_touched"], 0)

    def test_variants_of_one_index_list_its_files_their_own_way(self):
        text = SUITE[:SUITE.index("[[system]]")] + """
[[system]]
name = "fake"
adapter = "tests.fake_adapter"

  [[system.build]]
  config_id = "flat"
  threads = 2

  [[system.cold]]
  config_id = "flat"
  trials = 2
  next = 0

[[system]]
name = "fake"
adapter = "tests.fake_adapter"
variant = "loaded"

  [[system.build]]
  config_id = "flat"
  index = "{root}/indexes/fake/flat-n1500-s1"

  [[system.cold]]
  config_id = "flat"
  trials = 2
  next = 0
"""
        suite = driver.load_suite(write_suite(self.root, text))
        self.assertEqual(driver.Driver(suite, only={"build", "cold"}).run(), 0)
        cold = [x for x in self.lines() if x["kind"] == "cold"]
        self.assertEqual(len(cold), 4)

        # Both cells share the index's evictor, and each counts the files as its own variant lists them.
        for x in cold:
            loaded = x["extra"]["resident_loaded_bytes_after"]
            resident = x["resident_file_bytes_after"]
            self.assertEqual((loaded > 0, resident > 0), (x["variant"] == "loaded", x["variant"] == ""), x["variant"])

    def test_an_adapter_that_rejects_its_own_trials(self):
        self.batch("selfreject", only={"build", "cold"})
        c0 = [x for x in self.lines() if x["kind"] == "cold" and x["regime"] == "c0"]
        self.assertEqual([x["status"] for x in c0], ["rejected"] * 6)

    def test_crashes_are_failed_lines(self):
        self.batch("crash", only={"build", "cold"})
        cold = [x for x in self.lines() if x["kind"] == "cold"]
        self.assertTrue(all(x["status"] == "failed" for x in cold))
        self.assertIn("a crash on purpose", cold[0]["error"])
        self.assertEqual(sum(1 for x in cold if x["regime"] == "c0"), 3 * 2)

    def test_unsupported_regime_writes_one_line(self):
        self.batch("noc1", only={"build", "cold"})
        c1 = [x for x in self.lines() if x["kind"] == "cold" and x["regime"] == "c1"]
        self.assertEqual([x["status"] for x in c1], ["unsupported"])

    def test_fresh_clones(self):
        text = SUITE.replace("dummy_n = 300", "dummy_n = 300\n  fresh_clone = true")
        suite = driver.load_suite(write_suite(self.root, text))
        self.assertTrue(suite.cells[0].fresh_clone)
        self.assertEqual(driver.Driver(suite, only={"build", "cold"}).run(), 0)
        c0 = [x for x in self.lines() if x["kind"] == "cold" and x["regime"] == "c0"]
        self.assertEqual([x["status"] for x in c0], ["ok"] * 3)
        self.assertTrue(all("clones.noindex" in x["extra"]["clone"] for x in c0))
        self.assertEqual(len({x["extra"]["clone"] for x in c0}), 3)
        self.assertEqual(list((self.root / "indexes" / "clones.noindex").iterdir()), [])
        self.assertTrue(all(x["recall10"] == 1.0 and x["resident_pages_at_go"] == 0 for x in c0))

    def test_dry_run(self):
        suite = driver.load_suite(write_suite(self.root))
        out = io.StringIO()

        with redirect_stdout(out):
            driver.Driver(suite, dry_run=True).run()

        self.assertIn("cold fake:flat:c0 group=main", out.getvalue())
        self.assertIn("3 of 3 trials to do", out.getvalue())
        self.assertFalse((self.root / "results" / "test" / "fake.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
