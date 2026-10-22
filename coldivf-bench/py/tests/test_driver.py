import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from tests import helpers
from tests.test_writes import first_block
from adapters.inproc import LiveRows
import data
import driver
import evictor
import schema
import writes

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


# A write suite on 5,000 rows: the joint and inserts-only ladders, a policy pass, latency runs at S0 and S10, two
# maintenance actions whose files must match, a fresh build of S10's live rows, and warm and cold entries on states.
WRITE_SUITE = """
phase = "writes"
dataset = "tiny-16-cosine"
n = 5000
index_root = "{root}/indexes"
results_root = "{root}/results"
seed = 3
max_attempts = 2
wait_s = 0.05

[gate]
mode = "off"
invalid_vm = []

[writes]
stream = "{root}/writes/tiny-w.stream"
batch = 100
latency_states = ["s0", "s10"]
latency_calls = 20
warmup_calls = 5
durable_calls = 10
saves = 2
seed = 1
probes = 30

[writes.states]
s1 = {{ inserts = 50, deletes = 50 }}
s5 = {{ inserts = 250, deletes = 250 }}
s10 = {{ inserts = 500, deletes = 500 }}
s1i = {{ inserts = 50, deletes = 0 }}
s5i = {{ inserts = 250, deletes = 0 }}
s10i = {{ inserts = 500, deletes = 0 }}
s10p = {{ inserts = 500, deletes = 500, policy = "compact" }}

[[system]]
name = "fake"
adapter = "tests.fake_adapter"
variant = "{variant}"

  [[system.build]]
  config_id = "flat"
  threads = 2

  [[system.build]]
  config_id = "fresh"
  data = "{root}/writes/tiny-w-s10-live.base.fbin"
  threads = 2

  [[system.write]]
  config_id = "flat"
  role = "{role}"
  {states}
{maintain}
  [[system.warm]]
  config_id = "flat"
  states = "all"
  split = "test"
  passes = 1
  warmup = 4
  settings = [{{ unused = 1 }}]
  findability = true

  [[system.warm]]
  config_id = "fresh"
  split = "test"
  passes = 1
  warmup = 4
  settings = [{{ unused = 1 }}]

  [[system.cold]]
  config_id = "flat"
  states = {cold}
  trials = 2
  next = 0
"""

MAINTAIN = """
  [[system.maintain]]
  config_id = "flat"
  actions = ["compact", "recompact"]
  threads = 2
  action_threads = { recompact = 1 }
  params = { pending = 150 }
  unsearched = ["recompact"]
  same_file = { recompact = "compact" }
"""

FULL_STATES = 'states = ["s1", "s5", "s10", "s1i", "s5i", "s10i", "s10p"]'

# The drift pass: the same rounds on a drift stream, whose inserts are the rows nearest one topic, kept only at D10,
# with a maintenance action and a fresh build of D10's live rows.
DRIFT_SUITE = """
phase = "drift"
dataset = "tiny-16-cosine"
n = 5000
index_root = "{root}/indexes"
results_root = "{root}/results"
max_attempts = 2
wait_s = 0.05

[gate]
mode = "off"
invalid_vm = []

[writes]
stream = "{root}/writes/tiny-d.stream"
pattern = "drift"
topics = 4
hot_topics = 1
topic_sample = 400
kmeans_rounds = 3
batch = 100
probes = 30
latency_calls = 0
warmup_calls = 0
durable_calls = 0

[writes.states]
d1 = { inserts = 50, deletes = 50 }
d5 = { inserts = 250, deletes = 250 }
d10 = { inserts = 500, deletes = 500 }

[[system]]
name = "fake"
adapter = "tests.fake_adapter"

  [[system.build]]
  config_id = "flat"
  threads = 2

  [[system.build]]
  config_id = "fresh"
  data = "{root}/writes/tiny-d-d10-live.base.fbin"
  threads = 2

  [[system.write]]
  config_id = "flat"
  states = ["d10"]

  [[system.maintain]]
  config_id = "flat"
  actions = ["compact"]
  threads = 2

  [[system.warm]]
  config_id = "flat"
  states = ["d10", "d10-compact"]
  split = "test"
  passes = 1
  warmup = 4
  settings = [{ unused = 1 }]
  findability = true

  [[system.warm]]
  config_id = "fresh"
  split = "test"
  passes = 1
  warmup = 4
  settings = [{ unused = 1 }]

  [[system.cold]]
  config_id = "flat"
  states = ["d10"]
  trials = 2
  next = 0
"""


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

    def test_a_warm_setting_timed_while_the_machine_swapped_is_invalid(self):
        text = SUITE.replace("invalid_vm = []\n", "")
        suite = driver.load_suite(write_suite(self.root, text, variant="swapping"))
        self.assertEqual(driver.Driver(suite, only={"build", "warm"}).run(), 0)
        warm = [x for x in self.lines() if x["kind"] == "warm"]
        self.assertEqual([x["status"] for x in warm], ["invalid"] * 2)
        self.assertTrue(all(x["error"] == "swapins 2 during the timed passes" for x in warm), warm[0]["error"])
        self.assertTrue(all(x["swapins_delta"] == 2 and x["recall10"] is not None for x in warm))

    def test_lines_name_the_index_as_its_build_left_it_while_a_session_changes_it(self):
        # The "session" variant adds a log to its index while its warm run has it open, as AsterVec's store does. A
        # batch that reads the index's files only once a session is open sees a transient state, or a file that a
        # restore removes as it reads, so every line must name the index as it was before any process opened it.
        self.batch("session", only={"build"})
        self.batch("session", only={"warm"})
        lines = self.lines()
        built = next(x for x in lines if x["kind"] == "build" and x["n"] == 1500)
        warm = [x for x in lines if x["kind"] == "warm"]
        self.assertEqual([x["status"] for x in warm], ["ok"] * 2)
        self.assertEqual({x["index_identity"]["files"] for x in warm}, {built["index_identity"]["files"]})
        self.assertTrue(all(x["index_identity"] == built["index_identity"] for x in warm))
        self.assertFalse(any(p.name == "session.log" for p in (self.root / "indexes").rglob("*")))

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

    def test_lines_record_the_malloc_cache_setting(self):
        # A build whose index folder is gone builds again and writes a line of its own, as a re-measure of its peak
        # does. Each line records whether macOS malloc's cache of freed large blocks was off, since that changes the
        # footprints it holds.
        with mock.patch.dict(os.environ):
            os.environ.pop("MallocLargeCache", None)
            self.batch(only={"build"})

        shutil.rmtree(self.root / "indexes")

        with mock.patch.dict(os.environ, {"MallocLargeCache": "0"}):
            self.batch(only={"build"})

        builds = [x for x in self.lines() if x["kind"] == "build" and x["n"] == 1500]
        self.assertEqual([x["status"] for x in builds], ["ok", "ok"])
        self.assertEqual(builds[0]["run_key"], builds[1]["run_key"])
        self.assertNotIn("MallocLargeCache", builds[0]["provenance"]["env"])
        self.assertEqual(builds[1]["provenance"]["env"]["MallocLargeCache"], "0")

    def test_systems_select_a_variant_by_name_and_variant(self):
        suite = driver.load_suite(write_suite(self.root, variant="fast"))
        system = suite.systems[0]
        self.assertTrue(driver.Driver(suite, systems={"fake"}).selected(system))
        self.assertTrue(driver.Driver(suite, systems={"fake:fast"}).selected(system))
        self.assertFalse(driver.Driver(suite, systems={"fake:slow"}).selected(system))
        self.assertFalse(driver.Driver(suite, systems={"fake:"}).selected(system))
        plain = driver.load_suite(write_suite(self.root))
        self.assertTrue(driver.Driver(plain, systems={"fake:"}).selected(plain.systems[0]))

    def test_groups_run_only_their_own_cells(self):
        self.batch(only={"build", "cold"}, groups={"live"})
        cold = [x for x in self.lines() if x["kind"] == "cold"]
        self.assertEqual({x["regime"] for x in cold}, {"c1"})
        self.assertEqual(sum(1 for x in cold if x["status"] == "ok"), 3)
        suite = driver.load_suite(write_suite(self.root))
        out = io.StringIO()

        with redirect_stdout(out):
            driver.Driver(suite, dry_run=True, groups={"main"}, systems={"other"}).run()

        self.assertNotIn("cold ", out.getvalue())
        self.assertNotIn("build ", out.getvalue())
        out = io.StringIO()

        with redirect_stdout(out):
            driver.Driver(suite, dry_run=True, groups={"main"}).run()

        self.assertIn("group=main", out.getvalue())
        self.assertNotIn("group=live", out.getvalue())


class KeysTest(unittest.TestCase):
    """The run keys of a head-to-head suite stay as they were, so a rerun of its suites skips what is done."""

    TEXT = SUITE[:SUITE.index("[[system]]")].replace("{root}", "/nonexistent/keys") + """
[[system]]
name = "fake"
adapter = "tests.fake_adapter"
variant = "v"

  [[system.build]]
  config_id = "flat"
  config = { a = 1 }
  threads = 2

  [[system.build]]
  config_id = "small"
  n = 1000

  [[system.warm]]
  config_id = "flat"
  split = "tune"
  grid = { unused = [1, 2] }

  [[system.cold]]
  config_id = "flat"
  regime = "c0"
  params = { unused = 1 }
  trials = 2
  dummy_n = 300

  [[system.cold]]
  config_id = "small"
  regime = "c1"
  trials = 1
  group = "live"
"""

    def test_head_to_head_keys_dont_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.toml"
            path.write_text(self.TEXT.replace('phase = "test"', 'phase = "keys"'))
            suite = driver.load_suite(path)

        d = driver.Driver.__new__(driver.Driver)
        d.suite = suite
        self.assertEqual([d.build_key(b) for b in suite.builds],
                         ["00e55c46514f90ec39f6", "d4788145d20de4fbe8fc", "93ccd649d730411f9a77"])
        self.assertEqual([d.warm_key(w, x) for w in suite.warm for x in w.settings],
                         ["6b6cca30a25dda1ab1e1", "926927d4a5f366cc1fe2"])
        self.assertEqual([d.cold_key(c, t) for c in suite.cells for t in range(c.trials)],
                         ["4ab9328b7bb6c8298071", "310cc79ace5b4b29d166", "8b4b076b744ac22cbc13"])


class WriteSuiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, **over) -> driver.Suite:
        fill = {"variant": "", "role": "full", "states": FULL_STATES, "maintain": MAINTAIN,
                "cold": '["s0", "s10", "s10-compact"]'} | over
        path = self.root / "suite.toml"
        path.write_text(WRITE_SUITE.replace("{root}", str(self.root)).format(**fill))
        return driver.load_suite(path)

    def test_load(self):
        suite = self.load()
        system = suite.systems[0]
        self.assertEqual(suite.writes.rounds, [(50, 50), (200, 200), (250, 250)])
        self.assertEqual([(j.role, j.states) for j in system.writes],
                         [("full", ["s1", "s5", "s10", "s1i", "s5i", "s10i", "s10p"])])
        job = system.maintains[0]
        self.assertEqual((job.state, job.actions, job.threads_of("compact"), job.threads_of("recompact")),
                         ("s10", ["compact", "recompact"], 2, 1))
        # "all" leaves out the unsearched action's state, and findability runs at the ladders' ends and after
        # maintenance.
        warm = [w for w in suite.warm if w.build.state]
        self.assertEqual([w.build.state for w in warm],
                         ["s0", "s1", "s5", "s10", "s1i", "s5i", "s10i", "s10p", "s10-compact"])
        self.assertEqual([w.build.state for w in warm if w.findability], ["s10", "s10i", "s10-compact"])
        s5 = next(w.build for w in warm if w.build.state == "s5")
        self.assertEqual((s5.n, s5.index, s5.origin.config_id), (5000, self.root / "indexes" / "fake" / "flat-s5",
                                                                  "flat"))
        self.assertEqual(suite.builds[1].data, self.root / "writes" / "tiny-w-s10-live.base.fbin")
        self.assertEqual([c.build.state for c in suite.cells], ["s0", "s10", "s10-compact"])
        self.assertEqual(driver.default_states(suite.writes, "inserts"), ["s1i", "s5i", "s10i"])
        self.assertEqual(driver.default_states(suite.writes, "full"), ["s1", "s5", "s10", "s10i"])

    def test_bad_write_entries_fail(self):
        for over in ({"role": "rebuild"}, {"role": "inserts"}, {"role": "exact"}, {"cold": '["s7"]'},
                     {"states": 'states = ["s0"]'}, {"maintain": MAINTAIN.replace('unsearched = ["recompact"]',
                                                                                 'unsearched = ["major"]')}):
            with self.assertRaises(driver.SuiteError, msg=str(over)):
                self.load(**over)

        # States need a [writes] table.
        text = SUITE.replace('  [[system.cold]]\n  config_id = "flat"\n  regime = "c0"',
                             '  [[system.cold]]\n  config_id = "flat"\n  states = ["s0"]\n  regime = "c0"')

        with self.assertRaises(driver.SuiteError):
            driver.load_suite(write_suite(self.root, text))


class WriteBatchTest(unittest.TestCase):
    """Runs whole write batches with the test adapter on 5,000 rows: every pass, latency run and maintenance action,
    then warm and cold searches of every state, all scored against each state's exact truth."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=5600, nq=30)
        self.env = mock.patch.dict(os.environ, {"COLDIVF_DATA": str(self.root / "data")})
        self.env.start()
        self.quiet = mock.patch.object(driver.Driver, "log", lambda self, msg: None)
        self.quiet.start()

    def tearDown(self):
        self.quiet.stop()
        self.env.stop()
        self.tmp.cleanup()

    def suite(self, **over) -> driver.Suite:
        fill = {"variant": "", "role": "full", "states": FULL_STATES, "maintain": MAINTAIN,
                "cold": '["s0", "s10", "s10-compact"]'} | over
        path = self.root / "suite.toml"
        path.write_text(WRITE_SUITE.replace("{root}", str(self.root)).format(**fill))
        suite = driver.load_suite(path)
        cfg = suite.writes

        if not cfg.stream.exists():
            writes.ensure_stream(cfg, self.ds)
            writes.compute_truths(cfg, self.ds, "numpy", log=lambda m: None)
            writes.write_copy(cfg, self.ds, "s10")

        return suite

    def batch(self, only: set | None = None, **over) -> driver.Driver:
        d = driver.Driver(self.suite(**over), only=only)
        self.assertEqual(d.run(), 0)
        return d

    def lines(self, name: str = "fake") -> list[dict]:
        return schema.read(self.root / "results" / "writes" / f"{name}.jsonl")

    def folder(self, name: str, variant: str = "") -> Path:
        return self.root / "indexes" / ("fake" + (f"-{variant}" if variant else "")) / f"flat-{name}"

    def check_state(self, name: str, variant: str = "") -> None:
        """A state's files hold exactly the vectors the stream leaves it, the rows in order and the live ids."""
        cfg = self.suite(variant=variant).writes
        mask = writes.state_mask(cfg, writes.read_stream(cfg.stream), name)
        inserts, live = LiveRows.view(self.folder(name, variant))
        self.assertTrue(np.array_equal(live, mask[: len(live)]), name)
        self.assertEqual(len(live), cfg.n + cfg.spec(name).inserts, name)
        self.assertTrue(np.array_equal(np.asarray(inserts), np.asarray(self.ds.base)[cfg.n : len(live)]), name)

    def test_a_whole_write_batch_and_its_resume(self):
        d = self.batch()
        lines = self.lines()

        for line in lines + self.lines("batch"):
            self.assertEqual(schema.validate(line), [], line.get("error"))

        bad = [(x["kind"], x.get("op"), x["status"], x["error"]) for x in lines if x["status"] != "ok"]
        self.assertEqual(bad, [])
        self.assertEqual(d.only, driver.READ_PHASES | driver.WRITE_PHASES)

        for name in ("s0", "s1", "s5", "s10", "s1i", "s5i", "s10i", "s10p", "s10-compact", "s10-recompact"):
            record = writes.read_json(writes.record_path(self.folder(name)))
            self.assertEqual(record["state"], name)
            self.assertTrue(self.folder(name).is_dir(), name)

        for name in ("s1", "s5", "s10", "s1i", "s5i", "s10i"):
            self.check_state(name)

        # A system that writes nothing in place keeps clones, which share the rows no write changed.
        records = {name: writes.read_json(writes.record_path(self.folder(name))) for name in ("s0", "s1", "s10")}
        self.assertEqual({r["unshared"] for r in records.values()}, {False})

        if first_block(self.folder("s0") / "vectors.npy") is not None:
            self.assertEqual(first_block(self.folder("s1") / "vectors.npy"),
                             first_block(self.folder("s0") / "vectors.npy"))
            self.assertGreaterEqual(records["s10"]["extents"]["total"], records["s10"]["extents"]["files"] - 1)

        self.assertEqual(sorted(p.name for p in self.folder("s0").parent.iterdir() if "work" in p.name), [])
        stream = [x for x in lines if x["kind"] == "write" and x["mode"] == "stream"]
        ops = {(x["state_from"], x["state_to"]): sorted(y["op"] for y in stream if
                                                        (y["state_from"], y["state_to"]) == (x["state_from"],
                                                                                             x["state_to"]))
               for x in stream}
        joint, inserts = ["delete", "durable_point", "insert", "sync"], ["durable_point", "insert", "sync"]
        self.assertEqual(ops, {("s0", "s1"): joint, ("s1", "s5"): joint, ("s5", "s10"): joint,
                               ("s0", "s1i"): inserts, ("s1i", "s5i"): inserts, ("s5i", "s10i"): inserts,
                               ("s0", "s10p"): joint})
        insert = next(x for x in stream if x["op"] == "insert" and x["state_to"] == "s5")
        self.assertEqual((insert["calls"], insert["rows"], insert["batch"], insert["durability"]),
                         (2, 200, 100, "process"))
        self.assertEqual(insert["stream"]["sha256"], schema.file_sha256(d.suite.writes.stream))
        self.assertEqual((insert["stream"]["role"], insert["state"], insert["n"]), ("full", "s5", 5000))
        self.assertGreater(insert["open_us"], 0)
        self.assertEqual(insert["index_identity"], writes.read_json(writes.record_path(self.folder("s1")))["identity"])

        latency = [x for x in lines if x["kind"] == "write" and x["mode"] == "latency"]
        self.assertEqual(sorted((x["state_from"], x["op"], x["calls"]) for x in latency),
                         [(s, op, c) for s in ("s0", "s10") for op, c in (("delete", 20), ("insert", 20),
                                                                           ("insert_sync", 10), ("sync", 10))])
        self.assertTrue(all(x["state_from"] == x["state_to"] for x in latency))
        self.assertEqual(list((self.root / "indexes" / "clones.noindex").iterdir()), [])

        maintain = [x for x in lines if x["kind"] == "maintain"]
        runs = [x for x in maintain if x["state_to"] == "s10p"]
        self.assertGreater(len(runs), 0)
        self.assertTrue(all(x["state_from"] == "s10p" and x["action"] == "compact" for x in runs))
        actions = {x["action"]: x for x in maintain if x["state_from"] == "s10"}
        self.assertEqual(sorted(actions), ["compact", "recompact"])
        self.assertEqual((actions["compact"]["threads"], actions["recompact"]["threads"]), (2, 1))
        self.assertEqual(actions["compact"]["state_to"], "s10-compact")
        self.assertGreater(actions["compact"]["peak_disk_bytes"], 0)
        self.assertEqual(actions["recompact"]["extra"]["same_file"]["match"], True)

        warm = [x for x in lines if x["kind"] == "warm"]
        self.assertEqual(sorted(x["state"] for x in warm),
                         sorted(["s0", "s1", "s5", "s10", "s1i", "s5i", "s10i", "s10p", "s10-compact", "fresh"]))

        for x in warm:
            self.assertEqual((x["recall10"], x["dead_returned"]), (1.0, 0.0), x["state"])
            self.assertEqual(x["recall10_inserted"], None if x["state"] == "s0" else 1.0, x["state"])
            self.assertEqual(Path(x["gt"]["path"]).name,
                             f"tiny-w-{d.suite.writes.truth_state(x['state'] if x['state'] != 'fresh' else 's10')}"
                             f".gt.ibin", x["state"])

        found = {x["state"]: x["extra"].get("findability") for x in warm}
        self.assertEqual({k for k, v in found.items() if v}, {"s10", "s10i", "s10-compact"})
        self.assertTrue(all(v["inserted"] == 1.0 and v["base"] == 1.0 for v in found.values() if v))

        cold = [x for x in lines if x["kind"] == "cold"]
        self.assertEqual(sorted(x["state"] for x in cold), ["s0", "s0", "s10", "s10", "s10-compact", "s10-compact"])
        self.assertTrue(all(x["recall10"] == 1.0 and x["dead_returned"] == 0.0 for x in cold))
        self.assertEqual(len({x["run_key"] for x in lines}), len(lines))

        again = self.batch()
        self.assertEqual(len(self.lines()), len(lines))
        self.assertGreater(again.resumed, 0)

    def test_a_system_that_writes_in_place_takes_plain_copies(self):
        self.batch({"build", "stream", "latency", "maintain", "warm"}, variant="inplace")
        lines = self.lines()
        self.assertEqual([(x["kind"], x.get("op"), x["error"]) for x in lines if x["status"] != "ok"], [])
        self.assertEqual({tuple(schema.validate(x)) for x in lines}, {()})

        for name in ("s1", "s5", "s10", "s1i", "s5i", "s10i"):
            self.check_state(name, "inplace")

        # S0 is a clone of the build, which nothing writes. Every state the writes made has blocks of its own: each
        # round's work folder was a plain copy of the state before it and became the state by a rename.
        names = ("s0", "s1", "s5", "s10", "s1i", "s10i", "s10p", "s10-compact")
        records = {name: writes.read_json(writes.record_path(self.folder(name, "inplace"))) for name in names}
        self.assertEqual([name for name, r in records.items() if r["unshared"]], list(names[1:]))
        blocks = {name: first_block(self.folder(name, "inplace") / "vectors.npy") for name in names}

        if blocks["s0"] is not None:
            self.assertEqual(len(set(blocks.values())), len(names))

        work = [p.name for p in self.folder("s0", "inplace").parent.iterdir() if "work" in p.name]
        self.assertEqual(work, [])
        self.assertEqual(list((self.root / "indexes" / "clones.noindex").iterdir()), [])
        warm = [x for x in lines if x["kind"] == "warm"]
        self.assertTrue(warm and all(x["recall10"] == 1.0 and x["dead_returned"] == 0.0 for x in warm))
        info = [x for x in self.lines("batch")][-1]["extra"]["systems"]
        self.assertEqual({v["writes_in_place"] for v in info.values()}, {True})

    def test_a_killed_round_starts_again_from_a_clone_of_the_last_state(self):
        # The fake dies like a killed process at the insert of row 5100, in round 2 of the joint pass.
        marker = self.root / "indexes" / "fake-writecrash" / "crash.once"
        marker.parent.mkdir(parents=True)
        marker.write_text("5100")
        self.batch({"build", "stream"}, variant="writecrash")
        lines = self.lines()
        failed = [(x["op"], x["state_from"], x["state_to"], x["status"]) for x in lines if x["status"] != "ok"]
        self.assertEqual(failed, [("session", "s1", "s5", "failed")])
        self.assertFalse(marker.exists())

        for name in ("s1", "s5", "s10", "s1i", "s5i", "s10i"):
            self.check_state(name, "writecrash")

        record = writes.read_json(writes.record_path(self.folder("s5", "writecrash")))
        self.assertEqual(record["from_key"],
                         writes.read_json(writes.record_path(self.folder("s1", "writecrash")))["index_key"])
        # The round that worked wrote its lines once, as their first attempts.
        ok = [x for x in lines if x["kind"] == "write" and x["status"] == "ok" and x["state_to"] == "s5"]
        self.assertEqual(sorted(x["op"] for x in ok), ["delete", "durable_point", "insert", "sync"])
        self.assertEqual({x["attempt"] for x in ok}, {1})

    def test_a_state_cut_short_is_made_again_with_the_states_after_it(self):
        self.batch({"build", "stream"})
        before = self.lines()
        s10 = writes.read_json(writes.record_path(self.folder("s10")))
        # A crash between the state's folder and its record leaves the folder alone.
        writes.record_path(self.folder("s5")).unlink()
        self.batch({"build", "stream"})
        new = self.lines()[len(before):]
        self.assertEqual(sorted({(x["state_from"], x["state_to"]) for x in new}), [("s1", "s5"), ("s5", "s10")])
        self.assertEqual({x["attempt"] for x in new if x["state_to"] == "s5"}, {2})
        self.assertEqual({x["attempt"] for x in new if x["state_to"] == "s10"}, {1})
        again = writes.read_json(writes.record_path(self.folder("s10")))
        self.assertNotEqual(again["from_key"], s10["from_key"])
        self.assertEqual(again["from_key"], writes.read_json(writes.record_path(self.folder("s5")))["index_key"])
        self.check_state("s10")

    def test_an_inserts_only_system_takes_the_inserts_ladder(self):
        self.batch({"build", "stream", "latency", "warm"}, variant="inserts", role="inserts", states="", maintain="",
                   cold='["s0"]')
        lines = self.lines()
        bad = [(x["kind"], x.get("op"), x["status"]) for x in lines if x["status"] != "ok"]
        self.assertEqual(bad, [("write", "delete", "unsupported")] * 2)
        stream = {(x["state_from"], x["state_to"]) for x in lines if x["kind"] == "write" and x["mode"] == "stream"}
        self.assertEqual(stream, {("s0", "s1i"), ("s1i", "s5i"), ("s5i", "s10i")})
        latency = {x["state_from"] for x in lines if x["kind"] == "write" and x["mode"] == "latency"}
        self.assertEqual(latency, {"s0", "s10i"})
        warm = {x["state"]: x for x in lines if x["kind"] == "warm"}
        self.assertEqual(sorted(warm), ["fresh", "s0", "s10i", "s1i", "s5i"])
        self.assertTrue(all(x["recall10"] == 1.0 for x in warm.values()))

    def test_a_system_whose_sync_rewrites_the_index_saves_once_a_round(self):
        self.batch({"build", "stream", "latency"}, variant="whole", maintain="", cold='["s0"]')
        lines = [x for x in self.lines() if x["kind"] == "write"]
        self.assertEqual({x["status"] for x in lines}, {"ok"})
        self.assertNotIn("sync", {x["op"] for x in lines if x["mode"] == "stream"})
        saves = [x for x in lines if x["mode"] == "latency" and x["op"] == "durable_point"]
        self.assertEqual([x["calls"] for x in saves], [2, 2])
        self.assertNotIn("insert_sync", {x["op"] for x in lines})

    def test_inserts_numbered_off_their_rows_fail_the_session(self):
        self.batch({"build", "stream"}, variant="misnumber", maintain="", cold='["s0"]')
        failed = [x for x in self.lines() if x["status"] != "ok"]
        self.assertEqual({(x["op"], x["state_to"]) for x in failed}, {("session", "s1"), ("session", "s1i"),
                                                                     ("session", "s10p")})
        self.assertTrue(all("not by their rows" in x["error"] for x in failed))
        self.assertEqual(len(failed), 2 * 3)
        self.assertFalse(self.folder("s1", "misnumber").exists())

    def test_a_drift_stream_numbers_its_inserts_apart_from_their_rows(self):
        ds = helpers.make_dataset(self.root / "data", n=7000, nq=30)
        path = self.root / "drift.toml"
        path.write_text(DRIFT_SUITE.replace("{root}", str(self.root)))
        suite = driver.load_suite(path)
        cfg = suite.writes
        stream = writes.ensure_stream(cfg, ds)
        writes.compute_truths(cfg, ds, "numpy", log=lambda m: None)
        writes.write_copy(cfg, ds, "d10")
        self.assertFalse(stream.identity)
        self.assertEqual(driver.Driver(suite).run(), 0)
        lines = schema.read(self.root / "results" / "drift" / "fake.jsonl")
        self.assertEqual([(x["kind"], x.get("op"), x["error"]) for x in lines if x["status"] != "ok"], [])
        self.assertEqual({(x["state_from"], x["state_to"]) for x in lines if x["kind"] == "write"}, {("s0", "d10")})
        # The fake numbers its inserts itself, which the session checks against the stream's ids, and stores the
        # vectors of the stream's rows under them.
        inserts, live = LiveRows.view(self.root / "indexes" / "fake" / "flat-d10")
        self.assertTrue(np.array_equal(np.asarray(inserts), np.asarray(ds.base)[stream.row_of[5000:]]))
        self.assertTrue(np.array_equal(live, writes.state_mask(cfg, stream, "d10")[stream.row_of]))
        warm = {x["state"]: x for x in lines if x["kind"] == "warm"}
        self.assertEqual(sorted(warm), ["d10", "d10-compact", "fresh"])

        for x in warm.values():
            self.assertEqual((x["recall10"], x["dead_returned"]), (1.0, 0.0), x["state"])

        self.assertEqual({k: (x["extra"]["findability"]["inserted"], x["extra"]["findability"]["base"])
                          for k, x in warm.items() if "findability" in x["extra"]},
                         {"d10": (1.0, 1.0), "d10-compact": (1.0, 1.0)})
        cold = [x for x in lines if x["kind"] == "cold"]
        self.assertTrue(cold and all(x["recall10"] == 1.0 and x["state"] == "d10" for x in cold))

    def test_dry_run_lists_the_write_jobs(self):
        self.batch({"build", "stream"})
        out = io.StringIO()

        with redirect_stdout(out):
            driver.Driver(self.suite(), dry_run=True).run()

        text = out.getvalue()
        self.assertIn("stream fake flat joint: s1 done, s5 done, s10 done", text)
        self.assertIn("latency fake flat s10: to do", text)
        self.assertIn("maintain fake flat compact on s10: to do", text)
        self.assertIn("warm fake flat state=s10-compact", text)
        self.assertIn("state=s10 group=main", text)



class ChildTest(unittest.TestCase):
    def test_a_child_whose_pipe_is_past_descriptor_1024_still_sends_its_events(self):
        # A cold trial holds every file of the index open, thousands of them for LanceDB after its writes, so the
        # child's pipe gets a descriptor past select()'s limit.
        held = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]
        self.addCleanup(lambda: [os.close(fd) for fd in held])
        log = Path(tempfile.mkdtemp()) / "child.log"
        self.addCleanup(shutil.rmtree, log.parent)
        code = "import json, time; time.sleep(0.1); print(json.dumps({'event': 'ready'}), flush=True)"
        child = driver.Child([sys.executable, "-c", code], dict(os.environ), log)
        self.addCleanup(child.proc.wait)
        self.assertGreater(child.proc.stdout.fileno(), 1024)
        self.assertEqual(child.event(10.0), {"event": "ready"})

if __name__ == "__main__":
    unittest.main()
