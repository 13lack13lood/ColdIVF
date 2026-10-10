import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
from adapters import base
import evictor
import rusage


def events(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]


class Adapter:
    """Runs the test adapter's subcommands as the driver does."""

    def __init__(self, root: Path, variant: str = ""):
        self.root, self.variant = root, variant
        self.data = root / "tiny-16-cosine.base.fbin"

    def cmd(self, *args: str) -> list[str]:
        return [helpers.python(), "-m", "tests.fake_adapter", *args]

    def run(self, *args: str, stdin: str = "") -> tuple[list[dict], str]:
        out = subprocess.run(self.cmd(*args), input=stdin, capture_output=True, text=True, cwd=helpers.PY,
                             env=helpers.env(), timeout=300)
        return events(out.stdout), out.stderr

    def common(self, rows: int = 500) -> list[str]:
        return ["--data", str(self.data), "--rows", str(rows), "--variant", self.variant]


class HelpersTest(unittest.TestCase):
    def test_percentile_is_nearest_rank(self):
        values = list(range(1, 101))
        self.assertEqual(base.percentile(values, 50), 50)
        self.assertEqual(base.percentile(values, 99), 99)
        self.assertEqual(base.percentile(values, 100), 100)
        self.assertEqual(base.percentile([7.0], 1), 7.0)
        self.assertIsNone(base.percentile([], 50))

    def test_median_of_passes(self):
        passes = [{"p50_us": 1.0, "p99_us": None}, {"p50_us": 3.0, "p99_us": 4.0}, {"p50_us": 2.0, "p99_us": 6.0}]
        self.assertEqual(base.median_of_passes(passes), {"p50_us": 2.0, "p99_us": 5.0})

    @unittest.skipUnless(rusage.MACOS, "CPU time is read on macOS")
    def test_spin_keeps_several_cores_busy(self):
        before = rusage.Usage.settled()
        start = time.perf_counter()
        base.spin(3, 0.2)
        wall = time.perf_counter() - start
        used = rusage.Usage.settled() - before
        self.assertLess(wall, 0.4)
        self.assertGreater(used.cpu_ms, 2.0 * 200)

    def test_cpu_states(self):
        start = time.perf_counter()
        base.cpu_state("idle", 1)
        self.assertGreater(time.perf_counter() - start, base.IDLE_S * 0.9)
        base.cpu_state("none", 1)

        with self.assertRaises(ValueError):
            base.cpu_state("hot", 1)

    def test_read_except_skips_the_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            path.write_bytes(b"x" * 1000)
            self.assertEqual(base.read_except(path, []), 1000)
            self.assertEqual(base.read_except(path, [(100, 300), (900, 2000)], chunk=64), 700)
            self.assertEqual(base.read_except(path, [(0, 1000)]), 0)

    @unittest.skipUnless(rusage.MACOS, "residency is read with mincore on macOS")
    def test_a_whole_warm_up_leaves_loaded_ranges_unread(self):
        page = evictor.PAGE

        class Loads(base.Adapter):
            def files(self, path):
                out = super().files(path)
                out[0]["ranges"] = [{"offset": 0, "length": 4 * page, "kind": "bulk"},
                                    {"offset": 4 * page, "length": 4 * page, "kind": "bulk", "loaded": True}]
                out[1]["ranges"][0]["loaded"] = True
                return out

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            for name in ("a.bin", "b.bin"):
                (root / name).write_bytes(os.urandom(8 * page))

            with evictor.Evictor([root]) as ev:
                ev.sync()
                ev.evict()
                ctx = base.ServeContext(index=root, dataset=None, rows=0, k=10, metric="cosine", threads=1,
                                        variant="")
                Loads().warm_up(None, ctx)
                a, b = ev.maps[str(root / "a.bin")], ev.maps[str(root / "b.bin")]
                # Another process may read the first two pages of a newly mapped file back, so those aren't checked.
                self.assertEqual((a.resident(0, 4 * page)[0], a.resident(4 * page)[0], b.resident(2 * page)[0]),
                                 (4, 0, 0))

    def test_the_warm_up_query_is_one_no_trial_uses(self):
        class Recorder(base.Adapter):
            def __init__(self):
                super().__init__()
                self.searched = []

            def open(self, ctx, path):
                return path

            def search(self, handle, query, k, params):
                self.searched.append((handle, float(query[0])))
                return np.arange(k)

        cell = base.ServeContext(index=Path("index"), dataset=None, rows=10, k=2, metric="cosine", threads=1,
                                 variant="", queries={1: np.full(4, 1.0), 3: np.full(4, 3.0)},
                                 warmup=[np.full(4, 8.0)], dummy=Path("dummy"))

        for regime, where in (("c1", Path("index")), ("c0", Path("dummy"))):
            adapter = Recorder()
            cell.regime = regime
            adapter.cold_setup(cell)
            self.assertEqual(adapter.searched, [(where, 8.0)], regime)

        # Without warm-up queries it falls back to the cell's first query.
        cell.warmup = []
        self.assertEqual(float(Recorder().warmup_query(cell)[0]), 1.0)

    def test_cold_hands_the_warm_up_ids_to_the_adapter(self):
        class Seen(base.Adapter):
            def cold_setup(self, ctx):
                self.seen = (ctx.warmup_ids, [float(v[0]) for v in ctx.warmup])
                raise base.Unsupported("seen")

        with tempfile.TemporaryDirectory() as tmp:
            ds = helpers.make_dataset(Path(tmp), n=200, nq=10)
            args = base._parser().parse_args(["cold", "--data", str(ds.base_path), "--rows", "200", "--index", tmp,
                                              "--queries", "1", "--warmup-queries", "4,5"])
            adapter = Seen()

            with self.assertRaises(base.Unsupported):
                base._cold(adapter, args, None, ds)

            self.assertEqual(adapter.seen, ([4, 5], [float(ds.queries[4][0]), float(ds.queries[5][0])]))

    def test_wait_until_never_ends_early(self):
        for spin in (False, True):
            deadline = base.now_ns() + 20_000_000
            base.wait_until(deadline, spin=spin)
            late = base.now_ns() - deadline
            self.assertGreaterEqual(late, 0)
            # A loaded machine can run the waiting thread late, so this only rules out a wait that sleeps on.
            self.assertLess(late, 50_000_000)


class SubcommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ds = helpers.make_dataset(cls.root, n=500, nq=40)
        cls.index = cls.root / "index"
        cls.adapter = Adapter(cls.root)
        got, _ = cls.adapter.run("build", *cls.adapter.common(), "--out", str(cls.index))
        cls.build = got

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build(self):
        self.assertEqual(len(self.build), 1)
        msg = self.build[0]
        self.assertEqual(msg["event"], "build")
        fields = msg["fields"]
        self.assertGreater(fields["seconds"], 0)
        self.assertEqual(fields["stored_vector_bytes"], 500 * 16 * 4)
        self.assertTrue(fields["exact_f32_stored"])
        self.assertGreater(fields["peak_footprint"], 0)
        self.assertEqual(msg["info"]["system"], "fake")
        self.assertTrue((self.index / "vectors.npy").exists())

    def test_info_and_files(self):
        got, _ = self.adapter.run("info")
        self.assertEqual(got[0]["event"], "info")
        self.assertIn("c1", got[0]["regimes"])
        self.assertEqual((got[0]["workers"], got[0]["worker_qos"]), ("", "default"))
        got, _ = self.adapter.run("files", "--index", str(self.index))
        files = {Path(f["path"]).name: f for f in got[0]["files"]}
        self.assertEqual(set(files), {"vectors.npy", "meta.json"})
        self.assertEqual(files["vectors.npy"]["ranges"][0]["kind"], "bulk")

    def test_library_output_stays_off_the_wire(self):
        got, err = Adapter(self.root, "noisy").run("info", "--variant", "noisy")
        self.assertEqual([m["event"] for m in got], ["info"])
        self.assertIn("noise on stdout", err)

    def test_warm_ids_match_the_truth(self):
        with tempfile.TemporaryDirectory() as ids:
            grid = json.dumps([{}, {"unused": 1}])
            got, _ = self.adapter.run("warm", *self.adapter.common(), "--index", str(self.index), "--grid", grid,
                                      "--split", "tune", "--ids-dir", ids, "--passes", "2", "--warmup", "5")
            settings = [m for m in got if m["event"] == "setting"]
            self.assertEqual(len(settings), 2)
            self.assertEqual(got[-1]["event"], "done")
            fields = settings[0]["fields"]
            self.assertEqual((fields["queries"], fields["passes"], fields["threads"]), (20, 2, 1))
            self.assertGreater(fields["p50_us"], 0)
            self.assertGreaterEqual(fields["p99_us"], fields["p50_us"])
            self.assertGreater(fields["resident_file_bytes"], 0)
            self.assertEqual((fields["resident_loaded_bytes"], fields["loaded_bytes_dropped"]), (0, 0))
            self.assertEqual(fields["memory_total"], fields["footprint"] + fields["resident_file_bytes"])
            self.assertEqual(fields["footprint_since_open"], fields["footprint"] - fields["footprint_before_open"])
            found = np.load(settings[0]["ids_file"])
            qids = np.asarray(settings[0]["qids"])
            np.testing.assert_array_equal(qids, np.arange(0, 40, 2))
            np.testing.assert_array_equal(found, np.asarray(self.ds.truth(500).ids)[qids, :10])

    def test_warm_qps_with_threads(self):
        with tempfile.TemporaryDirectory() as ids:
            got, _ = self.adapter.run("warm", *self.adapter.common(), "--threads", "2", "--index", str(self.index),
                                      "--grid", "[{}]", "--ids-dir", ids, "--warmup", "2", "--qps-seconds", "0.2",
                                      "--qps-repeats", "2")
            fields = [m for m in got if m["event"] == "setting"][0]["fields"]
            self.assertGreater(fields["qps"], 0)
            self.assertIsNone(fields.get("p50_us"))
            self.assertEqual(fields["threads"], 2)

    def test_warm_qps_with_worker_processes(self):
        workers = Adapter(self.root, "procqps")

        with tempfile.TemporaryDirectory() as ids:
            got, err = workers.run("warm", *workers.common(), "--threads", "2", "--index", str(self.index), "--grid",
                                   "[{}]", "--ids-dir", ids, "--warmup", "2", "--qps-seconds", "0.3",
                                   "--qps-repeats", "1")
            settings = [m for m in got if m["event"] == "setting"]
            self.assertTrue(settings, err)
            fields = settings[0]["fields"]
            self.assertEqual(fields["processes"], 2)
            self.assertGreater(fields["qps"], 0)
            # The workers' own CPU time and memory count, each a process with a Python of its own.
            self.assertGreater(fields["cpu_ms_per_query"], 0.0005)
            self.assertEqual(len(fields["worker_footprints"]), 2)
            self.assertEqual(fields["footprint"], sum(fields["worker_footprints"]))
            self.assertGreater(fields["footprint"], fields["parent_footprint"])
            self.assertEqual(fields["memory_total"], fields["footprint"] + fields["resident_file_bytes"])

    def test_loaded_files_are_read_by_the_open_and_counted_apart(self):
        loaded = Adapter(self.root, "loaded")
        got, _ = loaded.run("files", "--index", str(self.index), "--variant", "loaded")
        self.assertTrue(all(r["loaded"] for f in got[0]["files"] for r in f["ranges"]))

        with tempfile.TemporaryDirectory() as ids:
            got, err = loaded.run("warm", *loaded.common(), "--index", str(self.index), "--grid", "[{}]",
                                  "--ids-dir", ids, "--passes", "1", "--warmup", "5")
            fields = [m for m in got if m["event"] == "setting"][0]["fields"]

        # The fake maps its rows, so its searches cache them again after the drop, and they count apart.
        self.assertGreater(fields["loaded_bytes_dropped"], 0, err)
        self.assertEqual(fields["resident_file_bytes"], 0)
        self.assertGreater(fields["resident_loaded_bytes"], 0)

    def go(self, regime: str, queries: list[int], extra: list[str] = (), commands: list[dict] = ()) -> list[dict]:
        args = ["cold", *self.adapter.common(), "--index", str(self.index), "--queries",
                ",".join(map(str, queries)), "--regime", regime, "--cpu-state", "none", *extra]
        stdin = "".join(json.dumps(c) + "\n" for c in commands)
        got, err = self.adapter.run(*args, stdin=stdin)
        self.assertTrue(got, err)
        return got

    def test_c0_trial(self):
        got = self.go("c0", [1, 3, 5], ["--dummy", str(self.index)],
                      [{"cmd": "go", "trial": 0, "queries": [1, 3, 5]}])
        self.assertEqual([m["event"] for m in got], ["ready", "trial"])
        fields = got[1]["fields"]
        self.assertEqual(fields["ids"], np.asarray(self.ds.truth(500).ids)[1, :10].tolist())
        self.assertEqual(len(fields["next_us"]), 2)
        self.assertAlmostEqual(fields["total_us"], fields["open_us"] + fields["search_us"], delta=1.0)
        self.assertGreater(fields["resident_rerun_us"], 0)
        self.assertIn("legacy", fields["cpu_qos_ms"])

    def test_c1_trials_share_one_process(self):
        commands = [{"cmd": "go", "trial": t, "queries": [q]} for t, q in enumerate([1, 3, 5])] + [{"cmd": "stop"}]
        got = self.go("c1", [1, 3, 5], commands=commands)
        self.assertEqual([m["event"] for m in got], ["ready", "trial", "trial", "trial"])
        self.assertIsNone(got[1]["fields"]["open_us"])
        self.assertEqual(got[3]["fields"]["ids"], np.asarray(self.ds.truth(500).ids)[5, :10].tolist())

    def test_intent_counts_from_the_vector(self):
        args = self.adapter.common() + ["--index", str(self.index), "--queries", "7", "--regime", "intent",
                                        "--cpu-state", "prewarm", "--delay-ms", "30"]
        got, _ = self.adapter.run("cold", *args, stdin=json.dumps({"cmd": "go", "trial": 0, "queries": [7]}) + "\n")
        fields = got[1]["fields"]
        # The open finishes long before 30 ms here, so the timer covers the search and any lateness of the wait.
        self.assertLess(fields["open_us"], 30000)
        self.assertGreaterEqual(fields["total_us"], fields["search_us"])
        self.assertLess(fields["total_us"], 30000)

    def test_stop_before_go(self):
        got = self.go("c0", [1], commands=[{"cmd": "stop"}])
        self.assertEqual([m["event"] for m in got], ["ready"])

    def test_unsupported_regime(self):
        noc1 = Adapter(self.root, "noc1")
        got, _ = noc1.run("cold", *noc1.common(), "--index", str(self.index), "--queries", "1", "--regime", "c1")
        self.assertEqual(got[0]["event"], "unsupported")

    def test_errors_are_reported(self):
        crash = Adapter(self.root, "crash")
        got, _ = crash.run("cold", *crash.common(), "--index", str(self.index), "--queries", "1", "--cpu-state",
                           "none", stdin=json.dumps({"cmd": "go", "trial": 0, "queries": [1]}) + "\n")
        self.assertEqual([m["event"] for m in got], ["ready", "error"])
        self.assertIn("a crash on purpose", got[1]["message"])

    def test_a_program_can_take_over_the_wire(self):
        program = Adapter(self.root, "exec")
        got, err = program.run("cold", *program.common(), "--index", str(self.index), "--queries", "1",
                               stdin=json.dumps({"cmd": "go", "trial": 4, "queries": [1]}) + "\n")
        self.assertEqual([m["event"] for m in got], ["ready", "trial"])
        self.assertEqual(got[1]["trial"], 4)
        self.assertTrue(got[1]["fields"]["extra"]["program"])
        self.assertIn("goes to stderr", err)

    def test_overhead(self):
        got, _ = self.adapter.run("overhead", *self.adapter.common())
        self.assertEqual(got[0]["event"], "overhead")
        self.assertGreater(got[0]["fields"]["overhead_us"], 0)


if __name__ == "__main__":
    unittest.main()
