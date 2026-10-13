"""The write hooks of usearch, RaBitQ-Library's IVF and HNSW, and turbovec's IdMapIndex, on a tiny clustered dataset:
the hooks called directly, the write subcommand as the driver runs it, and one write batch of all four through the
driver.

A test that loads RaBitQ-Library runs in a process of its own, which runs just that test, since the library's OpenMP
runtime and faiss's abort a process that loads both, as a run of every test would. usearch and turbovec bring no OpenMP
runtime, so their tests run in the test process.
"""

import functools
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import helpers
from tests.inproc_runner import NAME, has, make_clustered
from adapters import graphs, rabitq_common, rabitq_hnsw, rabitq_ivf, turbovec, usearch_hnsw
from adapters.base import BuildContext, ServeContext, WriteContext
import driver
import schema
import writes

# Set in the process that runs one test of a class that takes `alone`.
ALONE = "COLDIVF_TEST_ALONE"

BASE = 2000
# Two rounds of the joint ladder and the inserts-only ladder's twin of the second, in calls of 50 rows.
WRITES = {
    "batch": 50, "latency_states": ["s0", "s5"], "latency_calls": 10, "warmup_calls": 2, "durable_calls": 4,
    "saves": 2, "seed": 1, "probes": 20,
    "states": {"s1": {"inserts": 50, "deletes": 50}, "s5": {"inserts": 200, "deletes": 200},
               "s5i": {"inserts": 200, "deletes": 0}},
}


def stream_config(path: Path) -> writes.WritesConfig:
    return writes.parse_writes(WRITES | {"stream": str(path)}, BASE, NAME)


def clone(src: Path, dst: Path) -> Path:
    writes.remove_tree(dst)
    writes.clone_tree(src, dst)
    return dst


def alone(cls: type) -> type:
    """Makes each test of `cls` run in a fresh process that runs only that test, where the test loads its library. The
    process that runs every test only starts it and checks that it passed."""

    def wrap(test):
        @functools.wraps(test)
        def run(self) -> None:
            if os.environ.get(ALONE):
                return test(self)

            name = f"{type(self).__module__}.{type(self).__name__}.{test.__name__}"
            out = subprocess.run([helpers.python(), "-m", "unittest", name], capture_output=True, text=True,
                                 cwd=helpers.PY, env=helpers.env({ALONE: "1"}), timeout=900)
            self.assertEqual(out.returncode, 0, out.stderr[-4000:])

        return run

    for name in [n for n in dir(cls) if n.startswith("test")]:
        setattr(cls, name, wrap(getattr(cls, name)))

    cls.isolated = True
    return cls


class SaveWholeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.file = self.folder / "index.bin"
        self.file.write_bytes(b"old")

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_new_file_and_its_side_file_replace_the_old_ones_by_a_rename(self):
        def save(path: str) -> None:
            Path(path).write_bytes(b"new index")
            Path(path + ".hnsw").write_bytes(b"graph")

        with open(self.file, "rb") as old:
            done = graphs.save_whole(self.file, save)
            # A reader of the old file keeps its bytes, since nothing wrote over them.
            self.assertEqual(old.read(), b"old")

        self.assertEqual(self.file.read_bytes(), b"new index")
        self.assertEqual((self.folder / "index.bin.hnsw").read_bytes(), b"graph")
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["index.bin", "index.bin.hnsw"])
        self.assertEqual(done | {"files": sorted(done["files"])},
                         {"call": "save", "files": ["index.bin", "index.bin.hnsw"], "bytes": 14, "sync": "F_FULLFSYNC",
                          "rename": True})

    def test_a_save_that_writes_nothing_fails_and_leaves_the_old_file(self):
        with self.assertRaises(FileNotFoundError):
            graphs.save_whole(self.file, lambda path: None)

        self.assertEqual(self.file.read_bytes(), b"old")


class HooksCase:
    """One system's write hooks on a build of the first 2,000 rows of a clustered dataset of 64 dimensions, with a
    stream of two rounds beside it. A test class mixes it in before unittest.TestCase."""

    module = ""
    cls: type | None = None
    variant = ""
    config: dict = {}
    params: dict = {}
    # The ops a stream session sends, and those of a latency run.
    stream_ops: list[str] = []
    latency_ops: list[str] = []
    # Set by `alone`, so the process that only starts each test builds nothing.
    isolated = False

    @classmethod
    def setUpClass(cls):
        if cls.isolated and not os.environ.get(ALONE):
            return

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ds = make_clustered(cls.root / "data", n=2300)
        cls.cfg = stream_config(cls.root / "writes" / "g.stream")
        cls.stream = writes.ensure_stream(cls.cfg, cls.ds)
        cls.adapter = cls.cls()
        cls.adapter.variant = cls.variant
        cls.adapter.load()
        cls.s0 = cls.root / "s0"
        cls.s0.mkdir()
        cls.adapter.build(BuildContext(config=cls.config, data_path=cls.ds.base_path, rows=BASE, out=cls.s0,
                                       clusters=None, threads=2, seed=1, metric="cosine", variant=cls.variant,
                                       dataset=cls.ds))

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "tmp", None) is not None:
            cls.tmp.cleanup()

    def rows(self, ids) -> np.ndarray:
        return writes.vectors(self.ds, np.asarray(ids, dtype=np.int64))

    def writer(self, folder: Path):
        ctx = WriteContext(index=folder, dataset=self.ds, rows=BASE, metric="cosine", threads=1, variant=self.variant)
        return self.adapter.writer_open(ctx, folder)

    def opened(self, folder: Path):
        ctx = ServeContext(index=folder, dataset=self.ds, rows=BASE, k=10, metric="cosine", threads=1,
                           variant=self.variant, params=self.params)
        handle = self.adapter.open(ctx, folder)
        self.adapter.configure(handle, self.params)
        return handle

    def search(self, handle, row: int) -> list[int]:
        query = np.asarray(self.ds.base[row], dtype=np.float32)
        return self.adapter.search(handle, query, 10, self.params).tolist()

    def check_searches(self, folder: Path, live: np.ndarray) -> None:
        """Every live insert comes back first as its own query, at least nearly every one, and no search returns an
        id the folder's state doesn't hold."""
        handle = self.opened(folder)
        inserted = np.flatnonzero(live[BASE:]) + BASE
        dead = set(np.flatnonzero(~live).tolist())
        first = [self.search(handle, r)[0] == r for r in inserted]
        self.assertGreaterEqual(np.mean(first), 0.95)
        found = [self.search(handle, r) for r in sorted(dead)] + [self.search(handle, r) for r in range(0, BASE, 7)]
        found += [self.adapter.search(handle, q, 10, self.params).tolist() for q in np.asarray(self.ds.queries)]
        self.assertEqual(sorted(dead & {i for f in found for i in f}), [])
        self.adapter.close(handle)

    def run_write(self, folder: Path, *args: str) -> list[dict]:
        cmd = [helpers.python(), "-m", self.module, "write", "--index", str(folder), "--stream", str(self.cfg.stream),
               "--data", str(self.ds.base_path), "--rows", str(BASE), "--threads", "1", "--variant", self.variant,
               *args]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=helpers.PY, env=helpers.env(), timeout=300)
        events = [json.loads(x) for x in out.stdout.splitlines() if x.strip()]
        self.assertTrue(events, out.stderr[-3000:])
        return events

    def ops(self, events: list[dict]) -> dict[str, dict]:
        self.assertEqual(events[-1]["event"], "done", events[-1])
        return {e["fields"]["op"]: e["fields"] for e in events if e["event"] == "op"}

    def test_hooks(self):
        folder = clone(self.s0, self.root / "hooks")
        w = self.writer(folder)
        self.adapter.resize(w, BASE + 51)
        ids = np.arange(BASE, BASE + 50)
        got = self.adapter.insert(w, ids, self.rows(ids))
        got_one = self.adapter.insert(w, np.array([BASE + 50]), self.rows([BASE + 50]))

        if got is not None:
            self.assertEqual(np.asarray(got, dtype=np.int64).tolist(), ids.tolist())
            self.assertEqual(np.asarray(got_one, dtype=np.int64).tolist(), [BASE + 50])

        self.adapter.delete(w, np.arange(0, 40))
        self.adapter.delete(w, np.array([45]))

        # A delete of an id the index no longer holds fails, so a stream that went wrong can't pass unseen.
        with self.assertRaises(RuntimeError):
            self.adapter.delete(w, np.array([45]))

        self.assertTrue(self.adapter.pending(w))
        done = self.adapter.durable_point(w)
        self.assertIsInstance(done, dict)
        self.adapter.writer_close(w)
        live = np.ones(BASE + 51, dtype=bool)
        live[list(range(40)) + [45]] = False
        self.check_searches(folder, live)

    def test_a_stream_session_and_a_latency_run(self):
        folder = clone(self.s0, self.root / "stream")
        got = self.run_write(folder, "--mode", "stream", "--pass", "joint", "--round", "2", "--batch", "50")
        ops = self.ops(got)
        self.assertEqual(sorted(ops), self.stream_ops)
        self.assertEqual((ops["insert"]["calls"], ops["insert"]["rows"], ops["delete"]["rows"]), (4, 200, 200))
        self.assertEqual((ops["insert"]["durability"], ops["durable_point"]["durability"]), ("memory", "power"))
        # The session's first op carries the writer's open.
        self.assertGreater(next(e["fields"] for e in got if e["event"] == "op")["open_us"], 0)
        self.assertTrue(ops["durable_point"]["extra"]["pending"])
        self.check_searches(folder, writes.state_mask(self.cfg, self.stream, "s5"))

        # A latency run on a clone of the state, with single writes of the reserved rows.
        folder = clone(folder, self.root / "latency")
        got = self.run_write(folder, "--mode", "latency", "--pass", "joint", "--round", "2", "--calls", "10",
                             "--warmup", "2", "--durable-calls", "4", "--saves", "2", "--reserve",
                             str(self.cfg.reserve_from), "--seed", "2")
        ops = self.ops(got)
        self.assertEqual(sorted(ops), self.latency_ops)
        self.assertEqual({ops[k]["batch"] for k in ("insert", "delete")}, {1})
        self.assertEqual((ops["insert"]["calls"], ops["delete"]["calls"]), (10, 10))

        if "durable_point" in ops:
            self.assertEqual(ops["durable_point"]["calls"], 2)
        else:
            self.assertEqual((ops["insert_sync"]["calls"], ops["sync"]["calls"]), (4, 4))
            self.assertEqual(ops["sync"]["durability"], "power")


@unittest.skipUnless(has("usearch"), "needs usearch")
class USearchHooksTest(HooksCase, unittest.TestCase):
    module = "adapters.usearch_hnsw"
    cls = usearch_hnsw.USearch
    config = {"dtype": "f16"}
    params = {"expansion_search": 64}
    stream_ops = ["delete", "durable_point", "insert"]
    latency_ops = ["delete", "durable_point", "insert"]

    def test_removed_nodes_stay_until_an_add_takes_them_over(self):
        folder = clone(self.s0, self.root / "reuse")
        w = self.writer(folder)
        self.adapter.delete(w, np.arange(0, 30))
        self.assertEqual(self.adapter.pending(w), {"live": BASE - 30, "nodes": BASE, "removed_nodes": 30})
        self.adapter.insert(w, np.arange(BASE, BASE + 10), self.rows(np.arange(BASE, BASE + 10)))
        self.assertEqual(self.adapter.pending(w), {"live": BASE - 20, "nodes": BASE, "removed_nodes": 20})
        self.adapter.writer_close(w)

    def test_a_view_of_the_old_file_keeps_working_through_a_durable_point(self):
        folder = clone(self.s0, self.root / "view")
        view = self.opened(folder)
        before = self.search(view, BASE + 3)
        w = self.writer(folder)
        self.adapter.insert(w, np.array([BASE + 3]), self.rows([BASE + 3]))
        self.adapter.durable_point(w)
        self.adapter.writer_close(w)
        # The view still maps the file it opened, which the save never wrote over.
        self.assertEqual(self.search(view, BASE + 3), before)
        self.assertNotIn(BASE + 3, before)
        self.adapter.close(view)
        fresh = self.opened(folder)
        self.assertEqual(self.search(fresh, BASE + 3)[0], BASE + 3)
        self.adapter.close(fresh)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), sorted([graphs.META, usearch_hnsw.FILE]))
        self.assertEqual(usearch_hnsw.layout(folder / usearch_hnsw.FILE).nodes, BASE + 1)

    def test_the_build_keeps_its_candidate_list_for_adds(self):
        folder = self.root / "ef-add"
        folder.mkdir()
        self.adapter.build(BuildContext(config={"dtype": "f16", "expansion_add": 40}, data_path=self.ds.base_path,
                                        rows=300, out=folder, clusters=None, threads=1, seed=1, metric="cosine",
                                        variant="", dataset=self.ds))
        w = self.writer(folder)
        self.assertEqual(w.index.expansion_add, 40)
        self.adapter.writer_close(w)


class RabitqCase(HooksCase):
    def test_inserts_numbered_past_the_stream_fail_the_session(self):
        folder = clone(self.s0, self.root / "numbers")
        args = ("--mode", "stream", "--pass", "inserts", "--round", "1", "--batch", "50")
        self.ops(self.run_write(folder, *args))
        # Room for more, so the HNSW's adds of the next session don't stop at its capacity first.
        w = self.writer(folder)
        self.adapter.resize(w, BASE + 200)
        self.adapter.durable_point(w)
        self.adapter.writer_close(w)
        # The same round again, on the state it left, numbers its inserts past the first round's.
        got = self.run_write(folder, *args)
        self.assertEqual(got[-1]["event"], "error")
        self.assertIn("not by their rows", got[-1]["message"])


@unittest.skipUnless(has("rabitqlib"), "needs rabitqlib")
@alone
class RabitqIvfHooksTest(RabitqCase, unittest.TestCase):
    module = "adapters.rabitq_ivf"
    cls = rabitq_ivf.RabitqIvf
    config = {"bits": 5, "nlist": 10}
    params = {"nprobe": 10}
    stream_ops = ["delete", "durable_point", "insert"]
    latency_ops = ["delete", "durable_point", "insert"]

    def test_an_add_grows_the_index_by_its_rows(self):
        folder = clone(self.s0, self.root / "grow")
        w = self.writer(folder)
        self.assertFalse(self.adapter.resize(w, BASE + 5))
        self.adapter.insert(w, np.arange(BASE, BASE + 5), self.rows(np.arange(BASE, BASE + 5)))
        self.adapter.delete(w, np.array([1, 2]))
        self.assertEqual(self.adapter.pending(w), {"slots": BASE + 5})
        self.adapter.writer_close(w)


@unittest.skipUnless(has("rabitqlib"), "needs rabitqlib")
@alone
class RabitqHnswHooksTest(RabitqCase, unittest.TestCase):
    module = "adapters.rabitq_hnsw"
    cls = rabitq_hnsw.RabitqHnsw
    config = {"bits": 4}
    params = {"ef": 64}
    stream_ops = ["delete", "durable_point", "insert", "resize"]
    latency_ops = ["delete", "durable_point", "insert", "resize"]

    def test_a_resize_makes_room_only_when_it_has_to(self):
        folder = clone(self.s0, self.root / "resize")
        w = self.writer(folder)
        self.assertFalse(self.adapter.resize(w, BASE))
        self.assertTrue(self.adapter.resize(w, BASE + 5))
        self.assertFalse(self.adapter.resize(w, BASE + 5))
        self.adapter.insert(w, np.arange(BASE, BASE + 5), self.rows(np.arange(BASE, BASE + 5)))

        # Past the room made, an add fails.
        with self.assertRaises(ValueError):
            self.adapter.insert(w, np.array([BASE + 5]), self.rows([BASE + 5]))

        self.adapter.delete(w, np.array([7]))
        self.assertEqual(self.adapter.pending(w), {"points": BASE + 5, "max_elements": BASE + 5})
        self.adapter.writer_close(w)


class RabitqRolesTest(unittest.TestCase):
    """Only the IVF and the HNSW take writes. SymphonyQG's index has no add or remove, and the base class they share
    takes none either."""

    def test_the_base_class_rebuilds(self):
        self.assertEqual(rabitq_common.RabitqAdapter.write_role, "rebuild")
        self.assertEqual((rabitq_ivf.RabitqIvf.write_role, rabitq_hnsw.RabitqHnsw.write_role), ("full", "full"))

    @unittest.skipUnless(has("rabitqlib"), "needs rabitqlib")
    def test_symphonyqg_takes_the_stream_as_a_fresh_build(self):
        out = subprocess.run([helpers.python(), "-m", "adapters.symphonyqg", "info"], capture_output=True, text=True,
                             cwd=helpers.PY, env=helpers.env(), timeout=300)
        info = json.loads(out.stdout.splitlines()[-1])
        self.assertEqual((info["write_role"], info["maintenance_actions"]), ("rebuild", []))


@unittest.skipUnless(has("turbovec"), "needs turbovec")
class TurboVecHooksTest(HooksCase, unittest.TestCase):
    module = "adapters.turbovec"
    cls = turbovec.TurboVec
    variant = "idmap"
    config = {"bits": 4, "calibrate": 500}
    params = {}
    stream_ops = ["delete", "durable_point", "insert", "sync"]
    latency_ops = ["delete", "insert", "insert_sync", "sync"]

    def test_the_build_is_a_synced_idmap_file(self):
        self.assertEqual(sorted(p.name for p in self.s0.iterdir()), [turbovec.IDMAP_FILE])
        index = self.adapter.tv.IdMapIndex.load(str(self.s0 / turbovec.IDMAP_FILE))
        self.assertEqual((len(index), index.calibration_state), (BASE, "calibrated"))
        self.assertTrue(index.contains(BASE - 1))
        self.assertFalse(index.contains(BASE))

    def test_a_remove_takes_the_row_out(self):
        folder = clone(self.s0, self.root / "remove")
        w = self.writer(folder)
        self.adapter.delete(w, np.arange(0, 25))
        self.adapter.insert(w, np.arange(BASE, BASE + 5), self.rows(np.arange(BASE, BASE + 5)))
        self.assertEqual(self.adapter.pending(w), {"vectors": BASE - 20})
        self.assertEqual(self.adapter.sync(w), {"call": "sync", "file": turbovec.IDMAP_FILE})
        self.adapter.writer_close(w)
        index = self.adapter.tv.IdMapIndex.load(str(folder / turbovec.IDMAP_FILE))
        self.assertEqual(len(index), BASE - 20)
        self.assertFalse(index.contains(3))

    def test_variants(self):
        def info(variant: str) -> dict:
            out = subprocess.run([helpers.python(), "-m", "adapters.turbovec", "info", "--variant", variant],
                                 capture_output=True, text=True, cwd=helpers.PY, env=helpers.env(), timeout=300)
            return json.loads(out.stdout.splitlines()[-1])

        self.assertEqual([info(v).get("write_role") for v in ("", "idmap")], ["rebuild", "full"])
        self.assertEqual(info("idmap")["sync_kind"], "incremental")
        self.assertEqual(info("sorted")["event"], "error")
        # A TurboQuantIndex has no delete by id, so it takes the stream as a fresh build.
        plain = self.root / "plain"
        plain.mkdir(exist_ok=True)
        adapter = turbovec.TurboVec()
        adapter.load()
        adapter.build(BuildContext(config={"bits": 4}, data_path=self.ds.base_path, rows=300, out=plain, clusters=None,
                                   threads=1, seed=1, metric="cosine", variant="", dataset=self.ds))
        self.assertEqual(sorted(p.name for p in plain.iterdir()), [turbovec.FILE])
        cmd = [helpers.python(), "-m", "adapters.turbovec", "write", "--index", str(plain), "--stream",
               str(self.cfg.stream), "--round", "1", "--data", str(self.ds.base_path), "--rows", "300"]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=helpers.PY, env=helpers.env(), timeout=300)
        last = json.loads(out.stdout.splitlines()[-1])
        self.assertEqual(last["event"], "unsupported")
        self.assertIn("fresh build", last["message"])


SUITE_HEAD = """
phase = "writes"
dataset = "{name}"
n = {n}
index_root = "{root}/indexes"
results_root = "{root}/results"
seed = 3
max_attempts = 1
wait_s = 0.05

[gate]
mode = "off"
invalid_vm = []

[writes]
stream = "{root}/writes/g.stream"
batch = 50
latency_states = ["s0", "s5"]
latency_calls = 10
warmup_calls = 2
durable_calls = 4
saves = 2
seed = 1
probes = 20

[writes.states]
s1 = {{ inserts = 50, deletes = 50 }}
s5 = {{ inserts = 200, deletes = 200 }}
s5i = {{ inserts = 200, deletes = 0 }}
"""

SUITE_SYSTEM = """
[[system]]
name = "{name}"
adapter = "{adapter}"
variant = "{variant}"

  [[system.build]]
  config_id = "c"
  config = {config}
  threads = 2

  [[system.write]]
  config_id = "c"
  role = "full"

  [[system.warm]]
  config_id = "c"
  states = "all"
  split = "test"
  passes = 1
  warmup = 4
  settings = [{params}]
  findability = true

  [[system.cold]]
  config_id = "c"
  states = ["s0", "s5"]
  params = {params}
  trials = 2
  next = 0
"""

SYSTEMS = (
    ("usearch", "usearch_hnsw", "", '{ dtype = "f16" }', "{ expansion_search = 64 }"),
    ("rabitq-ivf", "rabitq_ivf", "", "{ bits = 5, nlist = 10 }", "{ nprobe = 10 }"),
    ("rabitq-hnsw", "rabitq_hnsw", "", "{ bits = 4 }", "{ ef = 64 }"),
    ("turbovec", "turbovec", "idmap", "{ bits = 4, calibrate = 500 }", "{}"),
)


@unittest.skipUnless(all(has(m) for m in ("usearch", "rabitqlib", "turbovec")), "needs usearch, rabitqlib and turbovec")
class WriteBatchTest(unittest.TestCase):
    """All four systems through the driver: builds, both ladders, latency runs at S0 and S5, and warm and cold
    searches of their states, scored against each state's exact truth."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = make_clustered(self.root / "data", n=2300)
        self.env = mock.patch.dict(os.environ, {"COLDIVF_DATA": str(self.root / "data")})
        self.env.start()
        self.quiet = mock.patch.object(driver.Driver, "log", lambda self, msg: None)
        self.quiet.start()
        text = SUITE_HEAD.format(name=NAME, n=BASE, root=self.root) + "".join(
            SUITE_SYSTEM.format(name=n, adapter=a, variant=v, config=c, params=p) for n, a, v, c, p in SYSTEMS)
        path = self.root / "suite.toml"
        path.write_text(text)
        self.suite = driver.load_suite(path)
        writes.ensure_stream(self.suite.writes, self.ds)
        writes.compute_truths(self.suite.writes, self.ds, "numpy", log=lambda m: None)

    def tearDown(self):
        self.quiet.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_a_write_batch_of_all_four(self):
        self.assertEqual(driver.Driver(self.suite).run(), 0)

        for name, *_ in SYSTEMS:
            lines = schema.read(self.root / "results" / "writes" / f"{name}.jsonl")
            bad = [(x["kind"], x.get("op"), x.get("state"), x["status"], x["error"]) for x in lines
                   if x["status"] != "ok"]
            self.assertEqual(bad, [], name)

            for line in lines:
                self.assertEqual(schema.validate(line), [], name)

            stream = {(x["state_from"], x["state_to"]) for x in lines if x["kind"] == "write" and x["mode"] == "stream"}
            self.assertEqual(stream, {("s0", "s1"), ("s1", "s5"), ("s0", "s5i")}, name)
            latency = {x["state_from"] for x in lines if x["kind"] == "write" and x["mode"] == "latency"}
            self.assertEqual(latency, {"s0", "s5"}, name)
            warm = {x["state"]: x for x in lines if x["kind"] == "warm"}
            self.assertEqual(sorted(warm), ["s0", "s1", "s5", "s5i"], name)

            for state, x in warm.items():
                self.assertEqual(x["dead_returned"], 0.0, (name, state))
                self.assertGreaterEqual(x["recall10"], 0.5, (name, state))

            for state in ("s5", "s5i"):
                self.assertGreaterEqual(warm[state]["recall10_inserted"], 0.5, (name, state))
                self.assertGreaterEqual(warm[state]["extra"]["findability"]["inserted"], 0.95, (name, state))

            cold = [x for x in lines if x["kind"] == "cold"]
            self.assertEqual(sorted(x["state"] for x in cold), ["s0", "s0", "s5", "s5"], name)
            self.assertTrue(all(x["dead_returned"] == 0.0 for x in cold), name)


if __name__ == "__main__":
    unittest.main()
