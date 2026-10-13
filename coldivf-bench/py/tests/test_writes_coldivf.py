"""The write paths of ColdIVF's Python module adapter, the flat scan as the exact reference, and the systems that take
the stream as a fresh build, on tiny synthetic data.

The ColdIVF tests that call the module itself run its adapter in an interpreter that has it, this one or
$COLDIVF_BENCH_VENVS/venv-coldivf/bin/python3 as in the suites, and are skipped without one.
"""

import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests import helpers, inproc_runner
from adapters import coldivf as binaries
from adapters import coldivf_py as cp
from adapters.base import BuildContext, ServeContext, Unsupported, WriteContext
from adapters.flat import ExactReference, Flat
from adapters.inproc import HarnessWrites
import data
import schema
import writes

# The seven sections of a fake index file, the header first.
SIZES = [4096, 512, 1024, 256, 8192, 4096, 800]


def has(module: str) -> bool:
    """Whether this interpreter imports `module` from a file of its own, not a folder of the same name."""
    spec = importlib.util.find_spec(module)
    return spec is not None and spec.origin is not None


def module_python() -> str | None:
    """An interpreter that has the coldivf module: this one, or the one the suites name."""
    if has("coldivf"):
        return sys.executable

    venvs = os.environ.get("COLDIVF_BENCH_VENVS")
    path = Path(venvs) / "venv-coldivf" / "bin" / "python3" if venvs else None
    return str(path) if path is not None and path.exists() else None


PYTHON = module_python()


def run(python: str, module: str, *args: str) -> list[dict]:
    """The events of one adapter subcommand, run as the driver runs it."""
    out = subprocess.run([python, "-m", module, *args], capture_output=True, text=True, cwd=helpers.PY,
                         env=helpers.env(), timeout=300)
    events = [json.loads(x) for x in out.stdout.splitlines() if x.strip()]

    if not events:
        raise AssertionError(out.stderr[-3000:])

    return events


def ops(events: list[dict]) -> dict[str, dict]:
    return {e["fields"]["op"]: e["fields"] for e in events if e["event"] == "op"}


def stream_config(path: Path, n: int, name: str, inserts: tuple[int, int, int]) -> writes.WritesConfig:
    """Three rounds that end with `inserts` inserts and as many deletes, the inserts-only ladder and a policy pass."""
    a, b, c = inserts
    states = {"s1": {"inserts": a, "deletes": a}, "s5": {"inserts": b, "deletes": b},
              "s10": {"inserts": c, "deletes": c}, "s1i": {"inserts": a}, "s5i": {"inserts": b},
              "s10i": {"inserts": c}, "s10p": {"inserts": c, "deletes": c, "policy": "auto"}}
    table = {"stream": str(path), "batch": 50, "latency_states": ["s0", "s10"], "latency_calls": 20,
             "warmup_calls": 5, "durable_calls": 10, "saves": 2, "seed": 1, "probes": 40, "states": states}
    return writes.parse_writes(table, n, name)


def scorer(ds: data.Dataset, cfg: writes.WritesConfig, stream: writes.Stream, state: str) -> writes.StateRecall:
    """The scorer of a state against the exact truth over its live vectors."""
    live = writes.state_mask(cfg, stream, state)
    ids, dists = writes.exact_truth(ds, live, k=100)
    return writes.StateRecall(ds, data.Truth(Path("unused.gt.ibin"), ids, ds.count, dists), live, cfg.n)


def write_data(path: Path) -> None:
    """A file with a ColdIVF version 1 header whose seven sections have SIZES, one after another."""
    head = bytearray(4096)
    head[:8] = binaries.MAGIC
    struct.pack_into("<I", head, 8, 1)
    offset = 0

    for i, size in enumerate(SIZES):
        struct.pack_into("<QQII", head, 80 + 24 * i, offset, size, i + 1, 0)
        offset += size

    path.write_bytes(bytes(head) + b"\0" * (offset - 4096))


def module_data_file(path: str) -> str:
    """What coldivf.data_file returns: the index's path until its folder exists, then the data file there, with
    symbolic links resolved."""
    folder = Path(f"{path}.d")
    return os.path.realpath(next(folder.glob("g*.data")) if folder.is_dir() else path)


class FakeJob:
    def __init__(self, writer: "FakeWriter"):
        self.writer = writer

    def run(self, throttle: bool = True) -> None:
        self.writer.calls.append(("run", throttle))

    def finish(self) -> SimpleNamespace:
        self.writer.calls.append(("finish",))
        return report("minor")


def report(kind: str) -> SimpleNamespace:
    return SimpleNamespace(**{name: 0 for name in cp.REPORT_FIELDS} | {"kind": kind, "count": 4950})


class FakeWriter:
    """Records what an adapter calls on the module's Writer, and answers as the module does."""

    def __init__(self, path: str):
        self.path, self.calls, self.due, self.job = path, [], True, True

    def insert(self, id: int, vector: np.ndarray) -> None:
        self.calls.append(("insert", id, np.asarray(vector).shape))

    def insert_many(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        self.calls.append(("insert_many", list(ids), np.asarray(vectors).shape))

    def delete(self, id: int) -> None:
        self.calls.append(("delete", id))

    def delete_many(self, ids: np.ndarray) -> None:
        self.calls.append(("delete_many", list(ids)))

    def sync(self) -> None:
        self.calls.append(("sync",))

    def maintain(self, kind: str = "fold", **params) -> SimpleNamespace:
        self.calls.append(("maintain", kind, params))
        return report(kind)

    def begin_compact(self, **params) -> FakeJob | None:
        self.calls.append(("begin_compact", params))
        return FakeJob(self) if self.job else None

    def status(self, **params) -> SimpleNamespace:
        self.calls.append(("status", params))
        return SimpleNamespace(**{name: 0 for name in cp.STATUS_FIELDS} | {"due": self.due})

    def close(self) -> None:
        self.calls.append(("close",))


class IndexPathTest(unittest.TestCase):
    def test_an_index_keeps_the_files_path_once_maintenance_moved_it_into_its_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "tl320-s10"
            folder.mkdir()
            (folder / "index.coldivf").write_bytes(b"x")
            self.assertEqual(cp.index_path(folder), folder / "index.coldivf")
            (folder / "index.coldivf").unlink()
            (folder / "index.coldivf.d").mkdir()
            (folder / "index.coldivf.lock").touch()
            self.assertEqual(cp.index_path(folder), folder / "index.coldivf")
            self.assertEqual(cp.index_path(folder / "index.coldivf"), folder / "index.coldivf")
            (folder / "other.coldivf").write_bytes(b"x")

            with self.assertRaises(FileNotFoundError):
                cp.index_path(folder)

            with self.assertRaises(FileNotFoundError):
                cp.index_path(Path(tmp))


class FilesTest(unittest.TestCase):
    """The files of an index through a symbolic link, as /var leads to /private/var, which the module resolves."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        (Path(self.tmp.name) / "real").mkdir()
        self.folder = Path(self.tmp.name) / "link"
        self.folder.symlink_to(Path(self.tmp.name) / "real")
        self.adapter = cp.ColdIVFPython()
        self.adapter.lib = SimpleNamespace(data_file=module_data_file)

    def tearDown(self):
        self.tmp.cleanup()

    def names(self, files: list[dict]) -> list[str]:
        self.assertTrue(all(f["path"].startswith(str(self.folder) + "/") for f in files))
        return [os.path.relpath(f["path"], self.folder) for f in files]

    def test_a_built_index_lists_its_file_its_log_and_its_lock_as_the_binaries_adapter_does(self):
        write_data(self.folder / "index.coldivf")
        files = self.adapter.files(self.folder)
        self.assertEqual(self.names(files), ["index.coldivf", "index.coldivf.delta", "index.coldivf.lock"])
        self.assertEqual([r["kind"] for r in files[0]["ranges"]], ["routing"] * 4 + ["bulk"] * 3)
        self.assertEqual(files[1]["ranges"], [])
        self.assertEqual(files, binaries.ColdIVF().files(self.folder))

    def test_a_maintained_index_lists_its_folder(self):
        folder = self.folder / "index.coldivf.d"
        folder.mkdir()
        write_data(folder / "g1-0123456789abcdef.data")
        (folder / "g1-0123456789abcdef.log").write_bytes(b"\0" * 64)

        for name in ("root.0", "root.1", "runs"):
            (folder / name).write_bytes(b"\0" * 4096)

        (folder / "g2-fedcba9876543210.resume").write_bytes(b"\0" * 10)
        (self.folder / "index.coldivf.lock").touch()
        files = self.adapter.files(self.folder)
        names = [f"index.coldivf.d/{x}" for x in ("g1-0123456789abcdef.data", "g1-0123456789abcdef.log")]
        names += ["index.coldivf.lock"] + [f"index.coldivf.d/{x}" for x in cp.FOLDER_FILES]
        self.assertEqual(self.names(files), names + ["index.coldivf.d/g2-fedcba9876543210.resume"])
        self.assertEqual([r["kind"] for r in files[0]["ranges"]], ["routing"] * 4 + ["bulk"] * 3)
        self.assertEqual([files[i]["ranges"][0]["kind"] for i in (1, 3, 7)], ["bulk", "routing", "bulk"])
        # The activity word isn't there until a search maps it, so it is watched with no range.
        self.assertEqual((files[6]["bytes"], files[6]["ranges"]), (0, []))


class HooksTest(unittest.TestCase):
    """The write hooks over a stand-in for the module, which records each call."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        (self.folder / "index.coldivf").write_bytes(b"x")
        self.adapter = cp.ColdIVFPython()
        self.adapter.lib = SimpleNamespace(Writer=FakeWriter, data_file=module_data_file)
        ctx = WriteContext(index=self.folder, dataset=None, rows=5000, metric="cosine", threads=1, variant="python",
                           maintain_params={"vectors_per_list": 320}, maintain_threads=10)
        self.w = self.adapter.writer_open(ctx, self.folder)
        self.calls = self.w.writer.calls

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_writer_opens_the_index_and_keeps_the_policy_passs_settings(self):
        self.assertEqual(self.w.writer.path, str(self.folder / "index.coldivf"))
        self.assertEqual((self.w.params, self.w.threads), ({"vectors_per_list": 320}, 10))
        self.assertEqual((cp.ColdIVFPython.write_role, cp.ColdIVFPython.sync_kind, cp.ColdIVFPython.durability),
                         ("full", "incremental", {"call": "process", "sync": "power"}))
        self.assertEqual(cp.ColdIVFPython.maintenance_actions, ("fold", "auto", "major", "job"))
        self.adapter.writer_close(self.w)
        self.assertEqual(self.calls[-1], ("close",))

    def test_a_batch_of_one_takes_the_single_calls(self):
        self.assertIsNone(self.adapter.insert(self.w, np.array([7]), np.ones((1, 128), dtype=np.float32)))
        self.adapter.delete(self.w, np.array([3]))
        self.adapter.insert(self.w, np.array([8, 9]), np.ones((2, 128), dtype=np.float32))
        self.adapter.delete(self.w, np.array([4, 5]))
        self.assertEqual(self.adapter.sync(self.w)["call"], "Writer.sync")
        self.assertEqual(self.calls, [("insert", 7, (128,)), ("delete", 3), ("insert_many", [8, 9], (2, 128)),
                                      ("delete_many", [4, 5]), ("sync",)])
        self.assertIs(type(self.calls[0][1]), int)

    def test_every_run_takes_the_suites_params_and_its_own_threads(self):
        for action in ("fold", "auto", "major"):
            out = self.adapter.maintain(self.w, action, 4, {"vectors_per_list": 320})
            self.assertEqual(self.calls[-1], ("maintain", action, {"vectors_per_list": 320, "threads": 4}))
            self.assertEqual((out["kind"], out["rows_live"], out["durability"]), (action, 4950, "power"))

        # The data file is the index's own path until its folder exists, and then the file there, under the path the
        # session was given. An explicit thread count wins over one among the params.
        self.assertEqual(out["data_file"], self.folder / "index.coldivf")
        (self.folder / "index.coldivf").unlink()
        (self.folder / "index.coldivf.d").mkdir()
        (self.folder / "index.coldivf.d" / "g1-0123456789abcdef.data").touch()
        out = self.adapter.maintain(self.w, "fold", 3, {"vectors_per_list": 320, "threads": 0})
        self.assertEqual(self.calls[-1][2]["threads"], 3)
        self.assertEqual(out["data_file"], self.folder / "index.coldivf.d" / "g1-0123456789abcdef.data")

    def test_the_job_begins_runs_throttled_and_finishes_on_the_calling_thread(self):
        out = self.adapter.maintain(self.w, "job", 2, {"vectors_per_list": 320})
        self.assertEqual(self.calls, [("begin_compact", {"vectors_per_list": 320, "threads": 2}), ("run", True),
                                      ("finish",)])
        self.assertEqual(out["kind"], "minor")
        self.assertTrue(all(out[k] >= 0 for k in ("begin_s", "run_s", "finish_s")))
        self.w.writer.job = False

        with self.assertRaises(RuntimeError):
            self.adapter.maintain(self.w, "job", 2, {"vectors_per_list": 320})

        with self.assertRaises(Unsupported):
            self.adapter.maintain(self.w, "compact", 2, {})

    def test_status_calls_take_explicit_threads(self):
        self.assertTrue(self.adapter.due(self.w, {"vectors_per_list": 320}))
        self.assertEqual(self.calls[-1], ("status", {"vectors_per_list": 320, "threads": 10}))
        self.assertEqual(sorted(self.adapter.pending(self.w)), sorted(cp.STATUS_FIELDS))
        # After a run, the status calls take that run's settings.
        self.adapter.maintain(self.w, "fold", 4, {"vectors_per_list": 320})
        self.adapter.pending(self.w)
        self.assertEqual(self.calls[-1], ("status", {"vectors_per_list": 320, "threads": 4}))
        self.w.writer.due = False
        self.assertFalse(self.adapter.due(self.w, {}))


@unittest.skipUnless(PYTHON, "no interpreter here has the coldivf module")
class ModuleTest(unittest.TestCase):
    """The adapter's write and maintain subcommands on the module itself, as the driver runs them, on 3,000 clustered
    rows of 128 dimensions and a stream of 300 inserts and 300 deletes in batches of 50."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ds = inproc_runner.make_clustered(cls.root / "data", n=3400, dim=128, nq=40, groups=60)
        cls.cfg = stream_config(cls.root / "w.stream", 3000, inproc_runner.NAME, (30, 150, 300))
        cls.stream = writes.ensure_stream(cls.cfg, cls.ds)
        cls.s0 = cls.root / "s0"
        cls.s0.mkdir()
        build = ("import sys, numpy as np, coldivf, data; rows = data.Dataset.from_base(sys.argv[1]).base[:3000]; "
                 "coldivf.build(sys.argv[2], np.asarray(rows), vectors_per_list=64, threads=2)")
        subprocess.run([PYTHON, "-c", build, str(cls.ds.base_path), str(cls.s0 / "index.coldivf")], check=True,
                       cwd=helpers.PY, env=helpers.env(), timeout=300)
        # S1 in one session and S10 from it in another, as the driver's work folder takes the joint ladder.
        cls.s1 = cls.clone(cls.s0, "s1")
        cls.s1_events = cls.write(cls.s1, "--mode", "stream", "--pass", "joint", "--round", "1", "--batch", "50")
        cls.s10 = cls.clone(cls.s1, "s10")
        cls.s10_events = cls.write(cls.s10, "--mode", "stream", "--pass", "joint", "--round", "3", "--from-round",
                                   "1", "--batch", "50")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def clone(cls, src: Path, name: str) -> Path:
        dst = cls.root / name
        writes.clone_tree(src, dst)
        return dst

    @classmethod
    def common(cls, threads: int = 1) -> list[str]:
        return ["--data", str(cls.ds.base_path), "--rows", "3000", "--threads", str(threads), "--variant", "python"]

    @classmethod
    def write(cls, index: Path, *args: str) -> list[dict]:
        return run(PYTHON, "adapters.coldivf_py", "write", *cls.common(), "--index", str(index), "--stream",
                   str(cls.cfg.stream), *args)

    def maintain(self, index: Path, action: str, threads: int = 2) -> dict:
        got = run(PYTHON, "adapters.coldivf_py", "maintain", *self.common(threads), "--index", str(index),
                  "--action", action, "--params", json.dumps({"vectors_per_list": 64}))
        self.assertEqual(got[-1]["event"], "maintain", got[-1])
        return got[-1]["fields"]

    def check_search(self, index: Path, state: str) -> None:
        """Searches the state at every cluster. Deleted rows never come back, every live row searched as a query comes
        back first, and recall against the state's exact truth is about 1."""
        live = writes.state_mask(self.cfg, self.stream, state)
        probes = writes.probe_rows(self.cfg, self.stream, state)
        gone = np.flatnonzero(~live)[:40]

        with tempfile.TemporaryDirectory() as ids:
            np.save(Path(ids) / "probes.npy", np.concatenate([probes, gone]))
            got = run(PYTHON, "adapters.coldivf_py", "warm", *self.common(), "--k", "10", "--index", str(index),
                      "--grid", json.dumps([{"nprobe": 64, "rerank": 100}]), "--split", "test", "--ids-dir", ids,
                      "--passes", "1", "--warmup", "2", "--probes", str(Path(ids) / "probes.npy"))
            setting = got[0]
            self.assertEqual(setting["event"], "setting", got)
            found, probed = np.load(setting["ids_file"]), np.load(setting["probes_file"])

        summary = scorer(self.ds, self.cfg, self.stream, state).summary(np.asarray(setting["qids"]), found)
        self.assertEqual(summary["dead_returned"], 0.0, state)
        self.assertGreaterEqual(summary["recall10"], 0.95, state)
        findable = writes.findability(self.ds, probes, probed[: len(probes)], self.cfg.n)
        self.assertEqual((findable["inserted"], findable["base"]), (1.0, 1.0), (state, findable))
        self.assertFalse(np.isin(gone, probed[len(probes) :]).any(), state)

    def test_stream_sessions_make_each_change_durable_and_searches_see_them(self):
        s1 = ops(self.s1_events)
        self.assertEqual((self.s1_events[0]["event"], self.s1_events[-1]["event"]), ("opened", "done"))
        self.assertEqual(sorted(s1), ["delete", "durable_point", "insert", "sync"])
        self.assertEqual([s1[k]["calls"] for k in ("insert", "delete", "sync", "durable_point")], [1, 1, 2, 1])
        self.assertEqual([s1[k]["durability"] for k in ("insert", "delete", "sync", "durable_point")],
                         ["process", "process", "power", "power"])
        self.assertEqual((s1["durable_point"]["extra"]["pending"]["pending"],
                          s1["durable_point"]["extra"]["pending"]["deleted"]), (30, 30))
        s10 = ops(self.s10_events)
        self.assertEqual((s10["insert"]["calls"], s10["insert"]["rows"], s10["delete"]["rows"]), (6, 270, 270))
        self.assertGreater(s10["insert"]["rows_per_s"], 0)
        self.assertGreater(s10["durable_point"]["extra"]["pending"]["pending"], 200)
        self.assertEqual(sorted(p.name for p in self.s10.iterdir()),
                         ["index.coldivf", "index.coldivf.delta", "index.coldivf.lock"])
        self.check_search(self.s1, "s1")
        self.check_search(self.s10, "s10")

    def test_the_inserts_alone_reach_s10i(self):
        folder = self.clone(self.s0, "s10i")
        got = ops(self.write(folder, "--mode", "stream", "--pass", "inserts", "--round", "3", "--batch", "50"))
        self.assertEqual(sorted(got), ["durable_point", "insert", "sync"])
        self.assertEqual(got["insert"]["rows"], 300)
        self.check_search(folder, "s10i")

    def test_a_latency_run_times_single_writes_and_durable_inserts(self):
        folder = self.clone(self.s10, "latency-s10")
        got = ops(self.write(folder, "--mode", "latency", "--pass", "joint", "--round", "3", "--calls", "20",
                             "--warmup", "5", "--durable-calls", "10", "--reserve", str(self.cfg.reserve_from),
                             "--seed", "2"))
        self.assertEqual(sorted(got), ["delete", "insert", "insert_sync", "sync"])
        self.assertEqual([got[k]["calls"] for k in ("insert", "delete", "insert_sync", "sync")], [20, 20, 10, 10])
        self.assertEqual({got[k]["batch"] for k in got}, {1})
        self.assertEqual((got["insert"]["durability"], got["insert_sync"]["durability"]), ("process", "power"))
        self.assertLessEqual(got["sync"]["p50_us"], got["insert_sync"]["p50_us"])

    def test_each_action_folds_every_change_in_and_the_job_writes_the_auto_runs_file(self):
        done = {}

        for action in ("fold", "auto", "major", "job"):
            folder = self.clone(self.s10, f"s10-{action}")
            fields = self.maintain(folder, action, threads=2 if action == "job" else 3)
            report = fields["extra"]["report"]
            self.assertEqual((fields["action"], fields["rows_live"], fields["durability"]), (action, 3000, "power"))
            self.assertEqual(fields["threads"], 2 if action == "job" else 3)
            self.assertEqual(report["kind"], {"fold": "fold", "major": "major"}.get(action, "minor"))
            self.assertEqual((fields["extra"]["pending"]["pending"], fields["extra"]["pending"]["deleted"]), (0, 0))
            data_file = Path(fields["extra"]["data_file"])
            self.assertEqual(data_file.parent, folder / "index.coldivf.d")
            self.assertFalse((folder / "index.coldivf").exists())
            names = {os.path.relpath(f["path"], folder) for f in
                     run(PYTHON, "adapters.coldivf_py", "files", "--index", str(folder))[-1]["files"]}
            self.assertLessEqual({os.path.relpath(data_file, folder), "index.coldivf.lock", "index.coldivf.d/root.0",
                                  "index.coldivf.d/runs"}, names)
            done[action] = schema.file_sha256(data_file)

            if action != "job":
                self.check_search(folder, "s10")

        self.assertIn("run_s", report)
        self.assertEqual(done["job"], done["auto"])

    def test_the_policy_pass_runs_maintenance_whenever_it_is_due(self):
        folder = self.clone(self.s0, "s10p")
        got = self.write(folder, "--mode", "stream", "--pass", "policy", "--round", "3", "--batch", "50",
                         "--policy", "auto", "--maintain-params", json.dumps({"vectors_per_list": 64}),
                         "--maintain-threads", "2")
        runs = [e["fields"] for e in got if e["event"] == "maintain"]
        self.assertGreaterEqual(len(runs), 1)
        self.assertEqual({(r["action"], r["threads"], json.dumps(r["params"])) for r in runs},
                         {("auto", 2, json.dumps({"vectors_per_list": 64}))})
        self.assertTrue(all(r["extra"]["pending_before"]["due"] for r in runs))
        point = ops(got)["durable_point"]
        self.assertEqual(point["extra"]["runs"], len(runs))
        self.assertLess(point["extra"]["rows_per_s_with_runs"], point["extra"]["rows_per_s_durable"])
        self.check_search(folder, "s10p")


class ExactReferenceTest(unittest.TestCase):
    """The flat scan's writes, which the harness makes, and its search over them, on 5,000 rows of 16 dimensions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=5600, nq=12)
        self.cfg = stream_config(self.root / "w.stream", 5000, "tiny-16-cosine", (50, 250, 500))
        self.stream = writes.ensure_stream(self.cfg, self.ds)
        self.s0 = self.root / "s0"
        self.s0.mkdir()
        Flat().build(BuildContext(config={}, data_path=self.ds.base_path, rows=5000, out=self.s0, clusters=None,
                                  threads=1, seed=1, metric="cosine", variant="", dataset=self.ds))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, index: Path, *args: str) -> list[dict]:
        return run(helpers.python(), "adapters.flat", "write", "--data", str(self.ds.base_path), "--rows", "5000",
                   "--index", str(index), "--stream", str(self.cfg.stream), "--batch", "100", "--mode", "stream",
                   *args)

    def search(self, index: Path, queries: np.ndarray) -> np.ndarray:
        adapter = ExactReference()
        ctx = ServeContext(index=index, dataset=self.ds, rows=5500, k=10, metric="cosine", threads=1, variant="")
        handle = adapter.open(ctx, index)
        return np.stack([adapter.search(handle, np.asarray(q, dtype=np.float32), 10, {}) for q in queries])

    def check(self, index: Path, state: str) -> None:
        """The state's recall is 1 with no dead id, and every live row searched as a query comes back first, while a
        deleted one never comes back."""
        qids = np.arange(len(self.ds.queries))
        summary = scorer(self.ds, self.cfg, self.stream, state).summary(qids, self.search(index, self.ds.queries))
        self.assertEqual((summary["recall10"], summary["dead_returned"]), (1.0, 0.0), state)
        self.assertIn(summary["recall10_inserted"], (1.0, None), state)
        live = writes.state_mask(self.cfg, self.stream, state)
        probes = writes.probe_rows(self.cfg, self.stream, state)
        findable = writes.findability(self.ds, probes, self.search(index, np.asarray(self.ds.base[probes])), 5000)
        self.assertEqual(findable["base"], 1.0, state)
        self.assertIn(findable["inserted"], (1.0, None), state)
        gone = np.flatnonzero(~live)[:30]

        if len(gone):
            self.assertFalse(np.isin(gone, self.search(index, np.asarray(self.ds.base[gone]))).any(), state)

    def test_every_state_of_both_ladders_scores_one(self):
        inserts = self.root / "inserts"
        writes.clone_tree(self.s0, inserts)
        self.check(self.s0, "s0")
        work = self.s0

        for state, start, end in (("s1", 0, 1), ("s5", 1, 2), ("s10", 2, 3)):
            got = ops(self.write(work, "--pass", "joint", "--round", str(end), "--from-round", str(start)))
            self.assertEqual([got[k]["durability"] for k in ("insert", "delete", "sync", "durable_point")],
                             ["process", "process", "power", "power"])
            self.check(work, state)

        meta = json.loads((work / "meta.json").read_text())
        self.assertEqual((meta["rows"], meta["state_rows"], meta["live_rows"]), (5000, 5500, 5000))

        for state, start, end in (("s1i", 0, 1), ("s5i", 1, 2), ("s10i", 2, 3)):
            got = ops(self.write(inserts, "--pass", "inserts", "--round", str(end), "--from-round", str(start)))
            self.assertNotIn("delete", got)
            self.check(inserts, state)

    def test_a_build_without_writes_scans_its_rows_alone(self):
        handle = Flat().open(ServeContext(index=self.s0, dataset=self.ds, rows=5000, k=10, metric="cosine",
                                          threads=1, variant=""), self.s0)
        self.assertEqual((handle["inserts"], handle["live"], len(handle["blocks"])), (None, None, 1))
        want, _ = helpers.exact(np.asarray(self.ds.base)[:5000], np.asarray(self.ds.queries), 10)
        self.assertTrue(np.array_equal(self.search(self.s0, self.ds.queries), want))

    def test_it_mixes_in_the_harnesss_writes_and_the_test_adapter_still_can(self):
        from tests.fake_adapter import Fake

        for cls in (ExactReference, Fake):
            self.assertLess(cls.__mro__.index(HarnessWrites), cls.__mro__.index(Flat))

        info = run(helpers.python(), "adapters.flat", "info")[-1]
        self.assertEqual((info["system"], info["write_role"], info["sync_kind"], info["maintenance_actions"]),
                         ("flat", "exact", "incremental", []))


class RebuildOnlyTest(unittest.TestCase):
    """DiskANN, SymphonyQG and Paimon have no write call, so each takes the stream as a fresh build."""

    def test_each_refuses_a_write_session_and_reports_its_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = helpers.make_dataset(root / "data", n=5600, nq=6)
            cfg = stream_config(root / "w.stream", 5000, "tiny-16-cosine", (50, 250, 500))
            writes.ensure_stream(cfg, ds)

            for module, library in (("adapters.diskann", None), ("adapters.symphonyqg", "rabitqlib"),
                                    ("adapters.paimon", "paimon_vindex")):
                with self.subTest(module=module):
                    if library is not None and not has(library):
                        self.skipTest(f"{library} isn't installed")

                    got = run(helpers.python(), module, "write", "--index", str(root), "--stream", str(cfg.stream),
                              "--round", "1", "--data", str(ds.base_path), "--rows", "5000")
                    self.assertEqual(got[-1]["event"], "unsupported")
                    self.assertIn("fresh build", got[-1]["message"])
                    info = run(helpers.python(), module, "info")[-1]
                    self.assertEqual((info["write_role"], info["sync_kind"], info["maintenance_actions"]),
                                     ("rebuild", "whole", []))


if __name__ == "__main__":
    unittest.main()
