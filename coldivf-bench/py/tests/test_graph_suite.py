import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import helpers
import driver
import schema

LIBRARIES = ("usearch", "qdrant_edge", "astervec")
HAVE_ALL = all(importlib.util.find_spec(name) is not None for name in LIBRARIES)

SYSTEM = """
[[system]]
name = "{name}"
adapter = "{adapter}"

  [[system.build]]
  config_id = "g"
  config = {config}
  threads = 2

  [[system.warm]]
  config_id = "g"
  split = "tune"
  passes = 1
  warmup = 4
  grid = {grid}

  [[system.warm]]
  config_id = "g"
  split = "tune"
  threads = 2
  warmup = 4
  qps_seconds = 0.3
  qps_repeats = 1
  grid = {grid}

  [[system.cold]]
  config_id = "g"
  regime = "c0"
  params = {params}
  trials = 2
  next = 1
  dummy_n = 300

  [[system.cold]]
  config_id = "g"
  regime = "c1"
  params = {params}
  trials = 2
  group = "live"
"""

HEAD = """
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
"""

# A second C1 cell on AsterVec's index in the same group. Its process finds the store claimed by the first one's, so it
# serves its trials from a private copy.
SECOND_C1 = """
  [[system.cold]]
  config_id = "g"
  regime = "c1"
  params = { ef = 32 }
  trials = 2
  group = "live"
"""

SYSTEMS = (
    ("usearch", "usearch_hnsw", '{ dtype = "i8" }', "{ expansion_search = [64] }", "{ expansion_search = 64 }"),
    ("qdrant-edge", "qdrant_shard", '{ codes = "sq8" }', "{ hnsw_ef = [64], oversampling = [2] }",
     "{ hnsw_ef = 64, oversampling = 2 }"),
    ("astervec", "astervec_lsm", "{}", "{ ef = [64] }", "{ ef = 64 }"),
)


def identity(folder: Path) -> tuple:
    files = [p for p in folder.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files), len(files), max(p.stat().st_mtime_ns for p in files)


@unittest.skipUnless(HAVE_ALL, "usearch, qdrant-edge-py and aster-vec aren't all installed")
class GraphSuiteTest(unittest.TestCase):
    """Runs the three graph adapters through the driver on a tiny dataset, with builds, warm runs at 1 and 2 threads,
    and C0 and C1 trials, then runs the batch again."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        helpers.make_dataset(self.root / "data", n=1500, nq=30)
        self.env = mock.patch.dict(os.environ, {"COLDIVF_DATA": str(self.root / "data")})
        self.env.start()
        self.quiet = mock.patch.object(driver.Driver, "log", lambda self, msg: None)
        self.quiet.start()
        text = HEAD.format(root=self.root) + "".join(
            SYSTEM.format(name=n, adapter=a, config=c, grid=g, params=p) for n, a, c, g, p in SYSTEMS) + SECOND_C1
        self.suite = self.root / "suite.toml"
        self.suite.write_text(text)

    def tearDown(self):
        self.quiet.stop()
        self.env.stop()
        self.tmp.cleanup()

    def batch(self) -> driver.Driver:
        d = driver.Driver(driver.load_suite(self.suite))
        self.assertEqual(d.run(), 0)
        return d

    def lines(self, name: str) -> list[dict]:
        return schema.read(self.root / "results" / "test" / f"{name}.jsonl")

    def test_whole_protocol_and_resume(self):
        self.batch()
        built, counts = {}, {}

        for name, *_ in SYSTEMS:
            lines = self.lines(name)
            counts[name] = len(lines)
            bad = [(x["kind"], x.get("regime"), x["status"], x["error"]) for x in lines if x["status"] != "ok"]
            self.assertEqual(bad, [], name)

            for line in lines:
                self.assertEqual(schema.validate(line), [], name)

            warm = [x for x in lines if x["kind"] == "warm"]
            self.assertEqual(len(warm), 2)
            self.assertGreaterEqual(min(x["recall10"] for x in warm if x["threads"] == 1), 0.9, name)
            self.assertGreater(min(x["qps"] for x in warm), 0, name)
            cold = [x for x in lines if x["kind"] == "cold"]
            c1_trials = 4 if name == "astervec" else 2
            self.assertEqual(sorted(x["regime"] for x in cold), ["c0", "c0"] + ["c1"] * c1_trials)
            self.assertTrue(all(x["resident_pages_at_go"] == 0 for x in cold), name)
            self.assertTrue(all(x["recall10"] >= 0.8 for x in cold), name)
            self.assertTrue(all(x["files_touched"] > 0 for x in cold if x["regime"] == "c0"), name)
            build = next(x for x in lines if x["kind"] == "build" and x["n"] == 1500)
            built[name] = Path(build["index_identity"]["path"])

        c1 = [x for x in self.lines("astervec") if x["kind"] == "cold" and x["regime"] == "c1"]
        private = {x["search_params"]["ef"]: (x["extra"] or {}).get("private_copy", False) for x in c1}
        self.assertEqual(sorted(private.values()), [False, True])

        for name in ("qdrant-edge", "astervec"):
            index = built[name]
            ident = driver.identity(index, checksum=False)
            build = next(x for x in self.lines(name) if x["kind"] == "build" and x["n"] == 1500)
            self.assertEqual({k: ident[k] for k in ("bytes", "files", "mtime_ns")},
                             {k: build["index_identity"][k] for k in ("bytes", "files", "mtime_ns")}, name)
            self.assertFalse(index.with_name(index.name + ".sessions").exists(), name)

        again = self.batch()
        rerun = [(name, x["kind"], x.get("regime")) for name, *_ in SYSTEMS for x in self.lines(name)[counts[name]:]]
        # A C1 process keeps its store open, so the index differs from the build until the process ends. Each cell's
        # trials are keyed on the index before its process starts, so a rerun repeats nothing.
        self.assertEqual(rerun, [])
        self.assertEqual(again.resumed, sum(counts.values()))


if __name__ == "__main__":
    unittest.main()
