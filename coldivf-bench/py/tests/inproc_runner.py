"""Runs an in-process adapter's subcommands the way the driver does, and reads its events, for the library tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
import data

NAME = "tiny-64-cosine"


def make_clustered(root: Path, n: int = 2000, dim: int = 64, nq: int = 40, groups: int = 20,
                   seed: int = 0) -> data.Dataset:
    """Writes a dataset of unit vectors spread around `groups` centers, with its exact truth. Unlike vectors spread
    evenly over the sphere, it has near neighbors that an approximate search finds."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((groups, dim))

    def around(count: int) -> np.ndarray:
        rows = centers[rng.integers(0, groups, count)] + 0.35 * rng.standard_normal((count, dim))
        return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)

    base, queries = around(n), around(nq)
    ids, _ = helpers.exact(base, queries, 100)
    root.mkdir(parents=True, exist_ok=True)
    data.write_fbin(root / f"{NAME}.base.fbin", base)
    data.write_fbin(root / f"{NAME}.query.fbin", queries)
    data.write_ibin(root / f"{NAME}.gt.ibin", ids.astype(np.int32))
    return data.Dataset(NAME, root, root / "derived")


def recall10(ds: data.Dataset, setting: dict) -> float:
    """The tie-aware recall@10 of a warm setting's ids, which `Runner.warm` loads."""
    truth = data.Recall(ds, ds.truth(ds.count))
    return truth.summary(np.asarray(setting["qids"]), setting["ids"])["recall10"]


def has(module: str) -> bool:
    """Whether this interpreter can import `module`."""
    return importlib.util.find_spec(module) is not None


class Runner:
    """One adapter module, eg. "adapters.rabitq_ivf", over the first `rows` rows of a dataset's base."""

    def __init__(self, module: str, base: Path, rows: int):
        self.module, self.base, self.rows = module, Path(base), rows

    def run(self, *args: str, stdin: str = "", timeout: float = 300) -> tuple[list[dict], str]:
        cmd = [helpers.python(), "-m", self.module, *args]
        out = subprocess.run(cmd, input=stdin, capture_output=True, text=True, cwd=helpers.PY, env=helpers.env(),
                             timeout=timeout)
        return [json.loads(line) for line in out.stdout.splitlines() if line.strip()], out.stderr

    def common(self, threads: int = 1) -> list[str]:
        return ["--data", str(self.base), "--rows", str(self.rows), "--threads", str(threads)]

    def build(self, out: Path, config: dict, *extra: str, threads: int = 2) -> tuple[dict, str]:
        """The build event, or whatever event came instead, and the log."""
        got, err = self.run("build", *self.common(threads), "--out", str(out), "--config", json.dumps(config), *extra)
        return (got[-1] if got else {}), err

    def warm(self, index: Path, grid: list[dict], *extra: str, threads: int = 1) -> tuple[list[dict], str]:
        """Every event of a warm run on the tune split, with short passes and short rate measurements."""
        with tempfile.TemporaryDirectory() as ids:
            got, err = self.run("warm", *self.common(threads), "--index", str(index), "--grid", json.dumps(grid),
                                "--split", "tune", "--ids-dir", ids, "--passes", "1", "--warmup", "4",
                                "--qps-seconds", "0.3", "--qps-repeats", "1", *extra)

            for msg in got:
                if msg.get("ids_file"):
                    msg["ids"] = np.load(msg["ids_file"])

        return got, err

    def cold(self, index: Path, params: dict, regime: str, queries: list[int], commands: list[dict],
             *extra: str) -> tuple[list[dict], str]:
        args = ["cold", *self.common(), "--index", str(index), "--params", json.dumps(params), "--queries",
                ",".join(map(str, queries)), "--regime", regime, "--cpu-state", "none", *extra]
        return self.run(*args, stdin="".join(json.dumps(c) + "\n" for c in commands))


def go(trial: int, queries: list[int]) -> dict:
    return {"cmd": "go", "trial": trial, "queries": queries}


class LibraryCase(unittest.TestCase):
    """Builds of one adapter on a small clustered dataset, shared by its tests."""

    module = ""
    rows = 2000

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ds = make_clustered(cls.root, n=cls.rows)
        cls.runner = Runner(cls.module, cls.ds.base_path, cls.rows)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def built(self, name: str, config: dict, *extra: str) -> tuple[Path, dict]:
        out = self.root / name
        msg, err = self.runner.build(out, config, *extra)
        self.assertEqual(msg.get("event"), "build", err[-2000:])
        return out, msg

    def settings(self, index: Path, grid: list[dict], **kw) -> list[dict]:
        got, err = self.runner.warm(index, grid, **kw)
        found = [m for m in got if m["event"] == "setting"]
        self.assertEqual(len(found), len(grid), err[-2000:])
        return found

    def check_cold(self, index: Path, params: dict) -> None:
        got, err = self.runner.cold(index, params, "c0", [1, 3, 5], [go(0, [1, 3, 5])], "--dummy", str(index))
        self.assertEqual([m["event"] for m in got], ["ready", "trial"], err[-2000:])
        fields = got[1]["fields"]
        self.assertEqual(len(fields["ids"]), 10)
        self.assertTrue(all(0 <= i < self.rows for i in fields["ids"]))
        self.assertEqual(len(fields["next_us"]), 2)
        self.assertAlmostEqual(fields["total_us"], fields["open_us"] + fields["search_us"], delta=1.0)
        got, err = self.runner.cold(index, params, "c1", [1, 3], [go(0, [1]), go(1, [3]), {"cmd": "stop"}])
        self.assertEqual([m["event"] for m in got], ["ready", "trial", "trial"], err[-2000:])
        self.assertIsNone(got[1]["fields"]["open_us"])
        self.assertEqual(got[1]["fields"]["ids"], fields["ids"])

    def check_error(self, got: list[dict], text: str) -> None:
        self.assertEqual(got[-1]["event"], "error")
        self.assertIn(text, got[-1]["message"])
