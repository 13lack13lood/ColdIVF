import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import schema

NEEDED = {
    "build": {"seconds": 1.0, "index_bytes": 10},
    "warm": {"mode": "whole", "threads": 1, "p50_us": 5.0, "queries": 10},
    "cold": {"regime": "c0", "trial": 0, "total_us": 100.0},
    "update": {"epoch": 0},
    "ref": {"position": "start", "wave_us_p50": 3.0},
    "batch": {"started": "a", "ended": "b", "suite": "s", "suite_sha256": "x", "counts": {}, "by_system": {},
              "resumed": 0, "refs": {}, "flagged": False, "provenance_changed": False, "disk_free_gb": {}},
}


def make_line(kind: str, **fields) -> dict:
    line = schema.blank(kind)
    line.update(run_id="r", run_key="k", ts="t", batch_id="b", phase="p", system="s")
    line.update(NEEDED[kind])
    line.update(fields)
    return line


class ValidateTest(unittest.TestCase):
    def test_every_kind_validates(self):
        for kind in schema.KINDS:
            self.assertEqual(schema.validate(make_line(kind)), [], kind)

    def test_problems_are_found(self):
        line = make_line("cold")
        del line["ids"]
        line["surprise"] = 1
        line["trial"] = "zero"
        line["regime"] = None
        line["status"] = "fine"
        errors = " | ".join(schema.validate(line))
        self.assertIn("missing field ids", errors)
        self.assertIn("unknown field surprise", errors)
        self.assertIn("trial should be int", errors)
        self.assertIn("regime is null", errors)
        self.assertIn("unknown status", errors)

    def test_ok_lines_need_their_measurement(self):
        self.assertTrue(schema.validate(make_line("cold", total_us=None)))
        self.assertEqual(schema.validate(make_line("cold", total_us=None, status="failed")), [])
        self.assertEqual(schema.validate(make_line("warm", p50_us=None, qps=100.0)), [])

    def test_types(self):
        self.assertTrue(schema.validate(make_line("cold", ids=[1, 2.5])))
        self.assertTrue(schema.validate(make_line("cold", trial=True)))
        self.assertTrue(schema.validate(make_line("cold", total_us=float("nan"))))
        self.assertEqual(schema.validate(make_line("cold", ids=[1, 2], next_us=[1, 2.5])), [])

    def test_clean(self):
        line = schema.clean({"a": np.float32(1.5), "b": np.int64(3), "c": (1, 2), "d": float("inf"),
                             "e": np.array([1, 2]), "f": Path("/x")})
        self.assertEqual(line, {"a": 1.5, "b": 3, "c": [1, 2], "d": None, "e": [1, 2], "f": "/x"})
        self.assertIsInstance(line["b"], int)


class FileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "p" / "s.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_and_read(self):
        schema.append(self.path, make_line("cold", run_key="a"))
        schema.append(self.path, make_line("cold", run_key="b", status="rejected"))
        lines = schema.read(self.path)
        self.assertEqual([x["run_key"] for x in lines], ["a", "b"])

    def test_append_refuses_a_bad_line(self):
        with self.assertRaises(ValueError):
            schema.append(self.path, {"kind": "cold"})

        self.assertFalse(self.path.exists())

    def test_a_cut_line_is_skipped(self):
        schema.append(self.path, make_line("cold", run_key="a"))

        with open(self.path, "a") as f:
            f.write('{"run_key": "b", "stat')

        self.assertEqual(len(schema.read(self.path)), 1)

    def test_progress(self):
        schema.append(self.path, make_line("cold", run_key="a", status="rejected"))
        schema.append(self.path, make_line("cold", run_key="a"))
        schema.append(self.path, make_line("cold", run_key="b", status="failed"))
        schema.append(self.path, make_line("cold", run_key="c", status="unsupported"))
        progress = schema.Progress([self.path], max_attempts=2)
        self.assertTrue(progress.done("a"))
        self.assertFalse(progress.done("b"))
        self.assertTrue(progress.done("c"))
        self.assertEqual(progress.attempts("a"), 2)
        progress.add("b", "rejected")
        self.assertTrue(progress.done("b"))
        self.assertFalse(progress.done("z"))


class KeyTest(unittest.TestCase):
    def test_keys_are_stable(self):
        a = schema.run_key({"x": 1, "y": {"b": 2, "a": 1}})
        b = schema.run_key({"y": {"a": 1, "b": 2}, "x": 1})
        self.assertEqual(a, b)
        self.assertNotEqual(a, schema.run_key({"x": 2, "y": {"a": 1, "b": 2}}))
        self.assertEqual(len(a), 20)

    def test_utc(self):
        self.assertRegex(schema.utc_now(), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")


def git(repo: Path, *args: str) -> str:
    env = os.environ | {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env=env).stdout.strip()


class GitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        git(self.repo, "init", "-q")
        (self.repo / "a.txt").write_text("a\n")
        (self.repo / "dir").mkdir()
        (self.repo / "dir" / "b.txt").write_text("b\n")
        (self.repo / "dir" / "run.sh").write_text("#!/bin/sh\n")
        os.chmod(self.repo / "dir" / "run.sh", 0o755)
        (self.repo / "a.txt-2").write_text("c\n")
        (self.repo / "dir.x").write_text("d\n")
        os.symlink("a.txt", self.repo / "link")
        git(self.repo, "add", "-A")

    def tearDown(self):
        self.tmp.cleanup()

    def test_tree_hash_matches_write_tree(self):
        self.assertEqual(schema.staged_tree_hash(self.repo), git(self.repo, "write-tree"))

    def test_diff_hash_moves_with_unstaged_changes(self):
        clean = schema.diff_hash(self.repo)
        (self.repo / "a.txt").write_text("changed\n")
        edited = schema.diff_hash(self.repo)
        (self.repo / "new.txt").write_text("new\n")
        untracked = schema.diff_hash(self.repo)
        self.assertEqual(len({clean, edited, untracked}), 3)
        state = schema.git_state(self.repo)
        self.assertTrue(state["unstaged"])
        self.assertEqual(state["tree_hash"], git(self.repo, "write-tree"))

    def test_outside_a_repo(self):
        with tempfile.TemporaryDirectory() as other:
            self.assertIsNone(schema.staged_tree_hash(Path(other)))


class MachineTest(unittest.TestCase):
    def test_record(self):
        m = schema.machine()

        for key in ("model", "brand", "perflevels", "pagesize", "cachelinesize", "os", "power", "rust"):
            self.assertIn(key, m)

        if m["perflevels"]:
            self.assertIn("l2cachesize", m["perflevels"][0])

        json.dumps(m)

    def test_provenance(self):
        p = schema.provenance(git={"head": "x"}, env={"A": "1"})
        self.assertEqual(p["head"], "x")
        self.assertEqual(p["env"], {"A": "1"})
        self.assertIsNotNone(p["pip_freeze_sha256"])


if __name__ == "__main__":
    unittest.main()
