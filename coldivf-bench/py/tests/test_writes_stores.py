"""The write hooks of LanceDB, sqlite-vec, qdrant-edge and AsterVec, and the write helpers of the sealed stores, on tiny
synthetic data. Each system's sessions run where its library is installed, eg. LanceDB's retrain needs pylance, which
venv-lance13 has."""

import importlib.util
import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers
from adapters import lancedb, qdrant_shard, sqlite_vec, stores
from adapters.base import BuildContext, ServeContext
import writes

# A stream of two rounds on the build, with latency runs of a few calls.
STREAM = {
    "batch": 40, "latency_states": ["s0", "s10"], "latency_calls": 6, "warmup_calls": 2, "durable_calls": 4,
    "saves": 2, "seed": 1, "probes": 20,
    "states": {"s1": {"inserts": 40, "deletes": 40}, "s10": {"inserts": 200, "deletes": 200},
               "s10i": {"inserts": 200, "deletes": 0}},
}
# The fields every op of a session reports, and those of the ops whose counters the session reads.
TIMES = ("op", "batch", "calls", "rows", "durability", "first_us", "max_us", "p50_us", "p90_us", "p99_us", "mean_us",
         "mode", "threads", "index_bytes_before", "index_bytes_after")
COUNTERS = ("cpu_ms", "logical_bytes_written", "disk_bytes_written", "footprint_before", "footprint_after",
            "lifetime_max_footprint", "swapins_delta", "compressions_delta")


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def identity(folder: Path) -> tuple:
    files = [p for p in folder.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files), len(files), max(p.stat().st_mtime_ns for p in files)


class ClonesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_clone_full_gives_hard_links_files_of_their_own(self):
        folder = self.root / "store"
        (folder / "sub").mkdir(parents=True)
        (folder / "sub" / "vectors.bin").write_bytes(b"v" * 5000)
        (folder / "alone.bin").write_bytes(b"a")
        os.utime(folder / "sub" / "vectors.bin", ns=(1_700_000_000_000_000_000,) * 2)
        os.link(folder / "sub" / "vectors.bin", self.root / "elsewhere.bin")
        before = (folder / "sub" / "vectors.bin").stat()
        self.assertEqual(stores.clone_full(folder), ["sub/vectors.bin"])
        after = (folder / "sub" / "vectors.bin").stat()
        self.assertNotEqual(after.st_ino, before.st_ino)
        self.assertEqual((after.st_size, after.st_mtime_ns, after.st_nlink), (5000, before.st_mtime_ns, 1))

        with open(folder / "sub" / "vectors.bin", "r+b") as f:
            f.write(b"x")

        self.assertEqual((self.root / "elsewhere.bin").read_bytes()[:1], b"v")
        self.assertEqual(stores.clone_full(folder), [])
        self.assertEqual(sorted(p.name for p in (folder / "sub").iterdir()), ["vectors.bin"])

    def test_a_store_sealed_again_gets_a_new_stash(self):
        index = self.root / "index"
        data = index / "store"
        data.mkdir(parents=True)
        (data / "config.json").write_bytes(b"{}")
        (data / "segment.bin").write_bytes(b"s" * 100)
        store = stores.Store(index, "store")

        def session(folder: Path) -> None:
            with open(folder / "config.json", "r+b") as f:
                f.write(b"[")

        self.assertEqual(store.seal(session)["stash"], ["config.json"])
        # Writes change the store, and a session now rewrites the segment instead.
        (data / "segment.bin").write_bytes(b"t" * 200)
        (data / "new.log").write_bytes(b"log")

        def other(folder: Path) -> None:
            (folder / "segment.bin").write_bytes(b"u" * 10)

        record = store.seal(other)
        self.assertEqual(record["stash"], ["segment.bin"])
        self.assertEqual(sorted(p.name for p in (index / "stash").iterdir()), ["segment.bin"])
        self.assertEqual(record["files"]["segment.bin"][0], 200)
        self.assertEqual(stores.Store(index, "store").record, record)
        other(data)
        self.assertEqual(store.restore(), ["segment.bin"])
        self.assertEqual((data / "segment.bin").read_bytes(), b"t" * 200)


class HelpersTest(unittest.TestCase):
    def test_qdrant_fixup_passes_the_builds_last_upsert(self):
        self.assertEqual(qdrant_shard.fixup_ops(1_000_000), 501)
        self.assertEqual(qdrant_shard.fixup_ops(20_000), 11)
        self.assertEqual(qdrant_shard.fixup_ops(4500), 4)
        self.assertEqual(qdrant_shard.fixup_ops(2000), 2)

    def test_qdrant_log_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp)
            self.assertEqual(qdrant_shard.log_bytes(shard)["wal_bytes"], 0)
            (shard / "wal").mkdir()

            (shard / "wal" / "open-1").write_bytes(b"w" * 4096)
            (shard / "wal" / "open-2").write_bytes(b"w" * 100)
            self.assertEqual(qdrant_shard.log_bytes(shard), {"wal_bytes": 4196, "wal_files": 2})

    def test_sqlite_scans_read_every_slot_and_rank_the_live_rows(self):
        self.assertEqual(sqlite_vec.scan_counts("f32", 900, 4, 10, 1, 1000),
                         {"dist_evals": 900, "bytes_scanned": 16000})
        self.assertEqual(sqlite_vec.scan_counts("f32", 900, 4, 10, 1), {"dist_evals": 900, "bytes_scanned": 14400})
        self.assertEqual(sqlite_vec.scan_counts("int8", 900, 4, 10, 4, 1000),
                         {"dist_evals": 940, "bytes_scanned": 4000 + 40 * 16})

    def test_lance_deletion_files_are_routing(self):
        self.assertEqual(lancedb.kind_of("t.lance/_deletions/0-3-96.arrow"), "routing")
        self.assertEqual(lancedb.kind_of("t.lance/_transactions/3-5d.txn"), "bulk")
        self.assertEqual(lancedb.kind_of("t.lance/data/0111.lance"), "bulk")

    def test_lance_sync_takes_new_and_renamed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "old.lance").write_bytes(b"o")
            (root / "hint.json").write_bytes(b"1")
            before = lancedb.listing(root)
            (root / "data" / "new.lance").write_bytes(b"n" * 10)
            (root / "hint.tmp").write_bytes(b"22")
            os.replace(root / "hint.tmp", root / "hint.json")
            after = lancedb.listing(root)
            self.assertEqual(sorted(Path(p).name for p in after), ["hint.json", "new.lance", "old.lance"])
            self.assertEqual(lancedb.sync_new(before, after), {"files": 2, "folders": 2, "bytes": 12})
            self.assertEqual(lancedb.sync_new(after, after), {"files": 0, "folders": 0, "bytes": 0})


class WriteCase(unittest.TestCase):
    """One system's write path on a tiny build, as the driver runs it: the stream's two rounds on one work folder, a
    latency run on a copy of the build and on one of the last state, and each maintenance action on a copy of the last
    state. Each copy is a plain copy for a system that writes in place and a clone for any other, as the driver makes
    them. Subclasses name the adapter, its build and the search settings that check the states."""

    module = ""
    cls_name = ""
    config: dict = {}
    params: dict = {}
    n = 1200
    rows = 1500

    @classmethod
    def setUpClass(cls):
        if not cls.module:
            raise unittest.SkipTest("the base of the write cases")

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ds = helpers.make_dataset(cls.root / "data", n=cls.rows, nq=8, tie_query=None)
        cls.cfg = writes.parse_writes(STREAM | {"stream": str(cls.root / "w.stream")}, cls.n, cls.ds.name)
        cls.stream = writes.ensure_stream(cls.cfg, cls.ds)
        cls.build = cls.root / "build"
        cls.build.mkdir()
        cls.make().build(BuildContext(config=cls.config, data_path=cls.ds.base_path, rows=cls.n, out=cls.build,
                                      clusters=None, threads=2, seed=1, metric="cosine", variant="", dataset=cls.ds))
        cls.work = cls.root / "work"
        cls.scratch(cls.build, cls.work)
        cls.rounds = [cls.write(cls.work, "--mode", "stream", "--round", "1", "--from-round", "0"),
                      cls.write(cls.work, "--mode", "stream", "--round", "2", "--from-round", "1")]
        cls.info = cls.call("info")[-1]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def scratch(cls, source: Path, dest: Path) -> None:
        """A folder for a session to write to: a plain copy for a system that writes in place, else a clone."""
        (writes.copy_tree if cls.make().writes_in_place else writes.clone_tree)(source, dest)

    @classmethod
    def make(cls):
        module = importlib.import_module(cls.module)
        adapter = getattr(module, cls.cls_name)()
        adapter.load()
        return adapter

    @classmethod
    def call(cls, command: str, *args: str, stdin: str = "") -> list[dict]:
        cmd = [helpers.python(), "-m", cls.module, command, *args]

        if command != "info":
            cmd += ["--data", str(cls.ds.base_path), "--rows", str(cls.n)]

        out = subprocess.run(cmd, input=stdin, capture_output=True, text=True, cwd=helpers.PY, env=helpers.env(),
                             timeout=600)
        events = [json.loads(x) for x in out.stdout.splitlines() if x.strip()]

        if not events or events[-1]["event"] in ("error", "unsupported"):
            raise AssertionError(f"{command} sent {events[-1:] or 'nothing'}: {out.stderr[-3000:]}")

        return events

    @classmethod
    def write(cls, index: Path, *args: str) -> list[dict]:
        return cls.call("write", "--index", str(index), "--stream", str(cls.cfg.stream), "--pass", "joint",
                        "--batch", str(cls.cfg.batch), *args)

    def latency(self, source: Path, round_: int) -> dict:
        clone = self.root / f"latency-{round_}"
        writes.remove_tree(clone)
        self.scratch(source, clone)
        got = self.write(clone, "--mode", "latency", "--round", str(round_), "--calls", "6", "--warmup", "2",
                         "--durable-calls", "4", "--saves", "2", "--reserve", str(self.cfg.reserve_from), "--seed",
                         "2")
        return {e["fields"]["op"]: e["fields"] for e in got if e["event"] == "op"} | {"_folder": clone}

    def ops(self, events: list[dict]) -> dict:
        return {e["fields"]["op"]: e["fields"] for e in events if e["event"] == "op"}

    def check_fields(self, op: dict, counters: bool = True) -> None:
        for key in TIMES + (COUNTERS if counters else ()):
            self.assertIsNotNone(op.get(key), f"{op['op']} has no {key}")

        self.assertIn(op["durability"], writes.LEVELS)

    def check_state(self, folder: Path, name: str) -> None:
        """Every insert the state holds comes back for its own vector, first nearly always, and no id the state
        doesn't hold ever comes back."""
        adapter = self.make()
        ctx = ServeContext(index=folder, dataset=self.ds, rows=self.cfg.id_bound, k=10, metric="cosine", threads=1,
                           variant="", params=self.params)
        handle = adapter.open(ctx, folder)
        live = writes.state_mask(self.cfg, self.stream, name)
        base = np.asarray(self.ds.base, dtype=np.float32)
        inserted = np.flatnonzero(live[self.n :]) + self.n
        dead = np.flatnonzero(~live)
        first = 0

        try:
            for row in inserted.tolist():
                ids = adapter.search(handle, base[row], 10, self.params).tolist()
                self.assertIn(row, ids, f"insert {row} isn't found")
                first += ids[0] == row

            probes = [base[row] for row in dead.tolist()] + list(np.asarray(self.ds.queries, dtype=np.float32))

            for q in probes:
                ids = np.asarray(adapter.search(handle, q, 10, self.params))
                ids = ids[ids >= 0]
                self.assertTrue(np.all(live[ids]), f"{name} returned ids it doesn't hold: {ids[~live[ids]]}")
        finally:
            adapter.close(handle)

        self.assertGreaterEqual(first, 0.95 * len(inserted))

    def test_info_names_the_write_path(self):
        self.assertEqual(self.info["write_role"], "full")
        self.assertIn(self.info["sync_kind"], ("incremental", "whole"))

    def test_stream_rounds_report_every_op(self):
        incremental = self.info["sync_kind"] == "incremental"
        want = {"insert", "delete", "durable_point"} | ({"sync"} if incremental else set())

        for r, events in enumerate(self.rounds, 1):
            names = [e["event"] for e in events]
            self.assertEqual((names[0], names[-1]), ("opened", "done"))
            ops = self.ops(events)
            self.assertEqual(set(ops), want)
            inserts, deletes = self.cfg.rounds[r - 1]
            self.assertEqual((ops["insert"]["rows"], ops["delete"]["rows"]), (inserts, deletes))
            self.assertEqual(ops["insert"]["calls"], -(-inserts // self.cfg.batch))
            self.assertEqual(ops["durable_point"]["calls"], 1)

            for op in ops.values():
                self.check_fields(op)

            self.assertGreater(ops["insert"]["rows_per_s"], 0)
            self.assertGreater(ops["delete"]["rows_per_s"], 0)
            self.assertGreater(ops["insert"]["open_us"], 0)
            self.assertGreater(ops["durable_point"]["extra"]["rows_per_s_durable"], 0)

            if incremental:
                self.assertEqual(ops["sync"]["calls"], ops["insert"]["calls"] + ops["delete"]["calls"])

    def test_the_last_state_holds_its_inserts_and_none_of_its_deletes(self):
        self.check_state(self.work, "s10")

    def test_latency_runs(self):
        incremental = self.info["sync_kind"] == "incremental"

        for source, round_ in ((self.build, 0), (self.work, 2)):
            ops = self.latency(source, round_)
            self.assertEqual((ops["insert"]["calls"], ops["delete"]["calls"]), (6, 6))
            self.assertEqual({ops[k]["batch"] for k in ("insert", "delete")}, {1})

            for key in ("insert", "delete"):
                self.check_fields(ops[key])

            if incremental:
                self.assertEqual((ops["insert_sync"]["calls"], ops["sync"]["calls"]), (4, 4))
                self.check_fields(ops["insert_sync"])
                self.check_fields(ops["sync"], counters=False)
                self.assertNotIn("durable_point", ops)
            else:
                self.assertEqual(ops["durable_point"]["calls"], 2)
                self.check_fields(ops["durable_point"])
                self.assertNotIn("sync", ops)

    def maintained(self, action: str) -> tuple[Path, dict]:
        folder = self.root / f"s10-{action}"
        writes.remove_tree(folder)
        self.scratch(self.work, folder)
        got = self.call("maintain", "--index", str(folder), "--action", action, "--threads", "2")
        self.assertEqual(got[-1]["event"], "maintain")
        fields = got[-1]["fields"]
        self.assertEqual((fields["action"], fields["threads"], fields["rows_live"]), (action, 2, self.n))
        self.assertGreater(fields["seconds"], 0)
        self.assertGreater(fields["peak_footprint"], 0)
        self.assertIn(fields["durability"], writes.LEVELS)
        self.check_state(folder, "s10")
        return folder, fields

    def test_maintenance_actions(self):
        for action in self.info["maintenance_actions"]:
            self.maintained(action)


@unittest.skipUnless(has("qdrant_edge"), "qdrant-edge-py isn't installed")
class QdrantWritesTest(WriteCase):
    module, cls_name = "adapters.qdrant_shard", "QdrantEdge"
    # Three batches of upserts at the build, so its last points carry version 2 and a delete needs the fix-up.
    n, rows = 4500, 4800
    params = {"hnsw_ef": 128, "oversampling": 4}

    def test_the_fixup_runs_once_and_every_delete_applies(self):
        first, second = (self.ops(r)["durable_point"]["extra"]["pending"] for r in self.rounds)
        self.assertEqual((first["fixup_ops"], second["fixup_ops"]), (qdrant_shard.fixup_ops(self.n), 0))
        meta = json.loads((self.work / "meta.json").read_text())
        calls = sum(-(-i // self.cfg.batch) + -(-d // self.cfg.batch) for i, d in self.cfg.rounds)
        self.assertEqual(meta["log_ops"], qdrant_shard.fixup_ops(self.n) + calls)
        self.assertGreater(second["wal_bytes"], 0)
        # The stream deletes points of the build's last batch of upserts, which carry its highest version.
        deleted = np.concatenate([r.deletes for r in self.stream.rounds])
        self.assertTrue(np.any((deleted >= 4000) & (deleted < self.n)))

    def test_the_state_is_sealed_with_its_log(self):
        store = stores.Store(self.work, "shard")
        self.assertEqual(store.restore(), [])
        self.assertTrue(any(rel.startswith("wal/") for rel in store.record["files"]))
        before = identity(self.work)
        self.check_state(self.work, "s10")
        self.assertEqual(identity(self.work), before)
        sealed = [e for e in self.rounds[1] if e["event"] == "done"][0]["sealed"]
        self.assertEqual(sealed["log_ops"], json.loads((self.work / "meta.json").read_text())["log_ops"])

    def test_a_latency_run_on_the_build_applies_its_deletes(self):
        ops = self.latency(self.build, 0)
        adapter = self.make()
        shard = adapter.qe.EdgeShard.load(str(ops["_folder"] / "shard"))

        try:
            plan = writes.latency_plan(self.stream, 0, True, self.cfg.reserve_from, 2, 6, 4, 2)
            self.assertEqual([i for i in plan.deletes.tolist() if shard.retrieve([i], False, False)], [])
            self.assertEqual(len(shard.retrieve(plan.inserts.tolist(), False, False)), len(plan.inserts))
        finally:
            shard.close()

    def test_optimize_is_a_rebuild(self):
        folder, fields = self.maintained("optimize")
        report = fields["extra"]["report"]
        self.assertTrue(report["rebuild"])
        self.assertEqual((report["indexed_after"], report["indexing_threads"]), (self.n, 2))
        self.assertEqual(fields["durability"], "power")
        self.assertEqual(stores.Store(folder, "shard").restore(), [])

    def test_searches_record_the_log(self):
        go = json.dumps({"cmd": "go", "trial": 0, "queries": [1]}) + "\n"
        got = self.call("cold", "--index", str(self.work), "--params", json.dumps(self.params), "--queries", "1",
                        "--regime", "c0", "--cpu-state", "none", stdin=go)
        self.assertEqual([e["event"] for e in got], ["ready", "trial"])
        extra = got[1]["fields"]["extra"]
        self.assertGreater(extra["wal_bytes"], 0)
        self.assertEqual(extra["log_ops"], json.loads((self.work / "meta.json").read_text())["log_ops"])

        with tempfile.TemporaryDirectory() as ids:
            got = self.call("warm", "--index", str(self.work), "--grid", json.dumps([self.params]), "--split",
                            "tune", "--ids-dir", ids, "--passes", "1", "--warmup", "2")

        setting = [e for e in got if e["event"] == "setting"][0]
        self.assertEqual(setting["fields"]["extra"]["wal_bytes"], extra["wal_bytes"])


@unittest.skipUnless(has("astervec"), "aster-vec isn't installed")
class AsterVecWritesTest(WriteCase):
    module, cls_name = "adapters.astervec_lsm", "AsterVec"
    params = {"ef": 128}

    def test_the_durable_point_is_a_close_and_an_open(self):
        for events in self.rounds:
            point = self.ops(events)["durable_point"]
            self.assertEqual(point["durability"], "process")
            self.assertIn("tombstones", point["extra"]["pending"]["delete_stats"])

        self.assertEqual(self.info["durability"], {"call": "memory", "durable_point": "process"})
        self.assertEqual(self.info["maintenance_actions"], [])

    def test_the_state_is_sealed(self):
        store = stores.Store(self.work, "store")
        self.assertEqual(store.restore(), [])
        self.assertIn("vectors.bin", store.record["files"])
        before = identity(self.work)
        self.check_state(self.work, "s10")
        self.assertEqual(identity(self.work), before)

    def test_a_held_store_fails_a_write_session(self):
        folder = self.root / "held"
        writes.remove_tree(folder)
        self.scratch(self.work, folder)
        store = stores.Store(folder, "store")
        self.assertTrue(store.claim())

        try:
            with self.assertRaises(AssertionError) as caught:
                self.write(folder, "--mode", "stream", "--round", "1", "--from-round", "0")

            self.assertIn("never opens a private copy", str(caught.exception))
        finally:
            store.release()


@unittest.skipUnless(has("sqlite_vec"), "sqlite-vec isn't installed")
class SqliteVecWritesTest(WriteCase):
    module, cls_name = "adapters.sqlite_vec", "SqliteVec"

    def mode(self, folder: Path) -> str:
        db = sqlite3.connect(f"file:{folder / sqlite_vec.DB}?mode=ro", uri=True)

        try:
            return db.execute("pragma journal_mode").fetchone()[0]
        finally:
            db.close()

    def test_each_state_opens_as_the_build_does(self):
        self.assertEqual(sorted(p.name for p in self.work.iterdir()), ["meta.json", sqlite_vec.DB])
        self.assertEqual(self.mode(self.work), "delete")
        meta = json.loads((self.work / "meta.json").read_text())
        self.assertEqual(meta["live_rows"], self.n)
        # vec0 reuses a free slot only in its last chunk, so the slots lie between the live rows and every row made.
        self.assertGreater(meta["slots"], self.n)
        self.assertLessEqual(meta["slots"], self.n + 200)
        sealed = [e for e in self.rounds[1] if e["event"] == "done"][0]["sealed"]
        self.assertEqual(sealed, {"slots": meta["slots"], "live_rows": self.n})

    def test_durable_inserts_commit_with_fullfsync(self):
        ops = self.latency(self.build, 0)
        # The sync after a commit that synced the log has nothing left to do.
        self.assertLess(ops["sync"]["p50_us"], 0.5 * ops["insert_sync"]["p50_us"])
        self.assertEqual(ops["insert_sync"]["durability"], "power")
        self.assertEqual(ops["insert"]["durability"], "process")
        # A session without a durable point still ends in the build's journal mode.
        self.assertEqual(self.mode(ops["_folder"]), "delete")
        self.assertFalse((ops["_folder"] / f"{sqlite_vec.DB}-wal").exists())

    def test_vacuum_leaves_no_log(self):
        folder, fields = self.maintained("vacuum")
        self.assertEqual(fields["durability"], "power")
        self.assertEqual(self.mode(folder), "delete")
        self.assertFalse((folder / f"{sqlite_vec.DB}-wal").exists())


@unittest.skipUnless(has("lancedb"), "lancedb isn't installed")
class LanceDBWritesTest(WriteCase):
    module, cls_name = "adapters.lancedb", "LanceDB"
    config = {"index": "ivf_rq", "bits": 1, "partitions": 8}
    params = {"nprobes": 8, "refine_factor": 50}

    def test_the_harness_syncs_each_commits_files(self):
        for r, events in enumerate(self.rounds, 1):
            ops = self.ops(events)
            pending = ops["durable_point"]["extra"]["pending"]
            sync = pending["harness_sync"]
            self.assertEqual(sync["syncs"], ops["sync"]["calls"])
            # Each insert makes a data file, a manifest and a transaction, and each delete a deletion file too.
            self.assertGreaterEqual(sync["files"], 3 * ops["insert"]["calls"] + 3 * ops["delete"]["calls"])
            self.assertEqual(pending["rows"], self.n)
            self.assertEqual(ops["sync"]["durability"], "power")

        self.assertGreater(pending["unindexed_rows"], 0)

    def test_optimize_indexes_the_new_rows(self):
        _, fields = self.maintained("optimize")
        report = fields["extra"]["report"]
        self.assertEqual(report["after"]["unindexed_rows"], 0)
        self.assertGreater(report["before"]["unindexed_rows"], 0)
        self.assertEqual(report["calls"], ["Table.optimize()"])
        self.assertEqual(fields["durability"], "process")

    @unittest.skipUnless(has("lance"), "retrain needs pylance")
    def test_retrain_trains_a_new_index(self):
        _, fields = self.maintained("retrain")
        report = fields["extra"]["report"]
        self.assertGreater(report["retrain_seconds"], 0)
        self.assertEqual(report["after"]["unindexed_rows"], 0)
        self.assertEqual(len(report["calls"]), 2)


if __name__ == "__main__":
    unittest.main()
