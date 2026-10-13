import argparse
import fcntl
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests import helpers
from adapters.base import Adapter, WriteContext
from adapters.inproc import LiveRows
import data
import writes

WRITES = {
    "stream": "/tmp/unused.stream", "batch": 100, "latency_states": ["s0", "s10"], "latency_calls": 20,
    "warmup_calls": 5, "durable_calls": 10, "saves": 2, "seed": 1, "probes": 40,
    "states": {
        "s1": {"inserts": 50, "deletes": 50}, "s5": {"inserts": 250, "deletes": 250},
        "s10": {"inserts": 500, "deletes": 500}, "s1i": {"inserts": 50, "deletes": 0},
        "s5i": {"inserts": 250, "deletes": 0}, "s10i": {"inserts": 500, "deletes": 0},
        "s10p": {"inserts": 500, "deletes": 500, "policy": "auto"},
    },
}


def config(n: int = 5000, **over) -> writes.WritesConfig:
    return writes.parse_writes(WRITES | over, n, "tiny-16-cosine")


def first_block(path: Path) -> int | None:
    """Where the file's first byte sits on the disk, which a clone shares and a plain copy doesn't, or None off
    macOS."""
    if sys.platform != "darwin":
        return None

    fd = os.open(path, os.O_RDONLY)

    try:
        asked = struct.pack("=Iqq", 0, os.fstat(fd).st_size, 0)
        return struct.unpack("=Iqq", fcntl.fcntl(fd, writes.F_LOG2PHYS_EXT, asked))[2]
    finally:
        os.close(fd)


class Recorder(Adapter):
    """A writer that only records the calls a session makes, in order."""

    write_role, sync_kind = "full", "incremental"

    def writer_open(self, ctx: WriteContext, path: Path) -> list:
        return []

    def prepare_writes(self, w: list) -> None:
        w.append("prepare")

    def insert(self, w: list, ids: np.ndarray, rows: np.ndarray) -> None:
        w.append("insert")

    def delete(self, w: list, ids: np.ndarray) -> None:
        w.append("delete")

    def sync(self, w: list) -> dict:
        w.append("sync")
        return {"call": "sync", "calls": len(w)}


class RngTest(unittest.TestCase):
    def test_it_is_splitmix64(self):
        rng = writes.Rng(0)
        self.assertEqual((rng.next_u64(), rng.next_u64()), (0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4))

    def test_below_is_in_range_and_even(self):
        rng = writes.Rng(1)
        counts = np.bincount([rng.below(7) for _ in range(7000)], minlength=7)
        self.assertEqual(len(counts), 7)
        self.assertTrue(np.all(np.abs(counts - 1000) < 150), counts)

        with self.assertRaises(ValueError):
            rng.below(0)

    def test_draw_takes_different_items_and_leaves_its_input(self):
        items = np.arange(100, 200)
        picked = writes.draw(writes.Rng(3), items, 30)
        self.assertEqual(len(set(picked.tolist())), 30)
        self.assertTrue(np.all((picked >= 100) & (picked < 200)))
        self.assertTrue(np.array_equal(items, np.arange(100, 200)))
        self.assertTrue(np.array_equal(picked, writes.draw(writes.Rng(3), items, 30)))

        with self.assertRaises(writes.StreamError):
            writes.draw(writes.Rng(3), items, 101)


class ConfigTest(unittest.TestCase):
    def test_rounds_and_states(self):
        cfg = config()
        self.assertEqual(cfg.rounds, [(50, 50), (200, 200), (250, 250)])
        self.assertEqual((cfg.id_bound, cfg.reserve_from, cfg.reserved, cfg.latency_seed), (5500, 5500, 35, 2))
        self.assertEqual([s.name for s in cfg.ladder("joint")], ["s1", "s5", "s10"])
        self.assertEqual([s.name for s in cfg.ladder("inserts")], ["s1i", "s5i", "s10i"])
        self.assertEqual([s.name for s in cfg.ladder("policy")], ["s10p"])
        self.assertEqual((cfg.spec("s5i").round, cfg.spec("s5i").pass_name, cfg.spec("s0").pass_name),
                         (2, "inserts", ""))
        self.assertEqual((cfg.live_count("s10"), cfg.live_count("s10i"), cfg.live_count("s0")), (5000, 5500, 5000))

    def test_policy_and_maintained_states_take_the_plain_states_truth(self):
        cfg = config()
        self.assertEqual(cfg.truth_state("s10p"), "s10")
        self.assertEqual(cfg.truth_state("s10-fold"), "s10")
        spec = cfg.spec("s10-fold")
        self.assertEqual((spec.origin, spec.action, spec.round, spec.pass_name), ("s10", "fold", 3, ""))
        self.assertEqual(cfg.truth_path("s10p").name, "unused-s10.gt.ibin")
        self.assertEqual(cfg.copy_base("s10").name, "unused-s10-live.base.fbin")
        self.assertEqual(writes.truth_states(cfg), ["s0", "s1", "s5", "s10", "s1i", "s5i", "s10i"])
        self.assertEqual(writes.gt_states(cfg, ["s0", "s1", "s10i"]), "s0=0,s1=1,s10i=3i")

        with self.assertRaises(writes.StreamError):
            cfg.spec("s7")

    def test_bad_tables_fail(self):
        bad = [
            {"surprise": 1},
            {"inserts_from": 4000},
            {"reserve_from": 5100},
            {"states": {"s1": {"inserts": 50, "deletes": 50}, "s2i": {"inserts": 60, "deletes": 0}}},
            {"states": {"s1": {"inserts": 50, "deletes": 50}, "s1p": {"inserts": 50, "deletes": 10, "policy": "a"}}},
            {"states": {"s1": {"inserts": 50, "deletes": 50}, "s2": {"inserts": 40, "deletes": 60}}},
            {"states": {"S1": {"inserts": 50}}},
            {"states": {"s1": {"inserts": 50, "deletes": 50, "when": 1}}},
            {"latency_states": ["s9"]},
        ]

        for over in bad:
            with self.assertRaises(writes.StreamError, msg=str(over)):
                config(**over)

    def test_an_inserts_only_suite_takes_its_rounds_from_its_states(self):
        cfg = config(states={"s1i": {"inserts": 50}, "s2i": {"inserts": 80}}, latency_states=[])
        self.assertEqual(cfg.rounds, [(50, 0), (30, 0)])
        self.assertEqual(cfg.final("joint"), None)
        self.assertEqual(cfg.final("inserts").name, "s2i")


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = config(stream=str(self.root / "w.stream"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_stream_file_round_trips(self):
        stream = writes.make_stream(self.cfg)
        stream.rounds[0].upserts = np.array([[3, 5600], [7, 5601]])
        writes.write_stream(self.cfg.stream, stream)
        back = writes.read_stream(self.cfg.stream)
        self.assertTrue(writes.same_stream(stream, back))
        self.assertEqual(back.rounds[0].upserts.tolist(), [[3, 5600], [7, 5601]])
        raw = self.cfg.stream.read_bytes()
        self.assertEqual(raw[:8], b"COLDSTRM")
        self.assertEqual(struct.unpack_from("<III", raw, 8), (1, 4, 5000))

        # A file cut short, one with bytes past its last round and one of another kind all fail.
        for broken in (raw[:-3], raw + b"\0", b"NOTSTRM!" + raw[8:]):
            self.cfg.stream.write_bytes(broken)

            with self.assertRaises(writes.StreamError):
                writes.read_stream(self.cfg.stream)

    def test_the_stream_inserts_in_row_order_and_deletes_live_vectors(self):
        stream = writes.make_stream(self.cfg)
        stream.check()
        self.assertTrue(np.array_equal(stream.base, np.arange(5000)))
        self.assertEqual([len(r.inserts) for r in stream.rounds], [50, 200, 250])
        self.assertTrue(np.array_equal(np.concatenate([r.inserts for r in stream.rounds]), np.arange(5000, 5500)))
        live = set(range(5000))
        hit = 0

        for r in stream.rounds:
            dels = r.deletes.tolist()
            self.assertEqual(len(set(dels)), len(dels))
            self.assertTrue(set(dels) <= live, "a round deletes only vectors live at its start")
            hit += sum(1 for d in dels if d >= 5000)
            live = (live - set(dels)) | set(r.inserts.tolist())

        # Later rounds delete some inserts of the rounds before them, at about the share of them that is live.
        self.assertGreater(hit, 0)
        self.assertTrue(writes.same_stream(stream, writes.make_stream(self.cfg)))
        self.assertFalse(writes.same_stream(stream, writes.make_stream(config(seed=2))))

    def test_ensure_stream_makes_the_file_once_and_refuses_another(self):
        made = writes.ensure_stream(self.cfg)
        self.assertTrue(writes.same_stream(made, writes.ensure_stream(self.cfg)))
        writes.write_stream(self.cfg.stream, writes.make_stream(config(seed=9)))

        with self.assertRaises(writes.StreamError):
            writes.ensure_stream(self.cfg)

        with self.assertRaises(writes.StreamError):
            writes.make_stream(self.cfg, SimpleNamespace(count=5520))

        # The record beside the file names what made it, and a suite of another recipe can't take it.
        writes.stream_record(self.cfg).unlink()
        self.cfg.stream.unlink()
        writes.ensure_stream(self.cfg)
        self.assertEqual(writes.read_json(writes.stream_record(self.cfg))["recipe"]["rounds"],
                         [[50, 50], [200, 200], [250, 250]])

        with self.assertRaises(writes.StreamError):
            writes.ensure_stream(config(stream=str(self.cfg.stream), seed=4))

    def test_states_of_all_three_passes(self):
        stream = writes.make_stream(self.cfg)
        s10 = writes.state_mask(self.cfg, stream, "s10")
        s10i = writes.state_mask(self.cfg, stream, "s10i")
        self.assertEqual((int(s10.sum()), int(s10i.sum()), len(s10)), (5000, 5500, 5500))
        self.assertTrue(np.all(s10i))
        self.assertTrue(np.array_equal(writes.state_mask(self.cfg, stream, "s10p"), s10))
        self.assertTrue(np.array_equal(writes.state_mask(self.cfg, stream, "s10-fold"), s10))
        s1 = writes.state_mask(self.cfg, stream, "s1")
        self.assertEqual((int(s1[:5000].sum()), int(s1[5000:].sum())), (4950, 50))

        joint = writes.stream_calls(stream, "joint", 1, 3, 100)
        self.assertEqual([[c.op for c in calls][:4] for calls in joint], [["insert", "delete"] * 2] * 2)
        self.assertEqual([sum(len(c.ids) for c in calls) for calls in joint], [400, 500])
        self.assertTrue(all(len(c.ids) <= 100 for calls in joint for c in calls))
        inserts = writes.stream_calls(stream, "inserts", 0, 3, 100)
        self.assertEqual({c.op for calls in inserts for c in calls}, {"insert"})
        self.assertEqual([len(calls) for calls in inserts], [1, 2, 3])
        policy = writes.stream_calls(stream, "policy", 0, 3, 100)
        joint = writes.stream_calls(stream, "joint", 0, 3, 100)
        self.assertEqual([[(c.op, c.ids.tolist()) for c in r] for r in policy],
                         [[(c.op, c.ids.tolist()) for c in r] for r in joint])

        # Replaying the calls in their order gives the state, whatever the order within a round.
        mask = writes.live_mask(stream, 0)

        for calls in joint:
            for c in reversed(calls):
                mask[c.ids] = c.op == "insert"

        self.assertTrue(np.array_equal(mask, s10))

        with self.assertRaises(writes.StreamError):
            writes.stream_calls(stream, "joint", 2, 2, 100)

    def test_latency_plans_and_probes(self):
        stream = writes.make_stream(self.cfg)
        plan = writes.latency_plan(stream, 3, True, 5500, 5, 20, 10, 2)
        self.assertTrue(np.array_equal(plan.inserts, np.arange(5500, 5535)))
        self.assertEqual(len(set(plan.deletes.tolist())), 25)
        live = writes.state_mask(self.cfg, stream, "s10")
        self.assertTrue(np.all(live[plan.deletes]) and np.all(plan.deletes < 5000))
        same = writes.latency_plan(stream, 3, True, 5500, 5, 20, 10, 2)
        self.assertTrue(np.array_equal(plan.deletes, same.deletes))
        self.assertTrue(np.all(writes.latency_plan(stream, 0, True, 5500, 5, 20, 10, 2).deletes < 5000))

        probes = writes.probe_rows(self.cfg, stream, "s10")
        self.assertEqual(len(probes), 80)
        self.assertTrue(np.all(probes[:40] >= 5000) and np.all(probes[40:] < 5000) and np.all(live[probes]))
        self.assertEqual(len(writes.probe_rows(self.cfg, stream, "s1", count=60)), 50 + 60)


class DriftTest(unittest.TestCase):
    """A drift stream on a dataset of clustered rows, whose inserts are the rows nearest a few topics."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(5)
        centers = rng.standard_normal((12, 16))
        rows = centers[rng.integers(0, 12, 9000)] + 0.3 * rng.standard_normal((9000, 16))
        rows = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)
        queries = rows[rng.integers(0, 9000, 10)]
        (self.root / "data").mkdir()
        data.write_fbin(self.root / "data" / "tiny-16-cosine.base.fbin", rows)
        data.write_fbin(self.root / "data" / "tiny-16-cosine.query.fbin", queries)
        self.ds = data.Dataset("tiny-16-cosine", self.root / "data", self.root / "derived")
        self.cfg = config(stream=str(self.root / "d.stream"), pattern="drift", topics=12, hot_topics=2,
                          topic_sample=2000, kmeans_rounds=8)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_inserts_are_the_candidates_nearest_the_topics(self):
        inserts = writes.drift_inserts(self.cfg, self.ds)
        self.assertEqual(self.cfg.drift_from, 5535)
        self.assertEqual((len(inserts), len(set(inserts.tolist()))), (500, 500))
        self.assertTrue(np.all(inserts >= 5535))
        self.assertTrue(np.array_equal(inserts, writes.drift_inserts(self.cfg, self.ds)))
        # The rows taken are at least as near a topic as every row left out, which shows the scan keeps the nearest.
        rng = writes.Rng(self.cfg.seed)
        candidates = np.arange(5535, 9000)
        sample = np.sort(writes.draw(rng, candidates, 2000))
        centers = writes.kmeans(self.ds.base[sample], 12, 8, rng, "cosine")
        topics = centers[np.sort(writes.draw(rng, np.arange(12), 2))]
        x = np.asarray(self.ds.base[candidates], dtype=np.float64)
        d = (2 - 2 * x @ topics.T.astype(np.float64)).min(axis=1)
        taken = np.isin(candidates, inserts)
        self.assertLessEqual(d[taken].max(), d[~taken].min() + 1e-9)

    def test_ids_map_to_rows_through_the_stream(self):
        stream = writes.ensure_stream(self.cfg, self.ds)
        stream.check()
        self.assertFalse(stream.identity)
        self.assertEqual(stream.id_bound, 5500)
        rows = stream.rounds[1].inserts[:3]
        self.assertTrue(np.array_equal(stream.ids(rows), np.arange(5050, 5053)))
        self.assertTrue(np.array_equal(stream.row_of[stream.ids(rows)], rows))
        self.assertTrue(np.array_equal(stream.ids(np.array([7, 4999])), [7, 4999]))

        with self.assertRaises(writes.StreamError):
            stream.ids(np.array([5600]))

        calls = writes.stream_calls(stream, "joint", 0, 3, 100)
        ins = np.concatenate([c.ids for r in calls for c in r if c.op == "insert"])
        self.assertTrue(np.array_equal(ins, np.arange(5000, 5500)))
        dels = [c for r in calls for c in r if c.op == "delete"]
        self.assertTrue(all(np.array_equal(stream.row_of[c.ids], c.rows) for c in dels))
        mask = writes.state_mask(self.cfg, stream, "s10")
        self.assertEqual((len(mask), int(mask.sum())), (stream.row_bound, 5000))
        self.assertTrue(np.all(mask[stream.rounds[2].inserts]))

        # A drift stream without its record can't be checked, so it is refused, and a record of another recipe too.
        writes.stream_record(self.cfg).unlink()

        with self.assertRaises(writes.StreamError):
            writes.ensure_stream(self.cfg, self.ds)

    def test_a_drift_state_scores_ids_through_their_rows(self):
        stream = writes.ensure_stream(self.cfg, self.ds)
        mask = writes.state_mask(self.cfg, stream, "s10")
        ids, dists = writes.exact_truth(self.ds, mask, k=20)
        truth = data.Truth(self.root / "t.gt.ibin", ids, 9000, dists)
        recall = writes.StateRecall(self.ds, truth, mask, 5000, rows=stream.row_of)
        found = stream.ids(np.asarray(ids)[:, :10].ravel()).reshape(-1, 10)
        out = recall.summary(np.arange(10), found)
        self.assertEqual((out["recall10"], out["dead_returned"]), (1.0, 0.0))
        # The rows themselves aren't ids, so they score as misses, or as dead where they fall on a deleted id.
        self.assertLess(recall.summary(np.arange(10), np.asarray(ids)[:, :10])["recall10"], 1.0)
        probes = writes.probe_rows(self.cfg, stream, "s10", count=5)
        hits = writes.findability(self.ds, probes, stream.ids(probes)[:, None], 5000, stream.row_of)
        self.assertEqual((hits["inserted"], hits["base"]), (1.0, 1.0))


class TruthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=5600, nq=12)
        self.cfg = config(stream=str(self.root / "writes" / "w.stream"))
        self.stream = writes.ensure_stream(self.cfg, self.ds)

    def tearDown(self):
        self.tmp.cleanup()

    def test_numpy_truth_matches_a_full_sort_over_the_live_vectors(self):
        mask = writes.state_mask(self.cfg, self.stream, "s5")
        ids, dists = writes.exact_truth(self.ds, mask, k=100)
        live = np.flatnonzero(mask)
        want, want_d = helpers.exact(np.asarray(self.ds.base)[live], np.asarray(self.ds.queries), 100)
        self.assertTrue(np.array_equal(ids, live[want]))
        self.assertTrue(np.allclose(dists, want_d, atol=1e-6))

    def test_each_state_gets_its_truth_file(self):
        paths = writes.compute_truths(self.cfg, self.ds, "numpy", log=lambda m: None)
        self.assertEqual([p.name for p in paths], [f"w-{s}.gt.ibin" for s in writes.truth_states(self.cfg)])
        s10 = self.ds.truth(5500, self.cfg.truth_path("s10"))
        self.assertTrue(s10.has_dist)
        self.assertTrue(np.all(writes.state_mask(self.cfg, self.stream, "s10")[np.asarray(s10.ids)]))
        s0 = data.ibin(self.cfg.truth_path("s0"))
        want, _ = helpers.exact(np.asarray(self.ds.base)[:5000], np.asarray(self.ds.queries), 100)
        self.assertTrue(np.array_equal(s0, want))

    def test_the_rust_engine_runs_gt_with_the_states(self):
        calls = []

        with mock.patch.object(subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw))):
            writes.compute_truths(self.cfg, self.ds, "rust", threads=3, names=["s1", "s10i"], log=lambda m: None)

        cmd, kw = calls[0]
        self.assertEqual(cmd[cmd.index("--states") + 1], "s1=1,s10i=3i")
        self.assertEqual(cmd[cmd.index("--out-prefix") + 1], str(self.root / "writes" / "w"))
        self.assertEqual(kw["env"]["COLDIVF_DATA"], str(self.ds.root))

    def test_the_copy_of_a_state_holds_its_live_rows_and_truth_by_position(self):
        writes.compute_truths(self.cfg, self.ds, "numpy", names=["s10"], log=lambda m: None)
        base = writes.write_copy(self.cfg, self.ds, "s10")
        copy = data.Dataset.from_base(base)
        rows = writes.copy_rows(base)
        live = np.flatnonzero(writes.state_mask(self.cfg, self.stream, "s10"))
        self.assertTrue(np.array_equal(rows, live))
        self.assertTrue(np.array_equal(np.asarray(copy.base), np.asarray(self.ds.base)[live]))
        self.assertTrue(np.array_equal(np.asarray(copy.queries), np.asarray(self.ds.queries)))
        truth = copy.truth(copy.count)
        self.assertTrue(np.array_equal(rows[np.asarray(truth.ids)], np.asarray(data.ibin(self.cfg.truth_path("s10")))))
        self.assertEqual(writes.copy_record(base)["state"], "s10")
        # The copy's own truth scores a search of it as a plain head-to-head suite would.
        self.assertEqual(data.Recall(copy, truth).summary(np.arange(12), np.asarray(truth.ids)[:, :10])["recall10"],
                         1.0)


class StateRecallTest(unittest.TestCase):
    """Scoring with live ids, on a base whose last row duplicates query 3's 10th neighbor."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=600, nq=8, tie_query=None)
        self.live = np.ones(600, dtype=bool)
        self.live[500:] = False
        self.live[[3, 4]] = False
        self.live[[520, 530]] = True
        ids, dists = writes.exact_truth(self.ds, self.live)
        data.write_ibin(self.root / "t.gt.ibin", ids)
        data.write_fbin(self.root / "t.gt.dist.fbin", dists)
        self.truth = self.ds.truth(600, self.root / "t.gt.ibin")
        self.recall = writes.StateRecall(self.ds, self.truth, self.live, 500)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_truth_scores_one(self):
        found = np.asarray(self.truth.ids)[:, :10]
        out = self.recall.summary(np.arange(8), found)
        self.assertEqual((out["recall10"], out["dead_returned"]), (1.0, 0.0))

    def test_a_dead_id_is_a_miss_even_at_the_same_distance(self):
        q = 0
        true = np.asarray(self.truth.ids[q, :10]).tolist()
        # Row 3 is deleted. Give it the vector of the true 10th neighbor, so a tie check alone would count it.
        base = np.array(self.ds.base)
        base[3] = base[true[9]]
        data.write_fbin(self.ds.base_path, base)
        ds = data.Dataset(self.ds.name, self.ds.root, self.ds.derived)
        recall = writes.StateRecall(ds, ds.truth(600, self.root / "t.gt.ibin"), self.live, 500)
        found = np.array(true[:9] + [3])
        self.assertEqual(recall.hits(q, found, 10), (9, 9))
        self.assertEqual(recall.dead(found), 1)
        self.assertEqual(recall.trial(q, found)["dead_returned"], 1.0)
        # A live duplicate at that distance still ties.
        plain = data.Recall(ds, ds.truth(600, self.root / "t.gt.ibin"))
        self.assertEqual(plain.hits(q, found, 10)[0], 10)

    def test_ids_past_the_base_count_and_inserts_found(self):
        q = 1
        true = np.asarray(self.truth.ids[q, :10]).tolist()
        # Ids past the truth's row count are kept, and a live insert counts as any other id.
        found = np.array([599] + true[:9])
        self.assertEqual(self.recall.dead(found), 1)
        self.assertEqual(self.recall.hits(q, found, 10), (9, 9))
        inserted = [i for i in np.asarray(self.truth.ids[:, :10]).ravel().tolist() if i >= 500]
        out = self.recall.summary(np.arange(8), np.asarray(self.truth.ids)[:, :10])
        self.assertEqual(out["recall10_inserted"], 1.0 if inserted else None)
        nothing = self.recall.summary(np.arange(8), np.full((8, 10), -1))
        self.assertEqual(nothing["recall10_inserted"], 0.0 if inserted else None)

    def test_positions_of_a_copy_become_rows(self):
        rows = np.flatnonzero(self.live)
        recall = writes.StateRecall(self.ds, self.truth, self.live, 500, rows=rows)
        found = np.searchsorted(rows, np.asarray(self.truth.ids)[:, :10])
        self.assertEqual(recall.summary(np.arange(8), found)["recall10"], 1.0)
        self.assertEqual(recall.clean(np.array([0, len(rows) + 5, -1, 1]), 10), [int(rows[0]), int(rows[1])])

    def test_findability(self):
        probes = np.array([520, 530, 7, 8])
        found = np.array([[520, 1, 2], [9, 530, 3], [7, 6, 5], [-1, -1, -1]])
        out = writes.findability(self.ds, probes, found, 500)
        self.assertEqual((out["inserted"], out["base"]), (0.5, 0.5))
        self.assertEqual(out["missed"], [530, 8])
        # A row behind only rows at the same distance, eg. its exact duplicate, still counts.
        base = np.array(self.ds.base)
        base[9] = base[530]
        data.write_fbin(self.ds.base_path, base)
        ds = data.Dataset(self.ds.name, self.ds.root, self.ds.derived)
        self.assertEqual(writes.findability(ds, probes, found, 500)["inserted"], 1.0)


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_sit_beside_the_state(self):
        folder = self.root / "fake" / "flat-s5"
        self.assertEqual(writes.record_path(folder), self.root / "fake" / "flat-s5.state.json")
        self.assertEqual(writes.latency_path(folder), self.root / "fake" / "flat-s5.latency.json")
        writes.write_json(writes.record_path(folder), {"state": "s5", "rows": np.int64(3)})
        self.assertEqual(writes.read_json(writes.record_path(folder)), {"state": "s5", "rows": 3})
        writes.record_path(folder).write_text('{"state": "s')
        self.assertIsNone(writes.read_json(writes.record_path(folder)))
        self.assertIsNone(writes.read_json(self.root / "missing.json"))

    def test_a_clone_is_a_copy_that_writes_dont_share(self):
        src = self.root / "a"
        (src / "sub").mkdir(parents=True)
        (src / "sub" / "f").write_bytes(b"one")
        writes.clone_tree(src, self.root / "b")
        (self.root / "b" / "sub" / "f").write_bytes(b"two")
        self.assertEqual((src / "sub" / "f").read_bytes(), b"one")
        self.assertEqual(writes.folder_bytes(self.root / "b"), 3)

        with self.assertRaises(FileExistsError):
            writes.clone_tree(src, self.root / "b")

        writes.remove_tree(self.root / "b")
        self.assertFalse((self.root / "b").exists())

    def test_a_plain_copy_has_blocks_of_its_own_and_keeps_holes(self):
        src = self.root / "a"
        (src / "sub").mkdir(parents=True)
        (src / "sub" / "f").write_bytes(np.random.default_rng(1).bytes(1 << 20))

        # A file of 32 MiB that holds 3 MiB, as qdrant makes its log segments and vector chunks.
        with open(src / "sparse", "wb") as f:
            f.truncate(32 << 20)
            f.seek(8 << 20)
            f.write(np.random.default_rng(3).bytes(3 << 20))

        os.utime(src / "sub" / "f", ns=(1_700_000_000_123_456_789, 1_700_000_000_987_654_321))
        writes.copy_tree(src, self.root / "b")
        writes.clone_tree(src, self.root / "c")
        self.assertEqual((self.root / "b" / "sub" / "f").read_bytes(), (src / "sub" / "f").read_bytes())
        self.assertEqual((self.root / "b" / "sparse").read_bytes(), (src / "sparse").read_bytes())
        self.assertLess((self.root / "b" / "sparse").stat().st_blocks * 512, 16 << 20)
        # A copy keeps each file's change time to the nanosecond, as a clone does, which a sealed store checks.
        for name in ("b", "c"):
            self.assertEqual((self.root / name / "sub" / "f").stat().st_mtime_ns, 1_700_000_000_987_654_321)

        if first_block(src / "sub" / "f") is not None:
            self.assertNotEqual(first_block(self.root / "b" / "sub" / "f"), first_block(src / "sub" / "f"))
            self.assertEqual(first_block(self.root / "c" / "sub" / "f"), first_block(src / "sub" / "f"))
            # The data of the file with holes goes in one piece, where macOS's cp takes ~24 even on a quiet disk.
            self.assertLessEqual(writes.file_extents(self.root / "b" / "sparse"), 2)

        with self.assertRaises(FileExistsError):
            writes.copy_tree(src, self.root / "b")

    def test_extents_count_the_runs_of_disk_blocks(self):
        path = self.root / "f"
        path.write_bytes(np.random.default_rng(2).bytes(4 << 20))

        if writes.extents(path) is None:
            self.skipTest("this system has no F_LOG2PHYS_EXT")

        whole = writes.extents(path)
        self.assertEqual((whole["files"], whole["most"]), (1, "f"))
        self.assertLessEqual(whole["total"], 4)
        # Writes in place to a clone move each block they write, which a plain copy never does.
        writes.clone_tree(path, self.root / "clone")
        writes.copy_tree(path, self.root / "copy")

        for name in ("clone", "copy"):
            with open(self.root / name, "r+b") as f:
                for block in range(0, 1024, 16):
                    f.seek(block * 4096)
                    f.write(b"y" * 4096)

                f.flush()
                os.fsync(f.fileno())

        self.assertGreater(writes.extents(self.root / "clone")["total"], 30)
        self.assertLessEqual(writes.extents(self.root / "copy")["total"], 4)
        folder = writes.extents(self.root)
        self.assertEqual((folder["files"], folder["most"]), (3, "clone"))

    def test_live_rows(self):
        folder = self.root / "idx"
        folder.mkdir()
        (folder / "meta.json").write_text(json.dumps({"rows": 4}))
        rows = LiveRows(folder, 4, 3)
        self.assertEqual(rows.insert(np.arange(6, dtype=np.float32).reshape(2, 3)).tolist(), [4, 5])
        rows.delete(np.array([1, 5]))
        self.assertEqual(rows.sync()["call"], "F_FULLFSYNC")
        rows.write_meta()
        rows.close()

        with self.assertRaises(ValueError):
            LiveRows(folder, 4, 3).delete(np.array([7]))

        # A width other than the build's is refused.
        with self.assertRaises(ValueError):
            LiveRows(folder, 4, 5)

        inserts, live = LiveRows.view(folder)
        self.assertEqual(inserts.tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(live.tolist(), [True, False, True, True, True, False])
        self.assertEqual(json.loads((folder / "meta.json").read_text()), {"rows": 4, "state_rows": 6, "live_rows": 4})

        # Rows written past the header's count, as a kill between them leaves, stay out of the view.
        with open(folder / "inserts.fbin", "ab") as f:
            f.write(np.zeros(3, dtype=np.float32).tobytes())

        self.assertEqual(len(LiveRows.view(folder)[0]), 2)
        self.assertEqual(LiveRows.view(self.root), (None, None))


class SessionTest(unittest.TestCase):
    """Runs the test adapter's write and maintain subcommands as the driver does."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=5600, nq=6)
        self.cfg = config(stream=str(self.root / "w.stream"))
        writes.ensure_stream(self.cfg, self.ds)
        self.index = self.root / "idx"
        self.call("build", "--out", str(self.index), "--rows", "5000")

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, *args: str, variant: str = "", rows: str | None = None) -> list[dict]:
        cmd = [helpers.python(), "-m", "tests.fake_adapter", *args, "--data", str(self.ds.base_path), "--variant",
               variant]
        cmd += [] if "--rows" in args else ["--rows", rows or "5000"]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=helpers.PY, env=helpers.env(), timeout=300)
        events = [json.loads(x) for x in out.stdout.splitlines() if x.strip()]
        self.assertTrue(events, out.stderr[-2000:])
        return events

    def write(self, *args: str, variant: str = "") -> list[dict]:
        return self.call("write", "--index", str(self.index), "--stream", str(self.cfg.stream), *args,
                         variant=variant)

    def test_a_stream_session_makes_its_round(self):
        got = self.write("--mode", "stream", "--pass", "joint", "--round", "2", "--batch", "100")
        names = [e["event"] for e in got]
        self.assertEqual((names[0], names[-1]), ("opened", "done"))
        ops = {e["fields"]["op"]: e["fields"] for e in got if e["event"] == "op"}
        self.assertEqual(sorted(ops), ["delete", "durable_point", "insert", "sync"])
        self.assertEqual((ops["insert"]["calls"], ops["insert"]["rows"], ops["delete"]["rows"]), (3, 250, 250))
        self.assertEqual((ops["sync"]["calls"], ops["sync"]["durability"], ops["insert"]["durability"]),
                         (6, "power", "process"))
        self.assertGreater(ops["insert"]["rows_per_s"], 0)
        self.assertIsNone(ops["sync"]["rows_per_s"])
        self.assertGreater(ops["insert"]["open_us"], 0)
        self.assertNotIn("open_us", ops["delete"])
        self.assertGreater(ops["durable_point"]["index_bytes_after"], ops["insert"]["index_bytes_before"])
        self.assertGreater(ops["durable_point"]["extra"]["rows_per_s_durable"], 0)
        # Each op keeps what its last call returned, eg. the files a sync synced.
        self.assertEqual(ops["sync"]["extra"]["returned"], {"files": [LiveRows.INSERTS, LiveRows.LIVE],
                                                            "call": "F_FULLFSYNC"})
        self.assertTrue(ops["durable_point"]["extra"]["returned"]["meta"])
        self.assertNotIn("returned", ops["insert"]["extra"])
        _, live = LiveRows.view(self.index)
        self.assertTrue(np.array_equal(live, writes.state_mask(self.cfg, writes.read_stream(self.cfg.stream),
                                                               "s5")[:5250]))

    def test_a_latency_run_times_single_writes_and_durable_inserts(self):
        got = self.write("--mode", "latency", "--pass", "joint", "--round", "0", "--calls", "20", "--warmup", "5",
                         "--durable-calls", "10", "--reserve", "5500", "--seed", "2")
        ops = {e["fields"]["op"]: e["fields"] for e in got if e["event"] == "op"}
        self.assertEqual(sorted(ops), ["delete", "insert", "insert_sync", "sync"])
        self.assertEqual([ops[k]["calls"] for k in ("insert", "delete", "insert_sync", "sync")], [20, 20, 10, 10])
        self.assertEqual({ops[k]["batch"] for k in ops}, {1})
        self.assertLessEqual(ops["sync"]["p50_us"], ops["insert_sync"]["p50_us"])
        self.assertIsNone(ops["sync"].get("disk_bytes_written"))
        self.assertIsNotNone(ops["insert_sync"]["disk_bytes_written"])
        self.assertIsNotNone(ops["insert"]["first_us"])

        # A system whose sync rewrites the index times whole saves instead, and one without deletes says so.
        got = self.write("--mode", "latency", "--pass", "inserts", "--round", "0", "--calls", "4", "--warmup", "1",
                         "--durable-calls", "3", "--saves", "2", "--reserve", "5530", variant="whole")
        ops = {e["fields"]["op"]: e["fields"] for e in got if e["event"] == "op"}
        self.assertEqual(sorted(ops), ["delete", "durable_point", "insert"])
        self.assertEqual(ops["durable_point"]["calls"], 2)
        got = self.write("--mode", "latency", "--pass", "inserts", "--round", "0", "--calls", "2", "--warmup", "0",
                         "--durable-calls", "0", "--reserve", "5540", variant="inserts")
        ops = {e["fields"]["op"]: e["fields"] for e in got if e["event"] == "op"}
        self.assertIn("no delete", ops["delete"]["unsupported"])

    def test_a_latency_run_syncs_once_untimed_before_its_durable_inserts(self):
        adapter = Recorder()
        ctx = WriteContext(index=self.index, dataset=self.ds, rows=5000, metric="cosine", threads=1, variant="",
                           mode="latency")
        args = argparse.Namespace(round=0, reserve=5500, warmup=1, calls=2, durable_calls=2, seed=2, saves=2)
        w = adapter.writer_open(ctx, self.index)
        ops = {op.op: op for op in adapter._latency_calls(ctx, w, writes.read_stream(self.cfg.stream), args)}
        # The untimed work, then the timed single writes, the sync that no op times, and the durable inserts.
        self.assertEqual(w, ["prepare"] + ["insert"] * 3 + ["delete"] * 3 + ["sync"] + ["insert", "sync"] * 2)
        self.assertEqual([len(ops[k].times) for k in ("insert", "delete", "insert_sync", "sync")], [2, 2, 2, 2])
        self.assertEqual(ops["insert_sync"].extra["returned"], {"call": "sync", "calls": 12})
        self.assertEqual(ops["sync"].extra["returned"], {"call": "sync", "calls": 12})

    def test_the_policy_pass_maintains_whenever_a_run_is_due(self):
        got = self.write("--mode", "stream", "--pass", "policy", "--round", "3", "--batch", "100", "--policy",
                         "compact", "--maintain-params", json.dumps({"pending": 150}), "--maintain-threads", "2")
        runs = [e["fields"] for e in got if e["event"] == "maintain"]
        self.assertEqual(len(runs), 3)
        self.assertEqual({r["action"] for r in runs}, {"compact"})
        self.assertEqual([r["extra"]["run"] for r in runs], [1, 2, 3])
        self.assertTrue(all(r["extra"]["pending_before"]["inserted"] >= 150 for r in runs))
        point = next(e["fields"] for e in got if e["event"] == "op" and e["fields"]["op"] == "durable_point")
        self.assertEqual(point["extra"]["runs"], 3)
        self.assertLess(point["extra"]["rows_per_s_with_runs"], point["extra"]["rows_per_s_durable"])

    def test_a_library_that_numbers_inserts_wrongly_fails_the_stream(self):
        got = self.write("--mode", "stream", "--pass", "inserts", "--round", "1", "--batch", "100",
                         variant="misnumber")
        self.assertEqual(got[-1]["event"], "error")
        self.assertIn("not by their rows", got[-1]["message"])

    def test_a_maintenance_run(self):
        self.write("--mode", "stream", "--pass", "joint", "--round", "1", "--batch", "100")
        got = self.call("maintain", "--index", str(self.index), "--action", "compact", "--threads", "2")
        fields = got[-1]["fields"]
        self.assertEqual((got[-1]["event"], fields["action"], fields["threads"], fields["rows_live"]),
                         ("maintain", "compact", 2, 5000))
        self.assertGreater(fields["seconds"], 0)
        self.assertGreater(fields["peak_footprint"], 0)
        self.assertEqual(fields["extra"]["report"]["folded"], 50)
        self.assertEqual(Path(fields["extra"]["data_file"]).name, "vectors.npy")
        got = self.call("maintain", "--index", str(self.index), "--action", "fold")
        self.assertEqual(got[-1]["event"], "unsupported")

    def test_a_rebuild_only_system_has_no_write_call(self):
        got = subprocess.run([helpers.python(), "-m", "adapters.diskann", "write", "--index", str(self.index),
                              "--stream", str(self.cfg.stream), "--round", "1", "--data", str(self.ds.base_path),
                              "--rows", "5000"], capture_output=True, text=True, cwd=helpers.PY, env=helpers.env(),
                             timeout=300)
        last = json.loads(got.stdout.splitlines()[-1])
        self.assertEqual(last["event"], "unsupported")
        self.assertIn("fresh build", last["message"])
        info = json.loads(subprocess.run([helpers.python(), "-m", "adapters.diskann", "info"], capture_output=True,
                                         text=True, cwd=helpers.PY, env=helpers.env(), timeout=300).stdout)
        self.assertEqual((info["write_role"], info["sync_kind"], info["maintenance_actions"]), ("rebuild", "whole", []))


if __name__ == "__main__":
    unittest.main()
