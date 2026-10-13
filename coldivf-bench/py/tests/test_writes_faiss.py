import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import helpers
from adapters import faiss_common as fc
from adapters import faiss_hnsw, faiss_ivf, faiss_layout
from adapters.base import BuildContext, ServeContext, Unsupported, WriteContext
import driver
import schema
import writes

HAS_FAISS = importlib.util.find_spec("faiss") is not None


class SaveTest(unittest.TestCase):
    """The parts of a save that need no faiss."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_meta_file_is_replaced_whole(self):
        fc.write_meta(self.dir, {"layout": "mapped", "rows": 5})
        fc.save_meta(self.dir, {"layout": "mapped", "rows": np.int64(5), "state_rows": 9})
        self.assertEqual(fc.read_meta(self.dir), {"layout": "mapped", "rows": 5, "state_rows": 9})
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["meta.json"])

    def test_a_dead_writers_temporary_file_is_one_the_driver_removes(self):
        tmp = fc.temporary(self.dir / "index.faiss")
        self.assertEqual(tmp.parent, self.dir)
        self.assertTrue(driver.STALE.search(tmp.name))
        tmp.write_bytes(b"half a file")

        with mock.patch.object(driver, "_alive", lambda pid: False):
            self.assertEqual(driver.remove_stale(self.dir), [str(tmp)])


def held_ids(ivf) -> np.ndarray:
    """Every id in the lists of a loaded IVF index, in order."""
    import faiss

    lists = ivf.invlists
    out = [faiss.rev_swig_ptr(lists.get_ids(i), lists.list_size(i)).copy() for i in range(ivf.nlist)
           if lists.list_size(i)]
    return np.sort(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


@unittest.skipUnless(HAS_FAISS, "faiss isn't installed")
class WriterTest(unittest.TestCase):
    """The write hooks in this process, on 2,000 rows with 600 more to insert."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=2600, dim=32, nq=20)
        self.base = np.asarray(self.ds.base, dtype=np.float32)

    def tearDown(self):
        self.tmp.cleanup()

    def adapter(self, cls: type, variant: str) -> object:
        a = cls()
        a.variant = variant
        a.load()
        return a

    def build(self, a: object, config: dict, name: str) -> Path:
        out = self.root / name
        out.mkdir()
        a.build(BuildContext(config=config, data_path=self.ds.base_path, rows=2000, out=out, clusters=None,
                             threads=2, seed=3, metric="cosine", variant=a.variant, dataset=self.ds))
        return out

    def ctx(self, path: Path, params: dict | None = None) -> WriteContext:
        return WriteContext(index=path, dataset=self.ds, rows=2000, metric="cosine", threads=1, variant="",
                            params=params or {})

    def first(self, a: object, h: object, rows: np.ndarray, params: dict, k: int = 10) -> np.ndarray:
        """Each row's top k ids, with the row itself as the query."""
        return np.stack([a.search(h, self.base[r], k, params) for r in rows])

    def test_ivf_writes_in_every_layout_and_the_states_they_save(self):
        import faiss

        params = {"nprobe": 20, "k_factor": 8}

        for variant in ("shipped", "batched", "fastscan"):
            a = self.adapter(faiss_ivf.FaissIVF, variant)
            out = self.build(a, {"vectors_per_list": 100}, variant)
            w = a.writer_open(self.ctx(out), out)
            got = a.insert(w, np.arange(2000, 2300), self.base[2000:2300])
            self.assertEqual(got.tolist(), list(range(2000, 2300)), variant)
            dead = np.concatenate([np.arange(0, 600, 3), np.arange(2000, 2100, 2)])
            a.delete(w, dead)
            pending = a.pending(w)
            self.assertEqual((pending["live_rows"], pending["state_rows"], pending["dead_rows"]), (2050, 2300, 250))
            # The deleted ids leave the lists and keep their SQ8 rows, so each id still finds its own row.
            live = np.setdiff1d(np.arange(2300), dead)
            self.assertTrue(np.array_equal(held_ids(w.ivf), live), variant)
            self.assertEqual((w.top.ntotal, w.pre.ntotal, w.rows.ntotal), (2300, 2050, 2300))
            decoded = w.rows.reconstruct_n(2000, 300)
            nearest = np.argmin(((decoded[:, None, :] - self.base[None, 2000:2300]) ** 2).sum(axis=2), axis=1)
            self.assertTrue(np.array_equal(nearest, np.arange(300)), variant)

            # A search of the loaded index finds the writes at once.
            w.ivf.nprobe, w.top.k_factor = 20, 8
            _, found = w.top.search(self.base[2001:2300:2], 1)
            self.assertEqual(float(np.mean(found[:, 0] == np.arange(2001, 2300, 2))), 1.0, variant)
            _, found = w.top.search(self.base[dead], 10)
            self.assertFalse(np.isin(found, dead).any(), variant)

            info = a.sync(w)
            self.assertEqual((info["call"], info["file"]), ("write_index", "index.faiss"))
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["index.faiss", "meta.json"])
            a.writer_close(w)
            meta = fc.read_meta(out)
            self.assertEqual((meta["state_rows"], meta["live_rows"], meta["rows"]), (2300, 2050, 2000))
            self.assertEqual(meta["parts"], fc.parts_record(out, ["index.faiss"], loaded=variant == "fastscan"))

            if variant != "fastscan":
                layout = faiss_layout.read(out / "index.faiss")
                self.assertEqual(meta["offsets"], faiss_ivf.offsets_record(layout))
                self.assertEqual(int(layout.lists.sizes.sum()), 2050)

            # The saved state opens as a build does, mapped, through the batched waves or loaded whole.
            s = self.adapter(faiss_ivf.FaissIVF, variant)
            h = s.open(ServeContext(index=out, dataset=self.ds, rows=2300, k=10, metric="cosine", threads=1,
                                    variant=variant, params=params), out)
            self.assertEqual(h.waves is not None, variant == "batched")
            found = self.first(s, h, np.arange(2001, 2300, 2), params, k=1)
            self.assertEqual(float(np.mean(found[:, 0] == np.arange(2001, 2300, 2))), 1.0, variant)
            found = self.first(s, h, dead, params)
            self.assertFalse(np.isin(found, dead).any(), variant)
            found = self.first(s, h, live[:300:3], params, k=1)
            self.assertEqual(float(np.mean(found[:, 0] == live[:300:3])), 1.0, variant)
            s.close(h)
            self.assertIsInstance(faiss.read_index(str(out / "index.faiss")), faiss.IndexRefine)

    def test_fastscan_rows_added_after_a_delete(self):
        import faiss

        params = {"nprobe": 20, "k_factor": 8}
        later = np.arange(2300, 2400)
        result = {}

        for fix in (False, True):
            a = self.adapter(faiss_ivf.FaissIVF, "fastscan")
            out = self.build(a, {"vectors_per_list": 100}, f"fs-{fix}")
            w = a.writer_open(self.ctx(out, {"clear_removed": fix}), out)
            a.insert(w, np.arange(2000, 2300), self.base[2000:2300])
            a.delete(w, np.concatenate([np.arange(0, 600, 3), np.arange(2000, 2100, 2)]))
            a.insert(w, later, self.base[later])
            self.assertEqual("cleared_slots" in a.pending(w), fix)
            self.assertEqual(w.cleared > 0, fix)
            a.sync(w)
            a.writer_close(w)
            h = a.open(ServeContext(index=out, dataset=self.ds, rows=2400, k=10, metric="cosine", threads=1,
                                    variant="fastscan", params=params), out)
            result[fix] = float(np.mean(self.first(a, h, later, params, k=1)[:, 0] == later))

        # Clearing the slots a delete leaves makes every later row its own nearest, as in the plain layout. Without
        # it faiss 1.15.1 mixes the new rows' codes with the removed ones'.
        self.assertEqual(result[True], 1.0)

        if faiss.__version__ == "1.15.1":
            self.assertLess(result[False], 0.9)

    def test_writes_it_refuses(self):
        a = self.adapter(faiss_ivf.FaissIVF, "shipped")
        out = self.build(a, {"vectors_per_list": 100}, "ivf")
        w = a.writer_open(self.ctx(out), out)

        with self.assertRaises(RuntimeError):
            a.delete(w, np.array([5, 2500]))

        for params in ({"clear_removed": True}, {"batch": 3}):
            with self.assertRaises(ValueError, msg=str(params)):
                a.writer_open(self.ctx(out, params), out)

        bare = self.build(a, {"vectors_per_list": 100, "refine": "none"}, "bare")

        with self.assertRaises(Unsupported):
            a.writer_open(self.ctx(bare), bare)

        ondisk = self.adapter(faiss_ivf.FaissIVF, "ondisk")
        self.assertEqual((ondisk.write_role, a.write_role, a.sync_kind), ("rebuild", "full", "whole"))

        with self.assertRaises(Unsupported):
            ondisk.writer_open(self.ctx(out), out)

        # A variant searches only the layout it builds, and writes the same way.
        with self.assertRaises(Unsupported):
            self.adapter(faiss_ivf.FaissIVF, "fastscan").writer_open(self.ctx(out), out)

    def test_hnsw_takes_inserts_only(self):
        a = self.adapter(faiss_hnsw.FaissHNSW, "")
        self.assertEqual((a.write_role, a.sync_kind, a.durability["sync"]), ("inserts", "whole", "power"))
        out = self.build(a, {"m": 16, "storage": "sq8", "ef_construction": 40}, "hnsw")
        w = a.writer_open(self.ctx(out), out)
        self.assertEqual(a.insert(w, np.arange(2000, 2100), self.base[2000:2100]).tolist(), list(range(2000, 2100)))
        self.assertEqual(a.insert(w, np.arange(2100, 2101), self.base[2100:2101]).tolist(), [2100])

        with self.assertRaises(Unsupported):
            a.delete(w, np.array([3]))

        self.assertEqual(a.pending(w), {"live_rows": 2101})
        a.sync(w)
        a.writer_close(w)
        meta = fc.read_meta(out)
        self.assertEqual((meta["state_rows"], meta["parts"]), (2101, fc.parts_record(out, ["index.faiss"])))
        params = {"ef_search": 64}
        h = a.open(ServeContext(index=out, dataset=self.ds, rows=2101, k=10, metric="cosine", threads=1,
                                variant="", params=params), out)
        rows = np.arange(2000, 2101)
        self.assertEqual(float(np.mean(self.first(a, h, rows, params, k=1)[:, 0] == rows)), 1.0)


WRITE_SUITE = """
phase = "writes"
dataset = "tiny-16-cosine"
n = 5000
index_root = "{root}/indexes"
results_root = "{root}/results"
seed = 3
max_attempts = 1
wait_s = 0.05

[gate]
mode = "off"
invalid_vm = []

[writes]
stream = "{root}/writes/tiny-w.stream"
batch = 100
latency_states = ["s0", "s10"]
latency_calls = 10
warmup_calls = 2
durable_calls = 4
saves = 2
seed = 1
probes = 30

[writes.states]
s1 = { inserts = 50, deletes = 50 }
s5 = { inserts = 250, deletes = 250 }
s10 = { inserts = 500, deletes = 500 }
s1i = { inserts = 50, deletes = 0 }
s5i = { inserts = 250, deletes = 0 }
s10i = { inserts = 500, deletes = 0 }
"""

IVF = """
[[system]]
name = "{name}"
adapter = "faiss_ivf"
variant = "{variant}"

  [[system.build]]
  config_id = "vpl100"
  config = { vectors_per_list = 100 }
  threads = 2

  [[system.write]]
  config_id = "vpl100"
  role = "full"
  params = {params}

  [[system.warm]]
  config_id = "vpl100"
  states = "all"
  split = "test"
  passes = 1
  warmup = 4
  settings = [{ nprobe = 50, k_factor = 32 }]
  findability = true

  [[system.cold]]
  config_id = "vpl100"
  states = ["s0", "s10"]
  params = { nprobe = 50, k_factor = 32 }
  trials = 2
  next = 0
"""

HNSW = """
[[system]]
name = "faiss-hnsw"
adapter = "faiss_hnsw"

  [[system.build]]
  config_id = "hnsw16"
  config = { m = 16, storage = "sq8", ef_construction = 40 }
  threads = 2

  [[system.write]]
  config_id = "hnsw16"
  role = "inserts"

  [[system.warm]]
  config_id = "hnsw16"
  states = "all"
  split = "test"
  passes = 1
  warmup = 4
  settings = [{ ef_search = 64 }]
  findability = true

  [[system.cold]]
  config_id = "hnsw16"
  states = ["s0", "s10i"]
  params = { ef_search = 64 }
  trials = 2
  next = 0
"""


@unittest.skipUnless(HAS_FAISS, "faiss isn't installed")
class WriteBatchTest(unittest.TestCase):
    """A whole write batch through the driver on 5,000 rows, for faiss IVF as shipped, with batched reads, FastScan with
    clear_removed, and HNSW with inserts only: every pass and latency run, then warm and cold searches of the states."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = helpers.make_dataset(self.root / "data", n=5600, dim=32, nq=60)
        self.env = mock.patch.dict(os.environ, {"COLDIVF_DATA": str(self.root / "data")})
        self.env.start()
        self.quiet = mock.patch.object(driver.Driver, "log", lambda self, msg: None)
        self.quiet.start()

    def tearDown(self):
        self.quiet.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_every_setup_writes_and_searches_its_states(self):
        import faiss

        text = WRITE_SUITE.replace("{root}", str(self.root))
        setups = (("faiss-ivf", "shipped", "{}"), ("faiss-ivf", "batched", "{}"),
                  ("faiss-fastscan", "fastscan", "{ clear_removed = true }"))

        for name, variant, params in setups:
            text += IVF.replace("{name}", name).replace("{variant}", variant).replace("{params}", params)

        path = self.root / "suite.toml"
        path.write_text(text + HNSW)
        suite = driver.load_suite(path)
        cfg = suite.writes
        stream = writes.ensure_stream(cfg, self.ds)
        writes.compute_truths(cfg, self.ds, "numpy", log=lambda m: None)
        self.assertEqual(driver.Driver(suite).run(), 0)
        lines = [x for name in ("faiss-ivf", "faiss-fastscan", "faiss-hnsw")
                 for x in schema.read(self.root / "results" / "writes" / f"{name}.jsonl")]

        for line in lines:
            self.assertEqual(schema.validate(line), [], line.get("error"))

        # Only the inserts-only system's latency deletes fail, as unsupported.
        bad = [(x["system"], x["kind"], x.get("op"), x["status"]) for x in lines if x["status"] != "ok"]
        self.assertEqual(bad, [("faiss-hnsw", "write", "delete", "unsupported")] * 2)
        ivf = [("faiss-ivf", "shipped", "vpl100"), ("faiss-ivf", "batched", "vpl100"),
               ("faiss-fastscan", "fastscan", "vpl100")]

        for name, variant, config in ivf + [("faiss-hnsw", "", "hnsw16")]:
            mine = [x for x in lines if (x["system"], x["variant"]) == (name, variant)]
            stream_ops = {(x["state_from"], x["state_to"], x["op"]) for x in mine
                          if x["kind"] == "write" and x["mode"] == "stream"}
            latency = sorted((x["state_from"], x["op"], x["calls"]) for x in mine
                             if x["kind"] == "write" and x["mode"] == "latency" and x["status"] == "ok")

            if name == "faiss-hnsw":
                ladder, ops, at = [("s0", "s1i"), ("s1i", "s5i"), ("s5i", "s10i")], ("insert", "durable_point"), "s10i"
            else:
                ladder = [("s0", "s1"), ("s1", "s5"), ("s5", "s10"), ("s0", "s10i")]
                ops, at = ("insert", "delete", "durable_point"), "s10"

            # A save rewrites the whole index, so a stream saves once at its end and never syncs.
            self.assertEqual(stream_ops, {(a, b, op) for a, b in ladder for op in ops
                                          if not (b == "s10i" and op == "delete")}, name)
            want = [(s, op, n) for s in sorted(("s0", at)) for op, n in (("delete", 10), ("durable_point", 2),
                                                                         ("insert", 10)) if
                    not (name == "faiss-hnsw" and op == "delete")]
            self.assertEqual(latency, want, name)
            point = next(x for x in mine if x.get("op") == "durable_point" and x["state_to"] == at
                         and x["mode"] == "stream")
            live = int(writes.state_mask(cfg, stream, at).sum())
            self.assertEqual((point["durability"], point["extra"]["pending"]["live_rows"]), ("power", live), name)
            self.assertGreater(point["disk_bytes_written"], 0)

            searched = [x for x in mine if x["kind"] in ("warm", "cold")]
            self.assertEqual({x["dead_returned"] for x in searched}, {0.0}, (name, variant))
            warm = {x["state"]: x for x in searched if x["kind"] == "warm"}
            self.assertEqual(sorted(warm), sorted(["s0"] + [b for _, b in ladder]), name)

            for x in warm.values():
                self.assertGreaterEqual(x["recall10"], 0.9, (name, variant, x["state"]))

                if x["state"] in ("s10", "s10i"):
                    self.assertGreaterEqual(x["recall10_inserted"], 0.8, (name, variant, x["state"]))

            cold = [x for x in searched if x["kind"] == "cold"]
            self.assertEqual(sorted(x["state"] for x in cold), sorted(["s0", "s0", at, at]), name)
            self.assertGreaterEqual(np.mean([x["recall10"] for x in cold]), 0.9, (name, variant))

            found = {x["state"]: x["extra"]["findability"] for x in mine if x["kind"] == "warm"
                     and "findability" in x["extra"]}
            self.assertEqual(sorted(found), ["s10i"] if name == "faiss-hnsw" else ["s10", "s10i"], name)
            self.assertTrue(all(v["inserted"] == 1.0 and v["base"] == 1.0 for v in found.values()), (name, found))

            # Each state's lists hold exactly the vectors the stream leaves it, and its meta file the new offsets.
            folder = self.root / "indexes" / (f"{name}-{variant}" if variant else name)

            for state in [b for _, b in ladder]:
                index = folder / f"{config}-{state}" / "index.faiss"
                meta = json.loads((index.parent / "meta.json").read_text())
                mask = writes.state_mask(cfg, stream, state)
                rows = cfg.n + cfg.spec(state).inserts
                self.assertEqual((meta["state_rows"], meta["live_rows"]), (rows, int(mask.sum())), state)
                loaded = faiss.read_index(str(index))

                if name == "faiss-hnsw":
                    self.assertEqual(loaded.ntotal, rows)
                    continue

                inner = faiss.downcast_index(faiss.extract_index_ivf(faiss.downcast_index(loaded.base_index)))
                self.assertTrue(np.array_equal(held_ids(inner), np.flatnonzero(mask)), (name, state))

                if variant != "fastscan":
                    self.assertEqual(meta["offsets"], faiss_ivf.offsets_record(faiss_layout.read(index)))


if __name__ == "__main__":
    unittest.main()
