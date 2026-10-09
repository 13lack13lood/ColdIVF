import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from adapters import graphs, stores
from adapters.base import ServeContext, Unsupported


def write(path: Path, data: bytes, mtime_ns: int = 1_700_000_000_000_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def fake_session(folder: Path) -> None:
    """What qdrant and RocksDB do to a store they open. It rewrites one file in place, replaces one through a rename,
    deletes one, and makes a new file in a new folder."""
    with open(folder / "config.json", "r+b") as f:
        f.write(b"{")

    tmp = folder / "CURRENT.tmp"
    tmp.write_bytes(b"MANIFEST-2\n")
    os.replace(tmp, folder / "CURRENT")
    (folder / "MANIFEST-1").unlink()
    write(folder / "logs" / "new.log", b"log", mtime_ns=1_800_000_000_000_000_000)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = Path(self.tmp.name) / "index"
        self.data = self.index / "store"
        write(self.data / "vectors.bin", b"v" * 50_000)
        write(self.data / "config.json", b"{}")
        write(self.data / "CURRENT", b"MANIFEST-1\n")
        write(self.data / "MANIFEST-1", b"m1")
        write(self.data / "seg" / "links.bin", b"l" * 5000)
        (self.data / "empty").mkdir()
        self.store = stores.Store(self.index, "store")

    def tearDown(self):
        self.tmp.cleanup()

    def identity(self) -> tuple:
        """What the driver's run keys see of a folder, its bytes, file count and newest change."""
        files = [p for p in self.index.rglob("*") if p.is_file()]
        return sum(p.stat().st_size for p in files), len(files), max(p.stat().st_mtime_ns for p in files)


class CloneTest(StoreCase):
    def test_a_clone_keeps_size_and_change_time_and_writes_stay_apart(self):
        src = self.data / "vectors.bin"
        dst = self.data / "copy.bin"
        stores.clone(src, dst)
        self.assertEqual((dst.stat().st_size, dst.stat().st_mtime_ns), (src.stat().st_size, src.stat().st_mtime_ns))
        self.assertNotEqual(dst.stat().st_ino, src.stat().st_ino)

        with open(dst, "r+b") as f:
            f.write(b"x")

        self.assertEqual(src.read_bytes()[:1], b"v")


class SealTest(StoreCase):
    def test_seal_stashes_what_a_session_changes_or_removes(self):
        record = self.store.seal(fake_session)
        self.assertEqual(record["stash"], ["CURRENT", "MANIFEST-1", "config.json"])
        self.assertEqual(record["made"], ["logs/new.log"])
        self.assertEqual(sorted(record["files"]), ["CURRENT", "MANIFEST-1", "config.json", "seg/links.bin",
                                                   "vectors.bin"])
        self.assertIn("empty", record["dirs"])
        self.assertEqual((self.index / "stash" / "config.json").read_bytes(), b"{}")
        self.assertFalse((self.index / "learn").exists())
        self.assertEqual((self.data / "config.json").read_bytes(), b"{}")
        self.assertEqual(stores.Store(self.index, "store").record, record)

    def test_a_session_that_writes_the_original_fails_the_seal(self):
        original = self.data

        def absolute(folder: Path) -> None:
            with open(original / "config.json", "r+b") as f:
                f.write(b"{")

        with self.assertRaises(stores.StoreChanged):
            self.store.seal(absolute)


class RestoreTest(StoreCase):
    def setUp(self):
        super().setUp()
        self.store.seal(fake_session)
        self.before = self.identity()

    def test_restore_undoes_a_session_in_place(self):
        fake_session(self.data)
        self.assertNotEqual(self.identity(), self.before)
        fixed = self.store.restore()
        self.assertEqual(sorted(fixed), ["CURRENT", "MANIFEST-1", "config.json", "logs/new.log"])
        self.assertEqual(self.identity(), self.before)
        self.assertEqual((self.data / "config.json").read_bytes(), b"{}")
        self.assertEqual((self.data / "CURRENT").read_bytes(), b"MANIFEST-1\n")
        self.assertFalse((self.data / "logs").exists())
        self.assertTrue((self.data / "empty").is_dir())

    def test_a_restore_of_an_untouched_store_does_nothing(self):
        inodes = {p.name: p.stat().st_ino for p in self.data.rglob("*") if p.is_file()}
        self.assertEqual(self.store.restore(), [])
        self.assertEqual({p.name: p.stat().st_ino for p in self.data.rglob("*") if p.is_file()}, inodes)

    def test_a_change_without_a_stashed_copy_raises(self):
        with open(self.data / "vectors.bin", "r+b") as f:
            f.write(b"x")

        with self.assertRaises(stores.StoreChanged):
            self.store.restore()


class PrivateCopyTest(StoreCase):
    def setUp(self):
        super().setUp()
        self.store.seal(fake_session)

    def test_bulk_files_are_shared_and_the_rest_are_clones(self):
        fake_session(self.data)
        copy = self.store.private_copy(lambda rel: rel == "vectors.bin")
        self.assertEqual(copy.parent, self.index.with_name("index.sessions"))
        self.assertTrue(copy.name.startswith(f"{os.getpid()}-"))
        self.assertEqual((copy / "vectors.bin").stat().st_ino, (self.data / "vectors.bin").stat().st_ino)
        self.assertNotEqual((copy / "seg" / "links.bin").stat().st_ino, (self.data / "seg" / "links.bin").stat().st_ino)
        # Stashed files come from the stash, as the build left them, even while the store itself is open.
        self.assertEqual((copy / "config.json").read_bytes(), b"{}")
        self.assertTrue((copy / "MANIFEST-1").exists())
        self.assertFalse((copy / "logs").exists())
        self.assertTrue((copy / "empty").is_dir())
        self.store.drop(copy)
        self.assertFalse(self.store.sessions.exists())

    def test_sweep_removes_copies_whose_process_is_gone(self):
        dead = subprocess.run([sys.executable, "-c", "import os\nprint(os.getpid())"], capture_output=True, text=True)
        stale = self.store.sessions / f"{dead.stdout.strip()}-abcdef"
        live = self.store.sessions / f"{os.getpid()}-123456"
        stale.mkdir(parents=True)
        live.mkdir()
        self.assertEqual(stores.sweep(self.store.sessions), [str(stale)])
        self.assertTrue(live.exists())


HOLD = """
import sys, time
sys.path.insert(0, sys.argv[1])
from adapters import stores
store = stores.Store(sys.argv[2], "store")
print(store.claim(), flush=True)
time.sleep(30)
"""


class ClaimTest(StoreCase):
    def setUp(self):
        super().setUp()
        self.store.seal(fake_session)

    def test_one_claim_at_a_time(self):
        self.assertTrue((self.index / stores.CLAIM).exists())
        self.assertTrue(self.store.claim())
        self.assertTrue(self.store.claim())
        other = stores.Store(self.index, "store")
        self.assertFalse(other.claim())
        self.store.release()
        self.assertTrue(other.claim())
        other.release()

    def test_a_store_another_process_holds_gives_a_private_copy(self):
        py = Path(__file__).resolve().parents[1]
        holder = subprocess.Popen([sys.executable, "-c", HOLD, str(py), str(self.index)], stdout=subprocess.PIPE,
                                  text=True)

        try:
            self.assertEqual(holder.stdout.readline().strip(), "True")
            fake_session(self.data)
            adapter = Kinds()
            graphs.write_meta(self.index, {"bulk": ["vectors.bin"]})
            store, folder = adapter.begin(self.index, adapter.meta(self.index))
            self.assertNotEqual(folder, self.data)
            # The holder's session stays as it is, since only the holder may restore the store.
            self.assertTrue((self.data / "logs" / "new.log").exists())
            self.assertEqual(adapter.stats(type("H", (), {"folder": folder, "store": store})()), {"private_copy": True})
            adapter.end(store, folder)
            self.assertFalse(folder.exists())
        finally:
            holder.kill()
            holder.wait()
            holder.stdout.close()

        self.assertTrue(self.store.claim())
        self.assertEqual(sorted(self.store.restore()), ["CURRENT", "MANIFEST-1", "config.json", "logs/new.log"])


class ReadDataTest(unittest.TestCase):
    def test_holes_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sparse.bin"

            with open(path, "wb") as f:
                f.write(b"a" * 16384)
                f.seek(64 << 20)
                f.write(b"b" * 100)

            self.assertEqual(stores.read_data(path), 16384 + 100)
            (Path(tmp) / "empty.bin").write_bytes(b"")
            self.assertEqual(stores.read_data(Path(tmp) / "empty.bin"), 0)


class Kinds(stores.StoreAdapter):
    system = "kinds"

    def kind(self, rel: str, meta: dict) -> str:
        return "bulk" if rel in meta["bulk"] else "routing"

    def files(self, path: Path) -> list[dict]:
        return self.store_files(Path(path), self.meta(path))


class StoreAdapterTest(StoreCase):
    def setUp(self):
        super().setUp()
        self.store.seal(fake_session)
        graphs.write_meta(self.index, {"bulk": ["vectors.bin", "seg/links.bin"]})
        self.adapter = Kinds()

    def test_files_mark_each_store_file_and_leave_the_stash_unread(self):
        files = {Path(f["path"]).relative_to(self.index).as_posix(): f for f in self.adapter.files(self.index)}
        self.assertEqual(files["store/vectors.bin"]["ranges"], [{"offset": 0, "length": 50_000, "kind": "bulk"}])
        self.assertEqual(files["store/config.json"]["ranges"][0]["kind"], "routing")
        self.assertEqual(files["stash/config.json"]["ranges"], [])
        self.assertEqual(files["store.json"]["ranges"], [])
        self.assertEqual(files["store.lock"]["ranges"], [])
        self.assertEqual(files["meta.json"]["ranges"], [])

    def test_a_private_session_shares_only_bulk_files(self):
        self.adapter.private = True
        store, folder = self.adapter.begin(self.index, self.adapter.meta(self.index))
        self.assertEqual((folder / "seg" / "links.bin").stat().st_ino, (self.data / "seg" / "links.bin").stat().st_ino)
        self.assertNotEqual((folder / "CURRENT").stat().st_ino, (self.data / "CURRENT").stat().st_ino)
        self.adapter.end(store, folder)
        self.assertFalse(folder.exists())

    def test_a_session_in_place_ends_with_a_restore(self):
        store, folder = self.adapter.begin(self.index, self.adapter.meta(self.index))
        self.assertEqual(folder, self.data)
        fake_session(folder)
        self.adapter.end(store, folder)
        self.assertEqual(self.store.restore(), [])

    def test_whole_warm_up_reads_store_files_only(self):
        ctx = ServeContext(index=self.index, dataset=None, rows=0, k=10, metric="cosine", threads=1, variant="")
        self.adapter.warm_up(None, ctx)
        ctx.mode = "t1"

        with self.assertRaises(Unsupported):
            self.adapter.warm_up(None, ctx)

    def test_loaded_store_files_are_marked_and_left_unread(self):
        class Loads(Kinds):
            def loaded(self, rel: str, meta: dict) -> bool:
                return rel == "vectors.bin"

        adapter = Loads()
        files = {Path(f["path"]).relative_to(self.index).as_posix(): f for f in adapter.files(self.index)}
        self.assertEqual(files["store/vectors.bin"]["ranges"],
                         [{"offset": 0, "length": 50_000, "kind": "bulk", "loaded": True}])
        self.assertNotIn("loaded", files["store/config.json"]["ranges"][0])
        ctx = ServeContext(index=self.index, dataset=None, rows=0, k=10, metric="cosine", threads=1, variant="")

        with mock.patch.object(stores, "read_data") as read:
            adapter.warm_up(None, ctx)

        names = {Path(c.args[0]).name for c in read.call_args_list}
        self.assertNotIn("vectors.bin", names)
        self.assertIn("config.json", names)


class GraphsTest(unittest.TestCase):
    def test_pad(self):
        np.testing.assert_array_equal(graphs.pad([4, 2], 4), [4, 2, -1, -1])
        np.testing.assert_array_equal(graphs.pad(np.array([7, 8, 9], dtype=np.uint64), 2), [7, 8])
        np.testing.assert_array_equal(graphs.pad([], 2), [-1, -1])
        self.assertEqual(graphs.pad([1], 3).dtype, np.int64)

    def test_unknown_settings_are_unsupported(self):
        graphs.check_keys({"ef": 1}, ("ef",), "search setting", "x")

        with self.assertRaises(Unsupported):
            graphs.check_keys({"ef": 1, "eff": 2}, ("ef",), "search setting", "x")

    def test_is_unit(self):
        rows = np.random.default_rng(0).standard_normal((500, 8)).astype(np.float32)
        self.assertFalse(graphs.is_unit(rows))
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        self.assertTrue(graphs.is_unit(rows))
        self.assertTrue(graphs.is_unit(rows[:0]))

    def test_batches(self):
        self.assertEqual(list(graphs.batches(5, 2)), [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(list(graphs.batches(0, 2)), [])

    def test_meta_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            graphs.write_meta(Path(tmp), {"a": 1})
            self.assertEqual(graphs.read_meta(Path(tmp)), {"a": 1})


if __name__ == "__main__":
    unittest.main()
