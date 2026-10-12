"""The coldivf module on small random indexes: its hits against an exact search, warm searches, searches from several
threads, inserts and deletes through its writer, maintenance in the foreground and as a background job, and its errors.

    python -m unittest discover -s coldivf-python/tests
"""

import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import coldivf

N, DIM, PER_LIST = 3000, 128, 100

# Random vectors have near-ties that the 8-bit copies a search rescores with can't order, so even a search of every
# cluster finds ~95% of the true 10 nearest, not all of them.
RECALL = 0.9


def rows(n: int, seed: int, unit: bool = True) -> np.ndarray:
    out = np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)
    return out / np.linalg.norm(out, axis=1, keepdims=True) if unit else out


class ColdIVFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.base = rows(N, 1)
        cls.queries = rows(20, 2)
        cls.path = cls.dir / "cosine.coldivf"
        coldivf.build(cls.path, cls.base, vectors_per_list=PER_LIST, parents=8, threads=2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def recall(self, index, base, queries, nearest, **params) -> float:
        found = 0

        for q in queries:
            ids, scores = index.search(q, 10, **params)
            self.assertEqual((len(ids), len(scores)), (10, 10))
            found += len(set(ids) & set(nearest(base, q)[:10].tolist()))

        return found / (10 * len(queries))

    def test_a_search_of_every_cluster_finds_the_true_neighbors(self):
        index = coldivf.Index(self.path, max_k=10, max_nprobe=N // PER_LIST, max_rerank=200)
        index.prepare()
        recall = self.recall(index, self.base, self.queries, lambda b, q: np.argsort(-(b @ q), kind="stable"),
                             nprobe=N // PER_LIST, rerank=200)
        self.assertGreaterEqual(recall, RECALL)

    def test_an_l2_index_finds_the_true_neighbors(self):
        base, path = rows(N, 3, unit=False), self.dir / "l2.coldivf"
        coldivf.build(path, base, metric="l2", vectors_per_list=PER_LIST, parents=8, threads=2)
        queries = rows(20, 4, unit=False)
        recall = self.recall(coldivf.Index(path), base, queries,
                             lambda b, q: np.argsort(((b - q) ** 2).sum(axis=1), kind="stable"),
                             nprobe=N // PER_LIST, rerank=200)
        self.assertGreaterEqual(recall, RECALL)

    def test_a_warm_index_gives_the_same_hits(self):
        index = coldivf.Index(self.path)
        cold = [index.search(q, 10, nprobe=8, rerank=40) for q in self.queries]
        self.assertTrue(index.warm_up(wait=True, timeout=30))
        self.assertTrue(index.is_warm)
        self.assertEqual([index.search(q, 10, nprobe=8, rerank=40) for q in self.queries], cold)
        index.set_warm(False)
        self.assertFalse(index.is_warm)

    def test_searches_from_several_threads_match_one_thread(self):
        index = coldivf.Index(self.path, max_k=10, max_nprobe=8, max_rerank=40)
        want = [index.search(q, 10, nprobe=8, rerank=40) for q in self.queries]
        got = [None] * len(self.queries)

        def work(i):
            for _ in range(20):
                got[i] = index.search(self.queries[i], 10, nprobe=8, rerank=40)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(self.queries))]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(got, want)

    def test_a_search_past_the_limits_grows_them(self):
        index = coldivf.Index(self.path, max_k=5, max_nprobe=2, max_rerank=5)
        small = index.search(self.queries[0], 5, nprobe=2, rerank=5)
        ids, _ = index.search(self.queries[0], 10, nprobe=16, rerank=50)
        self.assertEqual(len(ids), 10)
        self.assertEqual(index.search(self.queries[0], 5, nprobe=2, rerank=5), small)

    def test_reach_reads_at_most_nprobe_clusters(self):
        index = coldivf.Index(self.path)
        ids, _ = index.search(self.queries[0], 10, nprobe=8, rerank=40, reach=0.0)
        self.assertGreater(len(ids), 0)

    def test_the_file_checks_out(self):
        index = coldivf.Index(self.path)
        index.verify()
        self.assertFalse(index.refresh())

    def test_bad_requests_raise(self):
        index = coldivf.Index(self.path)

        with self.assertRaises(ValueError):
            index.search(np.zeros(DIM + 1, np.float32), 10, nprobe=4, rerank=20)

        with self.assertRaises((BufferError, TypeError, ValueError)):
            index.search(np.zeros(DIM, np.float64), 10, nprobe=4, rerank=20)

        with self.assertRaises(FileNotFoundError):
            coldivf.Index(self.dir / "missing.coldivf")

        with self.assertRaises(ValueError):
            coldivf.build(self.dir / "dot.coldivf", self.base, metric="dot")

        with self.assertRaises(ValueError):
            coldivf.build(self.dir / "flat.coldivf", self.base[0])


class WriterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.base = rows(N, 1)
        cls.built = cls.dir / "built.coldivf"
        coldivf.build(cls.built, cls.base, vectors_per_list=PER_LIST, parents=8, threads=2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy(self, name: str) -> Path:
        path = self.dir / f"{name}.coldivf"
        shutil.copyfile(self.built, path)
        return path

    def top(self, index, query) -> list:
        # Every cluster, so the nearest stored vector always comes first.
        ids, _ = index.search(query, 10, nprobe=N // PER_LIST, rerank=100)
        return ids

    def wait_for(self, done, timeout=30.0):
        deadline = time.monotonic() + timeout

        while not done() and time.monotonic() < deadline:
            time.sleep(0.001)

        self.assertTrue(done())

    def run_on_thread(self, job) -> tuple:
        # The thread that runs the job, and the InterruptedError that stops the run, once one does.
        stopped = []

        def run():
            try:
                job.run()
            except InterruptedError as e:
                stopped.append(e)

        thread = threading.Thread(target=run)
        thread.start()
        return thread, stopped

    def test_an_insert_is_found_by_a_search_of_the_same_process(self):
        path = self.copy("insert")
        index = coldivf.Index(path)
        new = rows(5, 10)

        with coldivf.Writer(path) as writer:
            writer.insert(N, new[0])
            self.assertEqual(self.top(index, new[0])[0], N)

            # A batch, from an array whose rows aren't contiguous.
            writer.insert_many(np.arange(N + 1, N + 3), new[1::2])
            self.assertEqual([self.top(index, new[r])[0] for r in (1, 3)], [N + 1, N + 2])

    def test_a_replace_keeps_one_copy(self):
        path = self.copy("replace")
        index = coldivf.Index(path)
        new = rows(1, 11)[0]

        with coldivf.Writer(path) as writer:
            writer.insert(7, new)
            ids = self.top(index, new)
            self.assertEqual((ids[0], ids.count(7)), (7, 1))
            self.assertNotIn(7, self.top(index, self.base[7]))

    def test_a_delete_hides_a_vector(self):
        path = self.copy("delete")
        index = coldivf.Index(path)
        new = rows(1, 12)[0]

        with coldivf.Writer(path) as writer:
            writer.insert(N, new)
            writer.delete(N)
            writer.delete(7)
            writer.delete_many([8, 9])
            self.assertNotIn(N, self.top(index, new))

            for id in (7, 8, 9):
                self.assertNotIn(id, self.top(index, self.base[id]))

    def test_a_reopen_after_a_sync_sees_the_changes(self):
        path = self.copy("reopen")
        new = rows(1, 13)[0]

        with coldivf.Writer(path) as writer:
            writer.insert(N, new)
            writer.delete(7)
            writer.sync()

        # No index of the file is open any more, so this one reads the changes from the log.
        index = coldivf.Index(path)
        self.assertEqual(self.top(index, new)[0], N)
        self.assertNotIn(7, self.top(index, self.base[7]))

        with coldivf.Writer(path) as writer:
            status = writer.status()
            self.assertEqual((status.pending, status.deleted, status.due), (1, 1, False))

    def test_maintenance_keeps_the_inserts_and_drops_the_deletes(self):
        path = self.copy("compact")
        index = coldivf.Index(path)
        new = rows(2, 14)

        with coldivf.Writer(path) as writer:
            writer.insert_many([N, N + 1], new)
            writer.delete_many([N + 1, 7])
            report = writer.compact()
            self.assertEqual((report.kind, report.count, report.lists), ("fold", N, N // PER_LIST))
            self.assertGreater(report.bytes_written, 0)
            status = writer.status()
            self.assertEqual((status.pending, status.deleted), (0, 0))

            # The first run moved the index into its folder, and the open index moved with it.
            self.assertFalse(path.exists())
            self.assertEqual(coldivf.data_file(path).parent.name, f"{path.name}.d")
            self.assertEqual(self.top(index, new[0])[0], N)
            self.assertNotIn(N + 1, self.top(index, new[1]))
            self.assertNotIn(7, self.top(index, self.base[7]))

            report = writer.maintain("major", vectors_per_list=PER_LIST, threads=2)
            self.assertEqual((report.kind, report.count), ("major", N))

        index = coldivf.Index(path)
        self.assertEqual(self.top(index, new[0])[0], N)
        self.assertNotIn(7, self.top(index, self.base[7]))

    def test_a_background_job_commits_while_searches_go_on(self):
        path = self.copy("job")
        index = coldivf.Index(path)
        new = rows(200, 15)

        with coldivf.Writer(path) as writer:
            self.assertFalse(writer.status().due)
            self.assertIsNone(writer.begin_compact())

            # 200 inserts pass 5% of the 3000 vectors, which makes a run due.
            writer.insert_many(np.arange(N, N + 200), new)
            self.assertTrue(writer.status().due)
            job = writer.begin_compact(threads=2)
            thread = threading.Thread(target=job.run)
            thread.start()
            self.wait_for(lambda: job.running or not thread.is_alive())

            # Searches see every change while the job runs, and finish waits for the run.
            self.assertEqual(self.top(index, new[0])[0], N)
            report = job.finish()
            thread.join()
            self.assertEqual(report.count, N + 200)
            self.assertFalse(job.running)
            self.assertEqual(writer.status().pending, 0)

            with self.assertRaises(ValueError):
                job.finish()

        self.assertEqual(self.top(index, new[1])[0], N + 1)

    def test_a_cancelled_job_leaves_the_index_as_it_was(self):
        path = self.copy("cancel")
        index = coldivf.Index(path)
        new = rows(200, 16)

        with coldivf.Writer(path) as writer:
            writer.insert_many(np.arange(N, N + 200), new)

            # A search just now and an hour's quiet time hold the job back until the cancel.
            self.top(index, new[0])
            job = writer.begin_compact(threads=2, throttle={"quiet": 3600})
            thread, stopped = self.run_on_thread(job)
            self.wait_for(lambda: job.running)
            job.cancel()
            thread.join(timeout=30)
            self.assertEqual(len(stopped), 1)

            with self.assertRaises(InterruptedError):
                job.finish()

            # A job cancelled before it ran ends at once, so the writer can start another.
            job = writer.begin_compact(threads=2)
            job.cancel()

            with self.assertRaises(InterruptedError):
                job.run()

            self.assertEqual(writer.status().pending, 200)
            self.assertEqual(self.top(index, new[0])[0], N)
            job = writer.begin_compact(threads=2)
            job.run(throttle=False)
            self.assertEqual(job.finish().count, N + 200)

    def test_a_finish_waits_for_a_run_that_has_not_started(self):
        path = self.copy("pool")

        with coldivf.Writer(path) as writer, ThreadPoolExecutor(1) as pool:
            writer.insert_many(np.arange(N, N + 200), rows(200, 19))

            # A cancel ends a finish that waits for a job no thread runs.
            job = writer.begin_compact(threads=2)
            threading.Timer(0.05, job.cancel).start()

            with self.assertRaises(InterruptedError):
                job.finish()

            # The pool's thread starts the run a moment after the submit, so the finish often comes first.
            job = writer.begin_compact(threads=2)
            run = pool.submit(job.run)
            self.assertEqual(job.finish().count, N + 200)
            run.result()

    def test_a_close_during_a_run_frees_the_index_once_the_run_stops(self):
        path = self.copy("close")
        index = coldivf.Index(path)
        new = rows(200, 20)
        writer = coldivf.Writer(path)
        writer.insert_many(np.arange(N, N + 200), new)

        # A search just now and an hour's quiet time hold the job back until the close.
        self.top(index, new[0])
        job = writer.begin_compact(threads=2, throttle={"quiet": 3600})
        thread, stopped = self.run_on_thread(job)
        self.wait_for(lambda: job.running)

        # The close cancels the run, which holds the index's lock until it stops.
        writer.close()
        thread.join(timeout=30)
        self.assertEqual(len(stopped), 1)

        with self.assertRaises(InterruptedError):
            job.finish()

        with coldivf.Writer(path) as writer:
            self.assertEqual(writer.status().pending, 200)

    def test_a_second_writer_raises_blocking_io_error(self):
        path = self.copy("second")

        with coldivf.Writer(path):
            with self.assertRaises(BlockingIOError):
                coldivf.Writer(path)

            # So does one in another process.
            child = "import sys, coldivf\ntry: coldivf.Writer(sys.argv[1])\nexcept BlockingIOError: sys.exit(3)"
            self.assertEqual(subprocess.run([sys.executable, "-c", child, str(path)]).returncode, 3)

        with coldivf.Writer(path) as writer:
            writer.insert_many(np.arange(N, N + 200), rows(200, 17))
            job = writer.begin_compact(threads=2)

            with self.assertRaises(BlockingIOError):
                writer.begin_compact()

            with self.assertRaises(BlockingIOError):
                writer.compact()

        # The close ended the job, which held the index's lock too.
        coldivf.Writer(path).close()

        with self.assertRaises(InterruptedError):
            job.run()

    def test_bad_requests_raise(self):
        path = self.copy("bad")
        writer = coldivf.Writer(path)

        for bad in (np.zeros(DIM + 1, np.float32), np.zeros(DIM, np.float32), np.full(DIM, np.nan, np.float32),
                    self.base[:1], np.ascontiguousarray(self.base[:2, :DIM // 2])):
            with self.assertRaises(ValueError):
                writer.insert(N, bad)

        with self.assertRaises((BufferError, TypeError, ValueError)):
            writer.insert(N, np.zeros(DIM, np.float64))

        for id in (-1, 2**63):
            with self.assertRaises(OverflowError):
                writer.insert(id, self.base[0])

        with self.assertRaises(ValueError):
            writer.insert_many([N], self.base[0])

        with self.assertRaises(ValueError):
            writer.insert_many([N, N + 1], self.base[:3])

        with self.assertRaises(ValueError):
            writer.insert_many([N], np.zeros((1, DIM + 1), np.float32))

        with self.assertRaises(TypeError):
            writer.insert_many(np.array([[N]]), self.base[:1])

        self.assertEqual(writer.status().pending, 0)

        # A batch stops at the row that fails and keeps the rows before it.
        batch = rows(3, 18)
        batch[1, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "row 1"):
            writer.insert_many([N, N + 1, N + 2], batch)

        self.assertEqual(writer.status().pending, 1)

        with self.assertRaises(ValueError):
            writer.maintain("huge")

        with self.assertRaises(ValueError):
            writer.maintain(vectors_per_list=0)

        with self.assertRaises(ValueError):
            writer.status(start={"min_interval": -1})

        # Values out of range raise even when no run is due.
        with self.assertRaises(ValueError):
            writer.status(minor={"shrink": 2.0})

        with self.assertRaises(ValueError):
            writer.begin_compact(throttle={"burst_bytes": 0})

        with self.assertRaises(OverflowError):
            writer.begin_compact(threads=-1)

        with self.assertRaises(TypeError):
            writer.begin_compact(threads=1.5)

        with self.assertRaises(TypeError):
            writer.begin_compact(start={"nonsense": 1})

        with self.assertRaises(TypeError):
            writer.begin_compact(nonsense=1)

        writer.close()
        writer.close()

        with self.assertRaises(ValueError):
            writer.insert(N, self.base[0])

        with self.assertRaises(ValueError):
            with writer:
                pass

        with self.assertRaises(FileNotFoundError):
            coldivf.Writer(self.dir / "missing.coldivf")


if __name__ == "__main__":
    unittest.main()
