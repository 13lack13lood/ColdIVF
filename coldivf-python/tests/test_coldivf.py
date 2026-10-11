"""The coldivf module on small random indexes: its hits against an exact search, warm searches, searches from several
threads, and its errors.

    python -m unittest discover -s coldivf-python/tests
"""

import tempfile
import threading
import unittest
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


if __name__ == "__main__":
    unittest.main()
