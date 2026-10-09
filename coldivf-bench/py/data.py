"""Datasets, query splits and recall.

Vectors come as .fbin files and ground truth as .ibin files. Both start with two 32-bit integers, the row count and
the row width, then hold the rows as 4-byte values. They are opened as numpy memory maps, so nothing loads until it is
read, and a prefix of N vectors is the row range 0 to N of the one base file.

Ground truth for the full base is <root>/<name>.gt.ibin. For a prefix of n rows it is <derived>/<name>-<n>.gt.ibin.
A <stem>.gt.dist.fbin next to an .ibin holds the matching distances or scores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Queries past this many are left out of the tune and test splits.
SPLIT_LIMIT = 1000

# A result ties the k-th distance within this share of it, two float32 units in the last place, and never less than
# this much in absolute terms. It is far below the gaps between true neighbors, eg. ~7e-4 between the 10th and the 11th
# of a Cohere query, so only distances float32 can't tell apart tie.
TIE_RTOL = 2.0 ** -22

# A distances file whose k-th value differs from the exact one by more than this share doesn't belong to the base.
DIST_CHECK_RTOL = 1e-3


def data_root() -> Path:
    """Where the datasets are. It is $COLDIVF_DATA, or else data/datasets in the repo."""
    env = os.environ.get("COLDIVF_DATA")
    return Path(env) if env else REPO / "data" / "datasets"


def derived_root() -> Path:
    """Where derived data goes, eg. the ground truth of a prefix. It is data/ in the repo."""
    return REPO / "data"


def header(path: Path | str) -> tuple[int, int]:
    """The row count and the row width of an .fbin or .ibin file."""
    with open(path, "rb") as f:
        head = f.read(8)

    if len(head) < 8:
        raise ValueError(f"{path} is shorter than its header")

    count, width = (int(x) for x in np.frombuffer(head, dtype="<i4"))
    return count, width


def _table(path: Path | str, dtype: str) -> np.ndarray:
    path = Path(path)
    count, width = header(path)

    if count < 0 or width <= 0:
        raise ValueError(f"{path} has a negative count or a width that isn't positive")

    if path.stat().st_size != 8 + count * width * 4:
        raise ValueError(f"{path} doesn't hold as many rows as its header says")

    if count == 0:
        return np.zeros((0, width), dtype=dtype)

    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(count, width))


def fbin(path: Path | str) -> np.ndarray:
    """A mapped .fbin file as a (count, width) float32 array."""
    return _table(path, "<f4")


def ibin(path: Path | str) -> np.ndarray:
    """A mapped .ibin file as a (count, width) int32 array."""
    return _table(path, "<i4")


def _write_table(path: Path | str, rows: np.ndarray, dtype: str) -> None:
    rows = np.asarray(rows, dtype=dtype)

    if rows.ndim != 2:
        raise ValueError("rows must be a 2-d array")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")

    with open(tmp, "wb") as f:
        np.array(rows.shape, dtype="<i4").tofile(f)
        rows.tofile(f)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def write_fbin(path: Path | str, rows: np.ndarray) -> None:
    """Writes rows of float32 as an .fbin file, through a temporary file and a rename."""
    _write_table(path, rows, "<f4")


def write_ibin(path: Path | str, rows: np.ndarray) -> None:
    """Writes rows of int32 as an .ibin file, the same way."""
    _write_table(path, rows, "<i4")


def metric_of(name: str) -> str:
    """The metric a dataset's name ends with. It is "l2" for -euclidean, "ip" for -ip or -dot, and "cosine"
    otherwise."""
    if name.endswith("-euclidean"):
        return "l2"

    if name.endswith("-ip") or name.endswith("-dot"):
        return "ip"

    return "cosine"


def distances(metric: str, query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Exact distances in float64 from `query` to each row, smaller is nearer. They are the squared length of the
    difference for l2, minus the inner product for ip, and minus the cosine for cosine."""
    q = np.asarray(query, dtype=np.float64)
    x = np.asarray(rows, dtype=np.float64).reshape(-1, q.shape[0])

    if metric == "l2":
        diff = x - q
        return np.einsum("ij,ij->i", diff, diff)

    dots = x @ q

    if metric == "ip":
        return -dots

    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(q)
    return -dots / np.where(norms > 0, norms, 1.0)


def split_ids(count: int, split: str, limit: int = SPLIT_LIMIT) -> np.ndarray:
    """The query ids of a split. Among the first `limit` queries, even positions are "tune" and odd positions "test".
    "all" is every query, and "rest" every query after the first `limit`."""
    head = min(count, limit)
    ids = {
        "tune": lambda: np.arange(0, head, 2),
        "test": lambda: np.arange(1, head, 2),
        "all": lambda: np.arange(count),
        "rest": lambda: np.arange(head, count),
    }

    if split not in ids:
        raise ValueError(f"unknown split {split}, expected tune, test, all or rest")

    return ids[split]().astype(np.int64)


def clusters(stem: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """A shared partition of the base, <stem>.centroids.fbin with one row per cluster and <stem>.assign.ibin with each
    base row's cluster."""
    centroids, assign = fbin(f"{stem}.centroids.fbin"), ibin(f"{stem}.assign.ibin")

    if assign.shape[1] != 1 or (len(assign) and int(np.max(assign)) >= len(centroids)):
        raise ValueError(f"{stem}.assign.ibin should hold one cluster below {len(centroids)} per row")

    return centroids, assign[:, 0]


def dist_path(truth_path: Path | str) -> Path:
    """The distances file that goes with an .ibin file of ground truth."""
    path = Path(truth_path)
    name = path.name

    if name.endswith(".gt.ibin"):
        return path.with_name(name[: -len(".gt.ibin")] + ".gt.dist.fbin")

    return path.with_name(path.stem + ".dist.fbin")


@dataclass
class Truth:
    """Each query's true nearest base rows, nearest first, among the first `n` rows of the base."""

    path: Path
    ids: np.ndarray
    n: int
    dists: np.ndarray | None = None
    # True when the distances file holds similarities, which shrink along a row, so they are negated before use.
    scores: bool = False

    @property
    def k(self) -> int:
        return int(self.ids.shape[1])

    @property
    def has_dist(self) -> bool:
        return self.dists is not None

    def record(self) -> dict:
        """The "gt" field of a result line."""
        return {"path": str(self.path), "k": self.k, "has_dist": self.has_dist}


def _descending(rows: np.ndarray) -> bool:
    """Whether the values of a distances file shrink along each row, as similarity scores do."""
    sample = np.asarray(rows[: min(16, len(rows))], dtype=np.float64)

    if sample.size == 0 or sample.shape[1] < 2:
        return False

    return bool(np.all(sample[:, 0] >= sample[:, -1]) and np.any(sample[:, 0] > sample[:, -1]))


@dataclass
class Dataset:
    """A dataset's base vectors and queries, from <root>/<name>.base.fbin and <root>/<name>.query.fbin."""

    name: str
    root: Path
    derived: Path = field(default_factory=derived_root)
    metric: str = ""
    _base: np.ndarray | None = field(default=None, repr=False)
    _queries: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.derived = Path(self.derived)
        self.metric = self.metric or metric_of(self.name)

    @classmethod
    def named(cls, name: str, root: Path | str | None = None, derived: Path | str | None = None) -> Dataset:
        return cls(name, Path(root) if root else data_root(), Path(derived) if derived else derived_root())

    @classmethod
    def from_base(cls, path: Path | str, derived: Path | str | None = None) -> Dataset:
        """The dataset whose base file is `path`, which must end with .base.fbin."""
        path = Path(path)

        if not path.name.endswith(".base.fbin"):
            raise ValueError(f"{path} isn't a <name>.base.fbin file")

        name = path.name[: -len(".base.fbin")]
        return cls(name, path.parent, Path(derived) if derived else derived_root())

    @property
    def base_path(self) -> Path:
        return self.root / f"{self.name}.base.fbin"

    @property
    def query_path(self) -> Path:
        return self.root / f"{self.name}.query.fbin"

    @property
    def base(self) -> np.ndarray:
        if self._base is None:
            self._base = fbin(self.base_path)

        return self._base

    @property
    def queries(self) -> np.ndarray:
        if self._queries is None:
            self._queries = fbin(self.query_path)

        return self._queries

    @property
    def count(self) -> int:
        return header(self.base_path)[0]

    @property
    def dim(self) -> int:
        return header(self.base_path)[1]

    def prefix(self, n: int) -> np.ndarray:
        """Rows 0 to n of the base, as a view of the one mapped file."""
        return self.base[: min(n, self.count)]

    def truth_path(self, n: int | None = None) -> Path:
        n = self.count if n is None else n

        if n >= self.count:
            return self.root / f"{self.name}.gt.ibin"

        return self.derived / f"{self.name}-{n}.gt.ibin"

    def truth(self, n: int | None = None, path: Path | str | None = None) -> Truth:
        """The ground truth of the first `n` base rows, with its distances when the file is there."""
        n = min(self.count if n is None else n, self.count)
        path = Path(path) if path else self.truth_path(n)
        ids = ibin(path)

        if ids.shape[0] != header(self.query_path)[0]:
            raise ValueError(f"{path} has {ids.shape[0]} rows, not one per query")

        dpath = dist_path(path)
        dists = fbin(dpath) if dpath.exists() else None

        if dists is not None and dists.shape != ids.shape:
            raise ValueError(f"{dpath} doesn't match the shape of {path}")

        return Truth(path, ids, n, dists, dists is not None and _descending(dists))


def _clean(found: np.ndarray, n: int, k: int) -> list[int]:
    """The first `k` distinct ids in a row of results that name a base row, in order."""
    out: list[int] = []

    for i in np.asarray(found).ravel().tolist():
        if 0 <= i < n and i not in out:
            out.append(int(i))

        if len(out) == k:
            break

    return out


class Recall:
    """Scores results against the ground truth. A result is a hit if its id is among the true k nearest, or if its
    exact distance ties or beats the true k-th distance, within TIE_RTOL. Both distances are computed here in float64
    from the base rows, so a duplicate of the k-th neighbor ties it exactly. A distances file, when there is one,
    checks that the truth belongs to this base, since its float32 values would need a tolerance wide enough to count
    near ties.
    """

    def __init__(self, dataset: Dataset, truth: Truth, rtol: float = TIE_RTOL):
        self.dataset = dataset
        self.truth = truth
        self.rtol = rtol
        self._kth: dict[tuple[int, int], float] = {}

    def kth(self, q: int, k: int) -> float:
        """The exact distance of query `q`'s true k-th neighbor."""
        key = (q, k)

        if key not in self._kth:
            row = self.dataset.base[int(self.truth.ids[q, k - 1])]
            exact = float(distances(self.dataset.metric, self.dataset.queries[q], row)[0])
            self._kth[key] = exact

            if self.truth.dists is not None:
                value = float(self.truth.dists[q, k - 1])
                value = -value if self.truth.scores else value

                if abs(value - exact) > DIST_CHECK_RTOL * max(1.0, abs(exact)):
                    raise ValueError(f"{dist_path(self.truth.path)} says query {q}'s neighbor {k} is at {value}, but "
                                     f"the base puts it at {exact}, so the truth doesn't belong to this base")

        return self._kth[key]

    def _tol(self, kth: float) -> float:
        return self.rtol * max(1.0, abs(kth))

    def hits(self, q: int, found: np.ndarray, k: int) -> tuple[int, int]:
        """The tie-aware hits and the plain id hits of one query's results among its true `k` nearest."""
        true = set(int(i) for i in self.truth.ids[q, :k])
        ids = _clean(found, self.truth.n, k)
        plain = sum(1 for i in ids if i in true)
        rest = [i for i in ids if i not in true]

        if not rest:
            return plain, plain

        kth = self.kth(q, k)
        d = distances(self.dataset.metric, self.dataset.queries[q], self.dataset.base[np.array(rest)])
        tied = int(np.count_nonzero(d <= kth + self._tol(kth)))
        return min(plain + tied, k), plain

    def per_query(self, qids: np.ndarray, found: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Each query's tie-aware recall@k and its plain id recall@k. Row i of `found` answers query qids[i]."""
        found = np.asarray(found)
        ties = np.zeros(len(qids))
        plain = np.zeros(len(qids))

        for i, q in enumerate(np.asarray(qids).tolist()):
            hit, ids_hit = self.hits(int(q), found[i], k)
            ties[i], plain[i] = hit / k, ids_hit / k

        return ties, plain

    def tie_free(self, qids: np.ndarray, k: int = 10) -> np.ndarray:
        """Whether each query's (k+1)-th true neighbor is farther than its k-th by more than the tie tolerance. For
        such a query the tie-aware recall@k equals the plain id recall@k."""
        out = np.zeros(len(qids), dtype=bool)

        if self.truth.k <= k:
            return out

        for i, q in enumerate(np.asarray(qids).tolist()):
            kth = self.kth(int(q), k)
            row = self.dataset.base[int(self.truth.ids[q, k])]
            nxt = float(distances(self.dataset.metric, self.dataset.queries[q], row)[0])
            out[i] = nxt > kth + self._tol(kth)

        return out

    def summary(self, qids: np.ndarray, found: np.ndarray) -> dict:
        """The recall fields of a result line for results `found`, one row per query of `qids`."""
        found = np.asarray(found)
        width = found.shape[1] if found.ndim == 2 else 0
        r10, ids10 = self.per_query(qids, found, 10)
        r1, _ = self.per_query(qids, found, 1)
        out = {
            "recall10": float(r10.mean()) if len(r10) else None,
            "recall10_ids_only": float(ids10.mean()) if len(ids10) else None,
            "recall1": float(r1.mean()) if len(r1) else None,
            "recall100": None,
            "recall10_p05": float(np.percentile(r10, 5, method="inverted_cdf")) if len(r10) else None,
            "recall10_p10": float(np.percentile(r10, 10, method="inverted_cdf")) if len(r10) else None,
            "share_ge_0_9": float(np.mean(r10 >= 0.9)) if len(r10) else None,
        }

        if width >= 100 and self.truth.k >= 100:
            out["recall100"] = float(self.per_query(qids, found, 100)[0].mean())

        return out
