"""faiss index files written byte for byte, so the faiss tests can check the reader and the adapters' file ranges
without faiss."""

from __future__ import annotations

import struct

import numpy as np


def vec(values: np.ndarray) -> bytes:
    """A faiss array, its item count and then its items."""
    values = np.ascontiguousarray(values)
    return struct.pack("<Q", values.size) + values.tobytes()


def header(d: int, n: int, metric: int = 1) -> bytes:
    return struct.pack("<iqqqBi", d, n, 1 << 20, 1 << 20, 1, metric)


def flat(d: int, rows: np.ndarray) -> bytes:
    rows = rows.astype("<f4")
    return b"IxF2" + header(d, len(rows)) + vec(rows)


def sq8(d: int, codes: np.ndarray) -> bytes:
    trained = np.zeros(2 * d, dtype="<f4")
    fields = struct.pack("<iifQQ", 1, 0, 0.0, d, d)
    return b"IxSQ" + header(d, len(codes)) + fields + vec(trained) + vec(codes.astype(np.uint8))


def rotation(d: int) -> bytes:
    a = np.eye(d, dtype="<f4")
    return b"rrot" + b"\x00" + vec(a) + vec(np.zeros(0, dtype="<f4")) + struct.pack("<iiB", d, d, 1)


def ilar(code_size: int, lists: list[np.ndarray], sparse: bool = False) -> bytes:
    """Array inverted lists, where lists[i] holds list i's ids. A code is the id's low byte repeated."""
    sizes = np.array([len(x) for x in lists], dtype="<u8")

    if sparse:
        pairs = np.array([(i, n) for i, n in enumerate(sizes) if n], dtype="<u8").ravel()
        table = b"sprs" + vec(pairs)
    else:
        table = b"full" + vec(sizes)

    body = b"".join(codes_of(ids, code_size) + ids.astype("<i8").tobytes() for ids in lists if len(ids))
    return b"ilar" + struct.pack("<QQ", len(lists), code_size) + table + body


def codes_of(ids: np.ndarray, code_size: int) -> bytes:
    return np.repeat((ids % 251).astype(np.uint8), code_size).tobytes()


def ivf_rabitq(d: int, centroids: np.ndarray, lists: bytes, n: int, code_size: int) -> bytes:
    nlist = len(centroids)
    head = b"Iwrq" + header(d, n) + struct.pack("<QQ", nlist, 1) + flat(d, centroids)
    direct = b"\x00" + vec(np.zeros(0, dtype="<i8"))
    rabitq = struct.pack("<QQi", d, code_size, 1)
    return head + direct + rabitq + struct.pack("<QBB", code_size, 1, 4) + lists


def pretransform(d: int, n: int, sub: bytes) -> bytes:
    return b"IxPT" + header(d, n) + struct.pack("<i", 1) + rotation(d) + sub


def refine(d: int, n: int, base: bytes, rows: bytes) -> bytes:
    return b"IxRF" + header(d, n) + base + rows + struct.pack("<f", 1.0)


def graph(n: int, degree: int) -> tuple[bytes, dict]:
    levels = np.ones(n, dtype="<i4")
    offsets = (np.arange(n + 1) * degree).astype("<u8")
    neighbors = np.arange(n * degree, dtype="<i4") % n
    body = (vec(np.array([0.5, 0.5], dtype="<f8")) + vec(np.array([0, degree], dtype="<i4")) + vec(levels)
            + vec(offsets) + vec(neighbors) + struct.pack("<iiiii", 0, 0, 40, 16, 1))
    return body, {"levels": levels, "offsets": offsets, "neighbors": neighbors}


def find(raw: bytes, arr: np.ndarray) -> int:
    at = raw.find(np.ascontiguousarray(arr).tobytes())
    assert at >= 0
    return at
