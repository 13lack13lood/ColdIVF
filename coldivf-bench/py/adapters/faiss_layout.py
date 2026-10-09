"""Where the parts of a faiss index file sit, read from the file's headers without faiss.

faiss writes an index as nested records. Each record starts with a four-character code (fourcc) that names its class,
then holds the class's fields in order. An array is a 64-bit item count followed by the items. This module reads the
records of the index types the faiss adapters write and seeks past every array, so even a large file costs only its
headers and its table of list sizes. The result says where each array sits, eg. the coarse centroids, each inverted
list and the 8-bit rows of the refine index. The adapters use it to tell the driver which bytes are bulk data and which
are routing data, and the batched variant uses it to read lists and rows ahead.

A part's name says what it holds.

| Name | Holds | Kind |
|---|---|---|
| rotation | The random rotation's matrix and bias | routing |
| coarse | The coarse centroids of an inverted file (IVF) index | routing |
| coarse_graph | The graph of a coarse quantizer that is itself a graph | routing |
| list_sizes | The table of inverted list sizes | routing |
| lists | The codes and ids of every inverted list | bulk |
| graph | The levels, offsets and neighbor lists of a graph index | routing |
| storage | The codes a graph index scores its nodes with | bulk |
| refine | The rows a refine index rescores candidates with | bulk |
| codes | The codes of a plain index, eg. a standalone 8-bit index | bulk |

Every byte not in a part, eg. a record's header, is routing data.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

ROUTING = "routing"
BULK = "bulk"

KINDS = {
    "rotation": ROUTING, "coarse": ROUTING, "coarse_graph": ROUTING, "list_sizes": ROUTING, "lists": BULK,
    "graph": ROUTING, "storage": BULK, "refine": BULK, "codes": BULK,
}

# The name of a flat index's codes for each role a record can play in its parent.
CODES_BY_ROLE = {"top": "codes", "coarse": "coarse", "refine": "refine", "storage": "storage"}

# Fourccs of flat indexes of float rows, by metric.
FLAT = {"IxF2", "IxFI", "IxFl"}
HNSW = {"IHNf", "IHNs", "IHNr", "IH00"}
LINEAR = {"rrot", "LTra"}

# Direct map types, as faiss numbers them. Only a hash table stores pairs after the array.
DIRECT_MAP_HASHTABLE = 2

# An on-disk list's entry in the index file holds its size, its capacity and its offset, 8 bytes each.
ONDISK_LIST_BYTES = 24
ONDISK_SLOT_BYTES = 16


@dataclass
class Part:
    """One array of the file, or the span of every inverted list."""

    name: str
    offset: int
    length: int

    @property
    def kind(self) -> str:
        return KINDS[self.name]

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass
class Lists:
    """The inverted lists of an IVF index. List i holds sizes[i] entries, its codes at byte codes[i] and its ids at
    byte ids[i]. They sit in the index file itself, or in the on-disk lists file `file` when that is set."""

    code_size: int
    sizes: np.ndarray
    codes: np.ndarray
    ids: np.ndarray
    file: str | None = None
    # Where the table of sizes sits in the index file, and how faiss stored it, "full" or "sprs" (sparse).
    sizes_offset: int | None = None
    sizes_type: str = "full"

    @property
    def nlist(self) -> int:
        return len(self.sizes)

    def spans(self, lists: np.ndarray) -> list[tuple[int, int]]:
        """The byte ranges, as (offset, length), that hold the codes and ids of `lists`. Lists stored in the index
        file keep their ids right after their codes, so each list is one range there."""
        lists = np.asarray(lists, dtype=np.int64)
        lists = lists[(lists >= 0) & (lists < self.nlist)]
        n = self.sizes[lists]
        lists, n = lists[n > 0], n[n > 0]

        if self.file is None:
            return list(zip(self.codes[lists].tolist(), (n * (self.code_size + 8)).tolist()))

        out = []

        for c, i, m in zip(self.codes[lists].tolist(), self.ids[lists].tolist(), n.tolist()):
            out += [(c, m * self.code_size), (i, m * 8)]

        return out


@dataclass
class Layout:
    """The parts of one index file, its size, and its inverted lists if it has any."""

    path: str
    size: int
    parts: list[Part] = field(default_factory=list)
    lists: Lists | None = None
    # The row width of each part made of fixed-size rows, eg. 768 bytes for 8-bit rows of 768 dimensions.
    row_bytes: dict[str, int] = field(default_factory=dict)
    # The fourcc of every record, outermost first.
    records: list[str] = field(default_factory=list)

    def part(self, name: str) -> Part | None:
        """The first part named `name`, or None."""
        return next((p for p in self.parts if p.name == name), None)

    def named(self, name: str) -> list[Part]:
        return [p for p in self.parts if p.name == name]

    def ranges(self, loaded: frozenset[str] = frozenset()) -> list[dict]:
        """Ranges that cover the whole file, each {offset, length, kind}. Bytes outside every part are routing data,
        and neighbors of the same kind merge. The parts named in `loaded`, which an open copies into memory, also get
        "loaded" set, and merge only with each other."""
        out: list[dict] = []
        at = 0

        for p in sorted((p for p in self.parts if p.length > 0), key=lambda p: p.offset):
            if p.offset > at:
                _extend(out, at, p.offset - at, ROUTING)

            _extend(out, p.offset, p.length, p.kind, p.name in loaded)
            at = max(at, p.end)

        if at < self.size:
            _extend(out, at, self.size - at, ROUTING)

        return out

    def record(self) -> dict:
        """A JSON form of the layout without the per-list tables, for an index's meta file."""
        out = {
            "size": self.size,
            "records": self.records,
            "parts": [{"name": p.name, "offset": p.offset, "length": p.length, "kind": p.kind} for p in self.parts],
            "row_bytes": dict(self.row_bytes),
        }

        if self.lists is not None:
            out["lists"] = {"nlist": self.lists.nlist, "code_size": self.lists.code_size, "file": self.lists.file,
                            "entries": int(self.lists.sizes.sum()), "sizes_offset": self.lists.sizes_offset,
                            "sizes_type": self.lists.sizes_type}

        return out


def _extend(out: list[dict], offset: int, length: int, kind: str, loaded: bool = False) -> None:
    last = out[-1] if out else None

    if (last is not None and last["kind"] == kind and bool(last.get("loaded")) == loaded
            and last["offset"] + last["length"] == offset):
        last["length"] += length
    else:
        out.append({"offset": offset, "length": length, "kind": kind} | ({"loaded": True} if loaded else {}))


class _Reader:
    """Reads fields at the file's position and seeks past arrays, checking that nothing runs past the end."""

    def __init__(self, f: BinaryIO, size: int):
        self.f = f
        self.size = size

    @property
    def at(self) -> int:
        return self.f.tell()

    def take(self, n: int) -> bytes:
        raw = self.f.read(n)

        if len(raw) != n:
            raise ValueError(f"the file ends at byte {self.at}, inside a field of {n} bytes")

        return raw

    def unpack(self, fmt: str) -> int | float:
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def u8(self) -> int:
        return int(self.unpack("B"))

    def i32(self) -> int:
        return int(self.unpack("i"))

    def u64(self) -> int:
        return int(self.unpack("Q"))

    def f32(self) -> float:
        return float(self.unpack("f"))

    def fourcc(self) -> str:
        return self.take(4).decode("latin-1")

    def skip(self, n: int) -> None:
        if n < 0 or self.at + n > self.size:
            raise ValueError(f"an array of {n} bytes at byte {self.at} runs past the end of the file")

        self.f.seek(n, 1)

    def vector(self, item: int) -> tuple[int, int]:
        """Skips an array of `item`-byte items, and returns where its items start and how many bytes they take."""
        count = self.u64()

        if item and count > self.size // item:
            raise ValueError(f"an array at byte {self.at} claims {count} items, more than the file holds")

        offset = self.at
        self.skip(count * item)
        return offset, count * item

    def values(self, dtype: str) -> tuple[int, np.ndarray]:
        """Reads an array of numbers, and returns where it starts and its values."""
        width = np.dtype(dtype).itemsize
        offset, length = self.vector(width)
        self.f.seek(offset)
        values = np.frombuffer(self.take(length), dtype=dtype).astype(np.int64)
        return offset, values


class _Walker:
    """Walks the records of one index file and collects its parts."""

    def __init__(self, path: Path, f: BinaryIO, size: int):
        self.r = _Reader(f, size)
        self.layout = Layout(path=str(path), size=size)
        self.dir = path.parent

    def add(self, name: str, offset: int, length: int) -> None:
        if length > 0:
            self.layout.parts.append(Part(name, offset, length))

    def header(self) -> tuple[int, int, int]:
        """Reads an index header, and returns the dimension, the vector count and the metric."""
        d = self.r.i32()
        ntotal = int(self.r.unpack("q"))
        self.r.skip(16)
        self.r.u8()
        metric = self.r.i32()

        if metric > 1:
            self.r.f32()

        return d, ntotal, metric

    def index(self, role: str) -> None:
        h = self.r.fourcc()
        self.layout.records.append(h)

        if h == "null":
            return

        if h in FLAT:
            d, _, _ = self.header()
            offset, length = self.r.vector(4)
            self.codes(role, offset, length, 4 * d)
        elif h == "IxSQ":
            self.header()
            code_size = self.scalar_quantizer()
            offset, length = self.r.vector(1)
            self.codes(role, offset, length, code_size)
        elif h in ("Ixrq", "Ixrr"):
            self.header()
            code_size = self.rabitq(multi_bit=h == "Ixrr")
            offset, length = self.r.vector(1)
            self.codes(role, offset, length, code_size)
            self.r.vector(4)
            self.r.u8()
        elif h in ("Iwrq", "Iwrr"):
            self.ivf_header()
            self.rabitq(multi_bit=h == "Iwrr")
            self.r.u64()
            self.r.u8()
            self.r.u8()
            self.inverted_lists()
        elif h == "IxPT":
            self.header()

            for _ in range(self.r.i32()):
                self.transform()

            self.index(role)
        elif h == "IxRF":
            self.header()
            self.index(role)
            self.index("refine")
            self.r.f32()
        elif h in HNSW:
            self.header()
            self.hnsw("coarse_graph" if role == "coarse" else "graph")
            self.index("coarse" if role == "coarse" else "storage")

            # A graph over RaBitQ codes ends with a flag that says whether its search is staged.
            if h == "IHNr":
                self.r.u8()
        else:
            raise ValueError(f"index record {h!r} at byte {self.r.at - 4} isn't one this reader knows")

    def codes(self, role: str, offset: int, length: int, row_bytes: int) -> None:
        name = CODES_BY_ROLE[role]
        self.add(name, offset, length)

        if row_bytes > 0:
            self.layout.row_bytes.setdefault(name, row_bytes)

    def scalar_quantizer(self) -> int:
        """Reads a scalar quantizer's fields, and returns its code size."""
        self.r.i32()
        self.r.i32()
        self.r.f32()
        self.r.u64()
        code_size = self.r.u64()
        self.r.vector(4)
        return code_size

    def rabitq(self, multi_bit: bool) -> int:
        """Reads a RaBitQ quantizer's fields, and returns its code size."""
        self.r.u64()
        code_size = self.r.u64()
        self.r.i32()

        if multi_bit:
            self.r.u64()

        return code_size

    def transform(self) -> None:
        h = self.r.fourcc()

        if h not in LINEAR:
            raise ValueError(f"vector transform {h!r} at byte {self.r.at - 4} isn't one this reader knows")

        self.r.u8()
        a = self.r.vector(4)
        b = self.r.vector(4)
        self.add("rotation", *a)
        self.add("rotation", *b)
        self.r.i32()
        self.r.i32()
        self.r.u8()

    def ivf_header(self) -> None:
        self.header()
        self.r.u64()
        self.r.u64()
        self.index("coarse")
        kind = self.r.u8()
        self.r.vector(8)

        if kind == DIRECT_MAP_HASHTABLE:
            self.r.vector(16)

    def hnsw(self, name: str) -> None:
        self.r.vector(8)
        self.r.vector(4)

        for item in (4, 8, 4):
            self.add(name, *self.r.vector(item))

        for _ in range(5):
            self.r.i32()

    def inverted_lists(self) -> None:
        h = self.r.fourcc()

        if h == "il00":
            return

        if h == "ilar":
            self.array_lists()
        elif h == "ilod":
            self.ondisk_lists()
        else:
            raise ValueError(f"inverted lists {h!r} at byte {self.r.at - 4} aren't ones this reader knows")

    def array_lists(self) -> None:
        nlist, code_size = self.r.u64(), self.r.u64()
        kind = self.r.fourcc()
        offset, values = self.r.values("<u8")
        sizes = np.zeros(nlist, dtype=np.int64)

        if kind == "full":
            if len(values) != nlist:
                raise ValueError(f"the list sizes table holds {len(values)} sizes for {nlist} lists")

            sizes[:] = values
        elif kind == "sprs":
            pairs = values.reshape(-1, 2)

            if len(pairs) and int(pairs[:, 0].max()) >= nlist:
                raise ValueError("the sparse list sizes name a list past the last one")

            sizes[pairs[:, 0]] = pairs[:, 1]
        else:
            raise ValueError(f"list sizes of type {kind!r} aren't ones this reader knows")

        self.add("list_sizes", offset, len(values) * 8)
        start = self.r.at
        lists = array_lists(start, sizes, code_size)
        total = int((sizes * (code_size + 8)).sum())
        self.r.skip(total)
        self.add("lists", start, total)
        lists.sizes_offset, lists.sizes_type = offset, kind
        self.layout.lists = lists

    def ondisk_lists(self) -> None:
        nlist, code_size = self.r.u64(), self.r.u64()
        offset, length = self.r.vector(ONDISK_LIST_BYTES)
        self.r.f.seek(offset)
        table = np.frombuffer(self.r.take(length), dtype="<u8").astype(np.int64).reshape(-1, 3)

        if len(table) != nlist:
            raise ValueError(f"the on-disk list table holds {len(table)} lists, not {nlist}")

        self.add("list_sizes", offset, length)
        self.r.vector(ONDISK_SLOT_BYTES)
        name_at, name_len = self.r.vector(1)
        self.r.f.seek(name_at)
        filename = self.r.take(name_len).decode()
        self.r.u64()
        sizes, capacity, codes = table[:, 0], table[:, 1], table[:, 2]
        self.layout.lists = Lists(code_size=code_size, sizes=sizes.copy(), codes=codes.copy(),
                                  ids=codes + capacity * code_size, file=filename, sizes_offset=offset,
                                  sizes_type="ondisk")


def array_lists(start: int, sizes: np.ndarray, code_size: int) -> Lists:
    """The lists that faiss stores in the index file from byte `start`, one list after another, each its codes then
    its ids. Empty lists take no bytes."""
    sizes = np.asarray(sizes, dtype=np.int64)
    step = sizes * (code_size + 8)
    codes = start + np.concatenate(([0], np.cumsum(step)[:-1])).astype(np.int64)
    return Lists(code_size=code_size, sizes=sizes, codes=codes, ids=codes + sizes * code_size)


def read(path: Path | str) -> Layout:
    """The layout of the faiss index file at `path`. It raises ValueError for a file it can't read to the end."""
    path = Path(path)
    size = path.stat().st_size

    with open(path, "rb") as f:
        walker = _Walker(path, f, size)
        walker.index("top")
        end = f.tell()

    if end != size:
        raise ValueError(f"{path} has {size - end} bytes after its last record, so its layout wasn't read right")

    return walker.layout


def ondisk_file(index_path: Path | str, layout: Layout) -> Path | None:
    """The on-disk lists file of an index, looked up beside the index file as faiss's IO_FLAG_ONDISK_SAME_DIR does."""
    if layout.lists is None or layout.lists.file is None:
        return None

    return Path(index_path).parent / Path(layout.lists.file).name
