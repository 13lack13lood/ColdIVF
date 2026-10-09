"""sqlite-vec's vec0 table in SQLite, as an exact scan of float32 vectors, or as a scan of int8 or bit codes whose best
candidates are rescored with the float32 vectors.

A build writes vec0.db, a database with one vec0 table. Its `embedding` column holds the float32 rows, which the exact
scan reads and which rescore the candidates of a coarse scan. The config's `codes` adds a column of int8 codes, of bit
codes or of both, eg. ["int8", "bit"], made by sqlite-vec's own vec_quantize_int8(v, 'unit') and
vec_quantize_binary(v). `page_size` and `chunk_size` set SQLite's page size and vec0's vectors per chunk, and SQLite's
and vec0's defaults hold unless given. Distances are L2, which ranks unit vectors as the cosine does and runs with NEON,
where sqlite-vec's cosine doesn't. meta.json beside the database names its columns, and a cold trial reads it before
READY, as an application knows its own schema.

`rescore` says where the float32 rows live. With "vec0", the default and sqlite-vec's own example, they are a vec0
column, which the exact scan needs. SQLite keeps each vec0 chunk of vectors as one value spread over a chain of pages,
so rescoring a candidate reads the chain up to it. One cold query at 50K rows with int8 codes and an oversample of 4
read 107 MB of a 195 MB database that way. With "aux" they are an auxiliary column, one row per vector, and the same
query read 53 MB. Such a database has no exact scan.

A search takes `codes`, the column it scans, which is "f32" for the exact scan and the build's coarse codes by default,
and `oversample`. A coarse scan keeps k·oversample candidates, where k is the number of results, and their float32
distances rank them. `mmap_size` and `cache_size` set those pragmas on the connection. The timed first call connects
read-only, loads the extension and runs the query. The extension is also loaded once at start, so its library is
already mapped, as in any process that has used it before. A connection serves one thread, so queries per second (QPS)
at several threads run one process per worker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from adapters import inproc
from adapters.base import Adapter, BuildContext, ServeContext, Unsupported, main
import schema

DB = "vec0.db"
META = "meta.json"
TABLE = "vectors"
CODES = ("f32", "int8", "bit")
COLUMNS = {"f32": "embedding", "int8": "embedding_int8", "bit": "embedding_bit"}
TYPES = {"f32": "float", "int8": "int8", "bit": "bit"}
QUANTIZE = {"int8": "vec_quantize_int8(?, 'unit')", "bit": "vec_quantize_binary(?)"}
# Rows per transaction of the build.
BATCH = 10000
RESCORE = ("vec0", "aux")
BUILD = {"codes": "f32", "page_size": None, "chunk_size": None, "rescore": "vec0"}
SEARCH = {"codes": None, "oversample": 1, "mmap_size": None, "cache_size": None}
PRAGMAS = ("mmap_size", "cache_size")


def code_list(value: Any) -> list[str]:
    """A build's coarse codes, from one name or a list of names, in a fixed order. "f32" alone means none, since the
    float32 column is always there."""
    names = [value] if isinstance(value, str) else list(value)
    unknown = sorted(set(names) - set(CODES))

    if unknown:
        raise ValueError(f"unknown codes {unknown}, expected some of {list(CODES)}")

    return [c for c in CODES[1:] if c in names]


def column_bytes(code: str, dim: int) -> int:
    """The bytes of one vector in the column of `code`."""
    return {"f32": 4 * dim, "int8": dim, "bit": -(-dim // 8)}[code]


def create_sql(dim: int, codes: list[str], chunk_size: Any = None, rescore: str = "vec0") -> str:
    """The statement that makes the vec0 table, with the float32 column and one column per coarse code. The float32
    column is a vec0 column, or an auxiliary column of the same name for `rescore` "aux"."""
    columns = [f"{COLUMNS[c]} {TYPES[c]}[{dim}]" for c in (["f32", *codes] if rescore == "vec0" else codes)]

    if rescore == "aux":
        columns.append(f"+{COLUMNS['f32']} blob")

    if chunk_size is not None:
        columns.append(f"chunk_size={int(chunk_size)}")

    return f"create virtual table {TABLE} using vec0({', '.join(columns)})"


def insert_sql(codes: list[str]) -> str:
    """The statement that adds one row. It takes the row number, then the float32 vector once per column."""
    columns = ["rowid"] + [COLUMNS[c] for c in ["f32", *codes]]
    values = ["?", "?"] + [QUANTIZE[c] for c in codes]
    return f"insert into {TABLE}({', '.join(columns)}) values ({', '.join(values)})"


def query_sql(code: str) -> str:
    """The query for a scan of the column of `code`. `query_args` gives its parameters."""
    if code == "f32":
        return f"select rowid from {TABLE} where {COLUMNS['f32']} match ? and k = ? order by distance"

    return (f"with coarse as (select rowid, {COLUMNS['f32']} from {TABLE} where {COLUMNS[code]} match "
            f"{QUANTIZE[code]} and k = ?) select rowid from coarse order by vec_distance_l2({COLUMNS['f32']}, ?) "
            f"limit ?")


def query_args(code: str, vector: bytes, k: int, oversample: int) -> tuple:
    """The parameters of `query_sql(code)`. They are the vector and k for the exact scan, and the vector, the
    candidate count, the vector again and k for a coarse scan."""
    if code == "f32":
        return (vector, k)

    return (vector, k * oversample, vector, k)


def scan_counts(code: str, rows: int, dim: int, k: int, oversample: int) -> dict:
    """The distances one search computes and the vector bytes it reads. Every scan reads every row of its column, and
    a coarse scan then reads the float32 vectors of its candidates."""
    if code == "f32":
        return {"dist_evals": rows, "bytes_scanned": rows * column_bytes("f32", dim)}

    candidates = min(rows, k * oversample)
    return {"dist_evals": rows + candidates,
            "bytes_scanned": rows * column_bytes(code, dim) + candidates * column_bytes("f32", dim)}


def search_settings(params: dict, codes: list[str], rescore: str = "vec0") -> dict:
    """The search settings with their defaults, for a database with coarse `codes` and float32 rows kept as `rescore`
    says."""
    out = inproc.settings(params, SEARCH, "sqlite-vec search settings")
    out["codes"] = out["codes"] or (codes[0] if codes else "f32")

    if out["codes"] != "f32" and out["codes"] not in codes:
        raise ValueError(f"the database has no {out['codes']} codes, only float32 and {codes}")

    if out["codes"] == "f32" and rescore == "aux":
        raise ValueError("the database keeps its float32 rows in an auxiliary column, which has no exact scan")

    if int(out["oversample"]) < 1:
        raise ValueError("oversample must be at least 1")

    return out


@dataclass
class Handle:
    """An open database, its meta.json, and the scan that searches run."""

    db: Any
    meta: dict
    params: dict | None = None
    code: str = "f32"
    oversample: int = 1
    sql: str = ""
    k: int = 10


class SqliteVec(Adapter):
    system = "sqlite-vec"
    binding = "python"
    qps_mode = "processes"

    def __init__(self) -> None:
        super().__init__()
        self.metas: dict[str, dict] = {}
        self.keep = None

    def load(self) -> None:
        import sqlite3

        import sqlite_vec

        self.sqlite3, self.vec = sqlite3, sqlite_vec
        # This connection keeps the extension's library loaded for the life of the process.
        self.keep = self.connect(":memory:")

    def connect(self, target: str, uri: bool = False) -> Any:
        db = self.sqlite3.connect(target, uri=uri)
        db.enable_load_extension(True)
        self.vec.load(db)
        db.enable_load_extension(False)
        return db

    def version(self) -> str:
        return f"{self.vec.__version__}, SQLite {self.sqlite3.sqlite_version}"

    def wheels(self) -> dict[str, str | None]:
        return {"sqlite-vec": schema.dist_sha256("sqlite-vec")}

    def simd_level(self) -> str | None:
        if self.keep is None:
            return None

        text = self.keep.execute("select vec_debug()").fetchone()[0]
        return text.split("Build flags:")[-1].strip() or "none"

    def overhead_config(self) -> dict:
        return {"codes": "f32"}

    def build(self, ctx: BuildContext) -> dict:
        cfg = inproc.settings(ctx.config, BUILD, "sqlite-vec build settings")
        codes, rescore = code_list(cfg["codes"]), str(cfg["rescore"])

        if rescore not in RESCORE:
            raise ValueError(f"rescore is {rescore}, not one of {list(RESCORE)}")

        if rescore == "aux" and not codes:
            raise ValueError("rescore aux needs int8 or bit codes, since an auxiliary column has no exact scan")

        if ctx.metric == "ip":
            raise Unsupported("sqlite-vec has no inner-product distance")

        if ctx.metric == "cosine":
            inproc.check_unit(ctx.data)

        db = self.connect(str(ctx.out / DB))

        if cfg["page_size"] is not None:
            db.execute(f"pragma page_size = {int(cfg['page_size'])}")

        db.execute(create_sql(ctx.dim, codes, cfg["chunk_size"], rescore))
        sql, copies = insert_sql(codes), 1 + len(codes)
        start = time.perf_counter()

        for lo in range(0, ctx.rows, BATCH):
            block = np.ascontiguousarray(ctx.data[lo : lo + BATCH], dtype=np.float32)

            with db:
                db.executemany(sql, ((lo + i, *[row.tobytes()] * copies) for i, row in enumerate(block)))

        encode = time.perf_counter() - start
        page = int(db.execute("pragma page_size").fetchone()[0])
        db.close()

        # SQLite ignores a page size it doesn't take, eg. one that isn't a power of 2.
        if cfg["page_size"] is not None and page != int(cfg["page_size"]):
            raise ValueError(f"SQLite kept pages of {page} bytes, not the {cfg['page_size']} asked for")

        meta = {"rows": ctx.rows, "dim": ctx.dim, "codes": codes, "rescore": rescore, "page_size": page,
                "chunk_size": cfg["chunk_size"], "distance": "l2"}
        (ctx.out / META).write_text(json.dumps(meta))
        return {
            "encode_seconds": encode,
            "stored_vector_bytes": ctx.rows * sum(column_bytes(c, ctx.dim) for c in ["f32", *codes]),
            "exact_f32_stored": True,
            "extra": meta | {"sqlite": self.sqlite3.sqlite_version},
        }

    def meta(self, path: Path) -> dict:
        """The meta.json of the index at `path`, read once."""
        key = str(path)

        if key not in self.metas:
            self.metas[key] = json.loads((Path(path) / META).read_text())

        return self.metas[key]

    def apply(self, handle: Handle, params: dict) -> None:
        """Checks the search settings, sets their pragmas and picks the scan."""
        s = search_settings(params, handle.meta["codes"], handle.meta.get("rescore", "vec0"))

        for name in PRAGMAS:
            if s[name] is not None:
                handle.db.execute(f"pragma {name} = {int(s[name])}")

        handle.params, handle.code, handle.oversample = params, s["codes"], int(s["oversample"])
        handle.sql = query_sql(handle.code)

    def open(self, ctx: ServeContext, path: Path) -> Handle:
        meta = self.meta(path)
        handle = Handle(self.connect(f"file:{quote(str(Path(path) / DB))}?mode=ro", uri=True), meta)
        self.apply(handle, ctx.params)
        return handle

    def configure(self, handle: Handle, params: dict) -> None:
        self.apply(handle, params)

    def cold_setup(self, ctx: ServeContext) -> None:
        meta = self.meta(ctx.index)
        search_settings(ctx.params, meta["codes"], meta.get("rescore", "vec0"))

        if ctx.dummy is not None:
            self.meta(ctx.dummy)

        super().cold_setup(ctx)

    def search(self, handle: Handle, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        if params is not handle.params:
            self.apply(handle, params)

        vector = np.ascontiguousarray(query, dtype=np.float32).tobytes()
        found = handle.db.execute(handle.sql, query_args(handle.code, vector, k, handle.oversample)).fetchall()
        handle.k = k
        return inproc.as_ids([row[0] for row in found], k)

    def stats(self, handle: Handle) -> dict:
        return scan_counts(handle.code, handle.meta["rows"], handle.meta["dim"], handle.k, handle.oversample)

    def close(self, handle: Handle) -> None:
        handle.db.close()


if __name__ == "__main__":
    main(SqliteVec)
