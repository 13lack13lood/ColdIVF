"""The flat adapter with failures on demand, chosen by the variant, for the driver's tests.

- "crash" raises inside every cold trial.
- "selfreject" reports every cold trial as rejected, as a binary that finds cached pages at its timer does.
- "noc1" offers no C1 regime.
- "noisy" prints to stdout while it loads, which must not reach the wire.
- "exec" hands its cold subcommand to a small program that speaks the wire itself.
- "procqps" measures QPS with one process per worker.
- "loaded" marks every file as loaded at open, though it maps them, so its cached pages show apart.
- "swapping" reports 2 swap-ins during every warm setting, as a machine that swaps while it is timed would.
- "session" writes a log into its index while a warm run has it open and removes it at the end, as AsterVec's store
  does at every open until its seal is restored.

Its writes are the harness's own, as the exact reference's are, and a search scans the inserted rows through the
mask of live ids. Its maintenance action "compact" folds the inserted rows into the build's file, "recompact" does the
same, so their files match, and its policy calls for a compaction once "pending" inserts wait. For the write
benchmark:
- "whole" has no sync that writes only what changed, so a latency run times whole saves.
- "inserts" has no delete, as an inserts-only system.
- "writecrash" dies like a killed process at the insert of the row that the file crash.once beside the system's index
  folders names, and removes the file first, so the round's next try runs through.
- "misnumber" numbers its inserts itself, one past their rows, which a stream session must refuse.
- "inplace" says it writes its files in place, as the exact reference does, so the driver gives its sessions plain
  copies. Every other variant says it doesn't, so its sessions and states are clones.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from adapters.base import ServeContext, Unsupported, main
from adapters.flat import Flat
from adapters.inproc import HarnessWrites, LiveRows

# A program that speaks the wire on its own, as a native binary could.
PROGRAM = """
import json, os, sys
print(json.dumps({"event": "ready", "pid": os.getpid()}), flush=True)
go = json.loads(sys.stdin.readline())
fields = {"total_us": 1.5, "ids": [7, 8], "extra": {"program": True}}
print(json.dumps({"event": "trial", "trial": go["trial"], "fields": fields}), flush=True)
"""


class Fake(HarnessWrites, Flat):
    system = "fake"
    write_role = "full"
    maintenance_actions = ("compact", "recompact")
    writes_in_place = False

    def load(self) -> None:
        if self.variant == "inplace":
            self.writes_in_place = True

        if self.variant == "noisy":
            print("noise on stdout from a library")

        if self.variant == "procqps":
            self.qps_mode = "processes"

        if self.variant == "loaded":
            self.loads_at_open = True

        if self.variant == "whole":
            self.sync_kind = "whole"

        if self.variant == "inserts":
            self.write_role = "inserts"

    def supported_regimes(self) -> tuple[str, ...]:
        return ("c0", "intent") if self.variant == "noc1" else self.regimes

    def cold_setup(self, ctx: ServeContext) -> None:
        if self.variant == "exec":
            print("this line goes to stderr, not the wire")
            self.exec_program([sys.executable, "-c", PROGRAM])

        super().cold_setup(ctx)

    def open(self, ctx: ServeContext, path: Path) -> Any:
        handle = super().open(ctx, path)
        handle["inserts"], handle["live"] = LiveRows.view(path)
        return handle

    def search(self, handle: Any, query: np.ndarray, k: int, params: dict) -> np.ndarray:
        inserts, live = handle.get("inserts"), handle.get("live")

        if (inserts is None or not len(inserts)) and live is None:
            return super().search(handle, query, k, params)

        q = np.asarray(query, dtype=np.float32)
        blocks = [handle["vectors"]] + ([inserts] if inserts is not None and len(inserts) else [])
        scores = np.concatenate([score(np.asarray(b), q, handle["metric"]) for b in blocks])

        if live is not None:
            scores[~live[: len(scores)]] = np.inf

        k = min(k, int(np.count_nonzero(np.isfinite(scores))))
        top = np.argpartition(scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        return top[np.lexsort((top, scores[top]))].astype(np.int64)

    def warm_run(self, ctx: ServeContext, *args, **kwargs) -> Iterator[dict]:
        session = ctx.index / "session.log"

        if self.variant == "session":
            session.write_bytes(b"open")

        try:
            for result in super().warm_run(ctx, *args, **kwargs):
                if self.variant == "swapping":
                    result["fields"].update(swapins_delta=2, compressions_delta=0)

                yield result

                # The session stays open while the driver writes the setting's line.
                if self.variant == "session":
                    time.sleep(1.0)
        finally:
            if self.variant == "session":
                session.unlink(missing_ok=True)

    def cold_trial(self, ctx: ServeContext, go: dict) -> dict:
        if self.variant == "crash":
            raise RuntimeError("a crash on purpose")

        fields = super().cold_trial(ctx, go)

        if self.variant == "selfreject":
            fields.update(status="rejected", error="rejected on purpose")

        return fields

    def insert(self, w: LiveRows, ids: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
        marker = w.folder.parent / "crash.once"

        if self.variant == "writecrash" and marker.exists() and int(marker.read_text()) in ids.tolist():
            marker.unlink()
            os._exit(9)

        got = super().insert(w, ids, rows)
        return got + 1 if self.variant == "misnumber" else got

    def delete(self, w: LiveRows, ids: np.ndarray) -> None:
        if self.variant == "inserts":
            raise Unsupported("the inserts-only fake has no delete")

        super().delete(w, ids)

    def due(self, w: LiveRows, params: dict) -> bool:
        return w.inserted >= int(params.get("pending", 1000))

    def pending(self, w: LiveRows) -> dict:
        return {"inserted": w.inserted, "base_rows": w.base_rows}

    def maintain(self, w: LiveRows, action: str, threads: int, params: dict) -> dict:
        """Folds the inserted rows into vectors.npy, so ids stay rows, and leaves the mask of live ids as it is."""
        if action not in self.maintenance_actions:
            raise Unsupported(f"the fake has no action {action}")

        folder = w.folder
        base = np.load(folder / "vectors.npy", mmap_mode="r")
        inserts, live = LiveRows.view(folder)
        rows = np.concatenate([np.asarray(base), np.asarray(inserts)]) if len(inserts) else np.asarray(base)
        tmp = folder / "vectors.npy.tmp"

        with open(tmp, "wb") as f:
            np.save(f, rows)

        os.replace(tmp, folder / "vectors.npy")
        meta = json.loads((folder / "meta.json").read_text())
        (folder / "meta.json").write_text(json.dumps(meta | {"rows": len(rows)}))
        os.ftruncate(w.ins_fd, 8)
        os.pwrite(w.ins_fd, struct.pack("<ii", 0, w.dim), 0)
        folded, w.base_rows, w.inserted = w.inserted, len(rows), 0
        w.sync()
        return {"rows_live": int(np.count_nonzero(live)), "durability": "power",
                "data_file": str(folder / "vectors.npy"), "folded": folded}


def score(block: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    """Each row's distance to the query, smaller is nearer, as the flat scan ranks them."""
    if metric == "l2":
        diff = block - query
        return np.einsum("ij,ij->i", diff, diff)

    out = -(block @ query)

    if metric == "cosine":
        norms = np.linalg.norm(block, axis=1)
        out = out / np.where(norms > 0, norms, 1.0)

    return out.astype(np.float32)


if __name__ == "__main__":
    main(Fake)
