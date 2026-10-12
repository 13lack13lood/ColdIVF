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
"""

from __future__ import annotations

import sys
import time
from typing import Iterator

from adapters.base import ServeContext, main
from adapters.flat import Flat

# A program that speaks the wire on its own, as a native binary could.
PROGRAM = """
import json, os, sys
print(json.dumps({"event": "ready", "pid": os.getpid()}), flush=True)
go = json.loads(sys.stdin.readline())
fields = {"total_us": 1.5, "ids": [7, 8], "extra": {"program": True}}
print(json.dumps({"event": "trial", "trial": go["trial"], "fields": fields}), flush=True)
"""


class Fake(Flat):
    system = "fake"

    def load(self) -> None:
        if self.variant == "noisy":
            print("noise on stdout from a library")

        if self.variant == "procqps":
            self.qps_mode = "processes"

        if self.variant == "loaded":
            self.loads_at_open = True

    def supported_regimes(self) -> tuple[str, ...]:
        return ("c0", "intent") if self.variant == "noc1" else self.regimes

    def cold_setup(self, ctx: ServeContext) -> None:
        if self.variant == "exec":
            print("this line goes to stderr, not the wire")
            self.exec_program([sys.executable, "-c", PROGRAM])

        super().cold_setup(ctx)

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


if __name__ == "__main__":
    main(Fake)
