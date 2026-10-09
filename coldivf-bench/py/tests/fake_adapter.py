"""The flat adapter with failures on demand, chosen by the variant, for the driver's tests.

- "crash" raises inside every cold trial.
- "selfreject" reports every cold trial as rejected, as a binary that finds cached pages at its timer does.
- "noc1" offers no C1 regime.
- "noisy" prints to stdout while it loads, which must not reach the wire.
- "exec" hands its cold subcommand to a small program that speaks the wire itself.
- "procqps" measures QPS with one process per worker.
- "loaded" marks every file as loaded at open, though it maps them, so its cached pages show apart.
"""

from __future__ import annotations

import sys

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

    def cold_trial(self, ctx: ServeContext, go: dict) -> dict:
        if self.variant == "crash":
            raise RuntimeError("a crash on purpose")

        fields = super().cold_trial(ctx, go)

        if self.variant == "selfreject":
            fields.update(status="rejected", error="rejected on purpose")

        return fields


if __name__ == "__main__":
    main(Fake)
