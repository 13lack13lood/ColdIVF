"""Small helpers for adapters that run a library in this process.

`settings` checks a build config or a search setting against the keys a system takes. `as_ids` turns one row of a
library's results into the ids the driver expects. `check_unit` makes sure a cosine dataset has unit vectors, for
libraries that rank by inner product or by L2 distance, which rank unit vectors as the cosine does.
"""

from __future__ import annotations

import platform
from typing import Any

import numpy as np

# The default of a key that has to be given.
REQUIRED = object()


def settings(given: dict | None, defaults: dict[str, Any], what: str) -> dict[str, Any]:
    """`given` laid over `defaults`. A key that isn't in `defaults` is an error, so a typo never runs a default, and
    so is a key whose default is REQUIRED and that `given` leaves out."""
    given = dict(given or {})
    unknown = sorted(set(given) - set(defaults))

    if unknown:
        raise ValueError(f"unknown {what} {unknown}, expected some of {sorted(defaults)}")

    out = defaults | given
    missing = sorted(k for k, v in out.items() if v is REQUIRED)

    if missing:
        raise ValueError(f"the {what} need {missing}")

    return out


def as_ids(found: Any, k: int, empty: int | None = None) -> np.ndarray:
    """One row of results as k int64 ids, nearest first. Slots the library left empty, which hold `empty`, become -1,
    and a short row is padded with -1."""
    row = np.asarray(found).ravel()[:k]
    ids = row.astype(np.int64)

    if empty is not None:
        ids[row == empty] = -1

    out = np.full(k, -1, dtype=np.int64)
    out[: len(ids)] = ids
    return out


def check_unit(rows: np.ndarray, samples: int = 1000, tol: float = 1e-3) -> None:
    """Raises ValueError unless `samples` rows spread evenly over `rows` have unit length, within `tol`."""
    if len(rows) == 0:
        return

    picks = np.unique(np.linspace(0, len(rows) - 1, min(samples, len(rows))).astype(np.int64))
    norms = np.linalg.norm(np.asarray(rows[picks], dtype=np.float64), axis=1)
    worst = int(np.argmax(np.abs(norms - 1.0)))

    if abs(norms[worst] - 1.0) > tol:
        raise ValueError(f"row {int(picks[worst])} has length {norms[worst]:.6f}, and cosine here needs unit vectors")


def simd(arm: str, x86: str) -> str:
    """The SIMD (single instruction, multiple data) level of a wheel that uses NEON on Arm and picks its x86 kernels at
    run time."""
    return arm if platform.machine() in ("arm64", "aarch64") else x86
