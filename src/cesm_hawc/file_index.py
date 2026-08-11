"""
cesm_hawc.file_index
=====================
Generic date/timestamp-based file indexing over a directory of CESM/WACCM
history files (h0 monthly, h1 hourly, h2 daily — all share the same
``*.cam.hN.<date>[-<seconds>].nc`` naming convention, just with different
date granularity).
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

_MONTH_RE = re.compile(r"\d{4}-\d{2}(?!-)")
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-\d+\.nc$")
_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{5})")


def _glob_sorted(directory: str, pattern: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {directory}")
    return paths


def index_by_month(directory: str, pattern: str,
                    month_filter: list[str] | None = None) -> dict[str, str]:
    """Return ``{"YYYY-MM": filepath}`` for monthly (h0) files in a directory.

    If ``month_filter`` is given and non-empty, only those months are kept.
    """
    result: dict[str, str] = {}
    for p in _glob_sorted(directory, pattern):
        m = _MONTH_RE.search(os.path.basename(p))
        if m is None:
            continue
        date = m.group(0)
        if month_filter and date not in month_filter:
            continue
        result[date] = p
    return result


def index_by_date(directory: str, pattern: str) -> dict[str, str]:
    """Return ``{"YYYY-MM-DD": filepath}`` for daily (h2) files in a
    directory, matching the ``*.cam.h2.YYYY-MM-DD-SSSSS.nc`` convention."""
    result: dict[str, str] = {}
    for p in _glob_sorted(directory, pattern):
        m = _DATE_RE.search(os.path.basename(p))
        if m is not None:
            result[m.group(1)] = p
    return result


def index_by_timestamp(directory: str, pattern: str,
                        regex: re.Pattern | None = None) -> dict[pd.Timestamp, str]:
    """Return ``{timestamp: filepath}`` for files carrying a
    ``YYYY-MM-DD-SSSSS`` timestamp (SSSSS = seconds of day), matching the
    hourly (h1) file convention by default. Pass a custom ``regex`` with
    two groups (date, seconds-of-day) for other conventions."""
    regex = regex or _TIMESTAMP_RE
    result: dict[pd.Timestamp, str] = {}
    for p in _glob_sorted(directory, pattern):
        m = regex.search(os.path.basename(p))
        if m is not None:
            ts = pd.Timestamp(m.group(1)) + pd.Timedelta(seconds=int(m.group(2)))
            result[ts] = p
    return result


def find_nearest(target_ts: pd.Timestamp, index: dict[pd.Timestamp, str],
                  max_gap_s: float) -> str | None:
    """Return the path in ``index`` whose key is nearest ``target_ts``, or
    ``None`` if the nearest is farther than ``max_gap_s`` away."""
    sorted_ts = sorted(index.keys())
    if not sorted_ts:
        return None
    times_s = np.array([t.timestamp() for t in sorted_ts])
    t0 = target_ts.timestamp()
    idx = int(np.searchsorted(times_s, t0))
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(times_s):
        candidates.append(idx)
    best = min(candidates, key=lambda i: abs(times_s[i] - t0))
    if abs(times_s[best] - t0) > max_gap_s:
        return None
    return index[sorted_ts[best]]
