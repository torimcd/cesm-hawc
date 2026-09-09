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

import pandas as pd

_MONTH_RE = re.compile(r"\d{4}-\d{2}(?!-)")
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-\d+\.nc$")
_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d+)\.nc$")


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
    directory, matching the ``*.cam.h2.YYYY-MM-DD-SSSSS.nc`` convention.

    Collapses to one file per calendar date -- if a directory holds more
    than one file for the same date (e.g. 12-hourly output, two files per
    day), only the last one in sorted order survives; the rest are
    silently dropped. Fine for genuinely-daily output; use
    ``index_by_timestamp`` instead for sub-daily output where every file
    needs to be kept.
    """
    result: dict[str, str] = {}
    for p in _glob_sorted(directory, pattern):
        m = _DATE_RE.search(os.path.basename(p))
        if m is not None:
            result[m.group(1)] = p
    return result


def index_by_timestamp(directory: str, pattern: str) -> dict[pd.Timestamp, str]:
    """Return ``{timestamp: filepath}`` for every file matching the
    ``*.cam.hN.YYYY-MM-DD-SSSSS.nc`` convention, one entry per distinct
    (date, seconds-of-day) -- unlike ``index_by_date``, nothing is
    collapsed when a directory holds more than one file per calendar date
    (e.g. 12-hourly output: ``...-00000.nc`` and ``...-43200.nc`` both
    survive as separate keys)."""
    result: dict[pd.Timestamp, str] = {}
    for p in _glob_sorted(directory, pattern):
        m = _DATETIME_RE.search(os.path.basename(p))
        if m is not None:
            date_str, seconds_str = m.group(1), m.group(2)
            ts = pd.Timestamp(date_str) + pd.Timedelta(seconds=int(seconds_str))
            result[ts] = p
    return result
