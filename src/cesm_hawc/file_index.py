"""
cesm_hawc.file_index
=====================
Generic date/timestamp-based file indexing over a directory of CESM/WACCM
history files (h0 monthly, h1 hourly, h2 daily).
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd

_FILE_DATE_RE = re.compile(r"(\d{4}-\d{2})(?:-(\d{2}))?(?:-(\d{5}))?\.nc$")
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-\d+\.nc$")
_DATETIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d+)\.nc$")


def _glob_sorted(directory: str, pattern: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {directory}")
    return paths


def list_files(directory: str, pattern: str) -> list[str]:
    """Return the sorted paths in ``directory`` matching ``pattern``.
    Raises ``FileNotFoundError`` if there are none."""
    return _glob_sorted(directory, pattern)


def filename_date(path: str) -> str | None:
    """Return the date in a CAM history file name, as ``"YYYY-MM"``
    (``*.YYYY-MM.nc``, e.g. monthly h0) or ``"YYYY-MM-DD"``
    (``*.YYYY-MM-DD-SSSSS.nc``), or ``None`` if the name has no date."""
    m = _FILE_DATE_RE.search(os.path.basename(path))
    if m is None:
        return None
    return f"{m.group(1)}-{m.group(2)}" if m.group(2) else m.group(1)


def filename_time(path: str) -> pd.Timestamp | None:
    """Return a representative time for a CAM history file from its name:
    the 15th at 12:00 for ``YYYY-MM``, otherwise the date plus the
    ``SSSSS`` seconds of day (0 if absent). ``None`` if the name has no
    date."""
    m = _FILE_DATE_RE.search(os.path.basename(path))
    if m is None:
        return None
    if m.group(2) is None:
        return pd.Timestamp(f"{m.group(1)}-15T12:00:00")
    seconds = int(m.group(3)) if m.group(3) else 0
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}") + pd.Timedelta(seconds=seconds)


def date_in_range(date: str, start: str | None, end: str | None) -> bool:
    """True if ``date`` (``"YYYY-MM"`` or ``"YYYY-MM-DD"``) falls within
    ``[start, end]`` (``"YYYY-MM-DD"``, either may be ``None``). Bounds are
    compared at ``date``'s own precision, so a month is kept if any part of
    it is in range."""
    n = len(date)
    if start and date < start[:n]:
        return False
    if end and date > end[:n]:
        return False
    return True


def index_by_date(directory: str, pattern: str) -> dict[str, str]:
    """Return ``{"YYYY-MM-DD": filepath}`` for daily (h2) files in a
    directory, matching the ``*.cam.h2.YYYY-MM-DD-SSSSS.nc`` convention.

    Collapses to one file per calendar date. If a directory holds more
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
    (date, seconds-of-day). Unlike ``index_by_date``, nothing is
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
