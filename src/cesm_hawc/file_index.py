"""Find CAM history files and read the dates in their names.

CAM history files are named ``<case>.cam.hN.YYYY-MM.nc`` (e.g. monthly
means) or ``<case>.cam.hN.YYYY-MM-DD-SSSSS.nc``, where ``SSSSS`` is the
second of the day.
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
    """Sorted glob matches; raises ``FileNotFoundError`` if there are none."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {directory}")
    return paths


def list_files(directory: str, pattern: str) -> list[str]:
    """List the files in a directory that match a glob pattern.

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str
        Glob pattern, e.g. ``"*.cam.h0.*.nc"``.

    Returns
    -------
    list of str
        Matching paths, sorted.

    Raises
    ------
    FileNotFoundError
        If nothing matches.
    """
    return _glob_sorted(directory, pattern)


def filename_date(path: str) -> str | None:
    """Read the date from a CAM history file name.

    Parameters
    ----------
    path : str
        File path; only the file name is used.

    Returns
    -------
    str or None
        ``"YYYY-MM"`` for ``*.YYYY-MM.nc`` names, ``"YYYY-MM-DD"`` for
        ``*.YYYY-MM-DD-SSSSS.nc`` names, or ``None`` if the name has no
        date.
    """
    m = _FILE_DATE_RE.search(os.path.basename(path))
    if m is None:
        return None
    return f"{m.group(1)}-{m.group(2)}" if m.group(2) else m.group(1)


def filename_time(path: str) -> pd.Timestamp | None:
    """Choose a representative time for a CAM history file from its name.

    Parameters
    ----------
    path : str
        File path; only the file name is used.

    Returns
    -------
    pandas.Timestamp or None
        The 15th at 12:00 for ``*.YYYY-MM.nc`` names; otherwise the date
        plus the ``SSSSS`` seconds of day (0 if absent). ``None`` if the
        name has no date.
    """
    m = _FILE_DATE_RE.search(os.path.basename(path))
    if m is None:
        return None
    if m.group(2) is None:
        return pd.Timestamp(f"{m.group(1)}-15T12:00:00")
    seconds = int(m.group(3)) if m.group(3) else 0
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}") + pd.Timedelta(seconds=seconds)


def date_in_range(date: str, start: str | None, end: str | None) -> bool:
    """Check whether a file date falls within a date range.

    Bounds are compared at ``date``'s own precision, so a month is in range
    if any part of it is.

    Parameters
    ----------
    date : str
        ``"YYYY-MM"`` or ``"YYYY-MM-DD"``, as returned by
        :func:`filename_date`.
    start, end : str or None
        Inclusive bounds as ``"YYYY-MM-DD"``; ``None`` means unbounded.

    Returns
    -------
    bool
        True if ``date`` is within ``[start, end]``.
    """
    n = len(date)
    if start and date < start[:n]:
        return False
    if end and date > end[:n]:
        return False
    return True


def index_by_date(directory: str, pattern: str) -> dict[str, str]:
    """Index daily history files by calendar date.

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str
        Glob pattern, e.g. ``"*.cam.h2.*.nc"``. Only names ending in
        ``YYYY-MM-DD-SSSSS.nc`` are indexed.

    Returns
    -------
    dict of str to str
        ``{"YYYY-MM-DD": path}``.

    Raises
    ------
    FileNotFoundError
        If nothing matches ``pattern``.

    Notes
    -----
    Keeps one file per date: if a date has several (e.g. 12-hourly output),
    only the last in sorted order is kept and the rest are dropped without
    warning. Use :func:`index_by_timestamp` for sub-daily output.
    """
    result: dict[str, str] = {}
    for p in _glob_sorted(directory, pattern):
        m = _DATE_RE.search(os.path.basename(p))
        if m is not None:
            result[m.group(1)] = p
    return result


def index_by_timestamp(directory: str, pattern: str) -> dict[pd.Timestamp, str]:
    """Index history files by date and time of day.

    Unlike :func:`index_by_date`, every file is kept when there are several
    per day (e.g. ``...-00000.nc`` and ``...-43200.nc`` for 12-hourly
    output).

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str
        Glob pattern. Only names ending in ``YYYY-MM-DD-SSSSS.nc`` are
        indexed.

    Returns
    -------
    dict of pandas.Timestamp to str
        ``{date + SSSSS seconds: path}``.

    Raises
    ------
    FileNotFoundError
        If nothing matches ``pattern``.
    """
    result: dict[pd.Timestamp, str] = {}
    for p in _glob_sorted(directory, pattern):
        m = _DATETIME_RE.search(os.path.basename(p))
        if m is not None:
            date_str, seconds_str = m.group(1), m.group(2)
            ts = pd.Timestamp(date_str) + pd.Timedelta(seconds=int(seconds_str))
            result[ts] = p
    return result
