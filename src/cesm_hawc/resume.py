"""Helpers that let long batch runs resume after being interrupted.

They work at two levels, used together:

- Job level: :func:`outputs_already_exist` lets a dispatcher skip a whole
  job (e.g. a day) whose output files already exist.
- Item level: :func:`load_completed_keys` and :func:`append_csv_row` let a
  job that processes many items (e.g. one day's observations) record each
  finished item in a CSV as it goes, and skip those items when re-run.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)


def outputs_already_exist(expected_paths: list[str]) -> bool:
    """Check whether every expected output file exists.

    Parameters
    ----------
    expected_paths : list of str
        Paths a finished job would have written.

    Returns
    -------
    bool
        True if all of them exist.
    """
    return all(os.path.exists(p) for p in expected_paths)


def load_completed_keys(csv_path: str, key_columns: list[str],
                         expected_fieldnames: list[str]) -> set[tuple]:
    """Read which items a progress CSV already records as done.

    Parameters
    ----------
    csv_path : str
        Progress CSV written by :func:`append_csv_row`.
    key_columns : list of str
        Columns that identify an item.
    expected_fieldnames : list of str
        The CSV's full expected header.

    Returns
    -------
    set of tuple
        One tuple of ``key_columns`` values (as strings) per recorded row.
        Empty if the file doesn't exist, is empty, or can't be read.

    Notes
    -----
    If the file's header doesn't match ``expected_fieldnames`` (for example,
    a file left by an older version), it is renamed to
    ``<csv_path>.schema_mismatch.bak`` and the run starts fresh, rather than
    appending inconsistent rows or misreading columns.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()

    with open(csv_path) as f:
        existing_header = f.readline().strip().split(",")
    if existing_header != expected_fieldnames:
        backup_path = csv_path + ".schema_mismatch.bak"
        log.warning(
            "%s has a different column schema than expected (existing: %s | "
            "expected: %s). Backing up to %s and starting fresh.",
            csv_path, existing_header, expected_fieldnames, backup_path,
        )
        os.rename(csv_path, backup_path)
        return set()

    try:
        existing = pd.read_csv(csv_path, usecols=key_columns)
    except Exception as e:
        log.warning("Could not read %s for resume (%s); treating as no prior progress.",
                    csv_path, e)
        return set()
    return {tuple(str(v) for v in row) for row in existing[key_columns].itertuples(index=False)}


def append_csv_row(csv_path: str, row: dict, fieldnames: list[str]) -> None:
    """Append one row to a progress CSV, writing it to disk immediately.

    Writing each row as it finishes, rather than once at the end of a job,
    means a killed job loses at most the item in progress.

    Parameters
    ----------
    csv_path : str
        Progress CSV; it and its directory are created if needed, with a
        header row for a new file.
    row : dict
        Values for this row, keyed by column name.
    fieldnames : list of str
        Column order.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header_needed = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    pd.DataFrame([row], columns=fieldnames).to_csv(
        csv_path, mode="a", index=False, header=header_needed
    )
