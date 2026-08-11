"""
cesm_hawc.resume
==================
Generic resumability helpers for long-running batch jobs: skip work whose
expected outputs already exist, and incrementally persist per-item progress
to a CSV so a killed job resumes instead of restarting.

Two layers, used together for expensive multi-hour batch jobs:

- Coarse (job-level): ``outputs_already_exist()`` lets a job dispatcher skip
  a whole job (e.g. a day) whose expected output files already exist, so
  re-submitting doesn't reprocess completed work from scratch.
- Fine (item-level, within one job): ``load_completed_keys()`` /
  ``append_csv_row()`` let a single job that processes many items (e.g. one
  day's many observations) skip items already recorded in a CSV written
  incrementally during a previous run, and pick up only the remaining
  items rather than redoing the whole job.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)


def outputs_already_exist(expected_paths: list[str]) -> bool:
    """True if every path in ``expected_paths`` already exists on disk."""
    return all(os.path.exists(p) for p in expected_paths)


def load_completed_keys(csv_path: str, key_columns: list[str],
                         expected_fieldnames: list[str]) -> set[tuple]:
    """
    Return the set of ``key_columns`` tuples already present in an existing
    progress CSV, so a resumed run skips re-doing that work.

    If the file's header doesn't match ``expected_fieldnames`` (e.g. left
    over from an older version of the caller), back it up and start fresh
    rather than risk corrupting it with inconsistent columns, or silently
    misreading columns that have since changed meaning.
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
    """Append one row to a progress CSV immediately, flushed to disk,
    rather than accumulating in memory for a single end-of-job write."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header_needed = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    pd.DataFrame([row], columns=fieldnames).to_csv(
        csv_path, mode="a", index=False, header=header_needed
    )
