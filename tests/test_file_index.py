from __future__ import annotations

import pandas as pd

from cesm_hawc.file_index import (
    date_in_range,
    filename_date,
    filename_time,
    index_by_date,
    index_by_timestamp,
)


def _touch(path):
    path.write_bytes(b"")


def test_filename_date():
    assert filename_date("/x/case.cam.h0.2030-01.nc") == "2030-01"
    assert filename_date("/x/case.cam.h2.2030-01-07-43200.nc") == "2030-01-07"
    assert filename_date("/x/notes.nc") is None


def test_filename_time():
    assert filename_time("case.cam.h0.2030-01.nc") == pd.Timestamp("2030-01-15 12:00")
    assert filename_time("case.cam.h2.2030-01-07-43200.nc") == pd.Timestamp("2030-01-07 12:00")


def test_date_in_range_compares_at_file_precision():
    # a month overlapping the range is kept
    assert date_in_range("2030-02", "2030-02-10", "2030-03-31")
    assert not date_in_range("2030-01", "2030-02-10", None)
    assert date_in_range("2030-01-07", None, None)
    assert not date_in_range("2030-01-07", None, "2030-01-06")


def test_index_by_date(tmp_path):
    _touch(tmp_path / "case.cam.h2.2030-01-07-00000.nc")
    _touch(tmp_path / "case.cam.h2.2030-01-08-00000.nc")
    index = index_by_date(str(tmp_path), "*.cam.h2.*.nc")
    assert set(index.keys()) == {"2030-01-07", "2030-01-08"}


def test_index_by_date_collapses_subdaily_files(tmp_path):
    """Documents the known limitation: index_by_date silently keeps only
    one file per calendar date. Use index_by_timestamp for sub-daily
    output where both files need to survive."""
    _touch(tmp_path / "case.cam.h2.2030-01-07-00000.nc")
    _touch(tmp_path / "case.cam.h2.2030-01-07-43200.nc")
    index = index_by_date(str(tmp_path), "*.cam.h2.*.nc")
    assert set(index.keys()) == {"2030-01-07"}
    assert len(index) == 1


def test_index_by_timestamp_keeps_all_subdaily_files(tmp_path):
    _touch(tmp_path / "case.cam.h2.2030-01-07-00000.nc")
    _touch(tmp_path / "case.cam.h2.2030-01-07-43200.nc")
    _touch(tmp_path / "case.cam.h2.2030-01-08-00000.nc")
    index = index_by_timestamp(str(tmp_path), "*.cam.h2.*.nc")
    assert set(index.keys()) == {
        pd.Timestamp("2030-01-07 00:00:00"),
        pd.Timestamp("2030-01-07 12:00:00"),
        pd.Timestamp("2030-01-08 00:00:00"),
    }
