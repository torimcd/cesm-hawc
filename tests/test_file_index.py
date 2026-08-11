from __future__ import annotations

import pandas as pd

from cesm_hawc.file_index import find_nearest, index_by_date, index_by_month, index_by_timestamp


def _touch(path):
    path.write_bytes(b"")


def test_index_by_month(tmp_path):
    _touch(tmp_path / "case.cam.h0.2030-01.nc")
    _touch(tmp_path / "case.cam.h0.2030-02.nc")
    index = index_by_month(str(tmp_path), "*.cam.h0.*.nc")
    assert set(index.keys()) == {"2030-01", "2030-02"}


def test_index_by_month_filter(tmp_path):
    _touch(tmp_path / "case.cam.h0.2030-01.nc")
    _touch(tmp_path / "case.cam.h0.2030-02.nc")
    index = index_by_month(str(tmp_path), "*.cam.h0.*.nc", month_filter=["2030-01"])
    assert set(index.keys()) == {"2030-01"}


def test_index_by_date(tmp_path):
    _touch(tmp_path / "case.cam.h2.2030-01-07-00000.nc")
    _touch(tmp_path / "case.cam.h2.2030-01-08-00000.nc")
    index = index_by_date(str(tmp_path), "*.cam.h2.*.nc")
    assert set(index.keys()) == {"2030-01-07", "2030-01-08"}


def test_index_by_timestamp_and_find_nearest(tmp_path):
    _touch(tmp_path / "case.cam.h1.2030-01-07-00000.nc")
    _touch(tmp_path / "case.cam.h1.2030-01-07-03600.nc")
    index = index_by_timestamp(str(tmp_path), "*.cam.h1.*.nc")
    assert len(index) == 2

    target = pd.Timestamp("2030-01-07T00:10:00")
    nearest = find_nearest(target, index, max_gap_s=1800.0)
    assert nearest is not None
    assert nearest.endswith("00000.nc")

    far_target = pd.Timestamp("2030-06-01T00:00:00")
    assert find_nearest(far_target, index, max_gap_s=1800.0) is None
