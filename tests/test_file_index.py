from __future__ import annotations

from cesm_hawc.file_index import index_by_date, index_by_month


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
