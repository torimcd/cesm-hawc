from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cesm_hawc.save_inputs import save_column_inputs
from cesm_hawc.waccm import WACCMAtmosphere

ALT_GRID_M = np.arange(0.0, 65001.0, 1000.0)


def test_save_column_inputs_profiles_only_flag(tmp_path, synthetic_waccm_file):
    """profiles_only=True must skip constituents regardless of whether
    sasktran2 happens to be installed."""
    waccm = WACCMAtmosphere(str(synthetic_waccm_file), alt_grid_km=ALT_GRID_M / 1e3)
    out_path = tmp_path / "column.nc"
    save_column_inputs(waccm, 30.6, 180.0, str(out_path), 0, ALT_GRID_M, profiles_only=True)

    ds = xr.open_dataset(out_path)
    assert not bool(ds.attrs["includes_constituents"])
    assert "aerosol_accum_reference_extinction_per_m" not in ds.data_vars
    assert "pressure_pa" in ds.data_vars


def test_save_column_inputs_without_sasktran2(tmp_path, synthetic_waccm_file):
    try:
        import sasktran2  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("sasktran2 is installed here; see test_save_column_inputs_with_constituents instead")

    waccm = WACCMAtmosphere(str(synthetic_waccm_file), alt_grid_km=ALT_GRID_M / 1e3)
    out_path = tmp_path / "column.nc"
    save_column_inputs(waccm, 30.6, 180.0, str(out_path), 0, ALT_GRID_M)

    ds = xr.open_dataset(out_path)
    assert not bool(ds.attrs["includes_constituents"])
    assert "pressure_pa" in ds.data_vars


def test_save_column_inputs_with_constituents(tmp_path, synthetic_waccm_file):
    pytest.importorskip("sasktran2")

    waccm = WACCMAtmosphere(str(synthetic_waccm_file), alt_grid_km=ALT_GRID_M / 1e3)
    out_path = tmp_path / "column.nc"
    wavelengths_nm = np.array([470.0, 745.0, 1020.0])
    save_column_inputs(waccm, 30.6, 180.0, str(out_path), 0, ALT_GRID_M, wavelengths_nm)

    ds = xr.open_dataset(out_path)
    assert bool(ds.attrs["includes_constituents"])
    assert ds.attrs["mode_width_accum"] == 1.8
    assert ds.attrs["mode_width_coarse"] == 1.2
    assert ds.attrs["mie_refractive_index"] == "H2SO4"
    np.testing.assert_array_equal(ds["wavelength_nm"].values, wavelengths_nm)

    for mode in ("aerosol_accum", "aerosol_coarse"):
        assert ds[f"{mode}_extinction_per_m"].dims == ("wavelength_nm", "altitude_m")
        assert ds[f"{mode}_reference_extinction_per_m"].dims == ("altitude_m",)
        assert ds[f"{mode}_median_radius_nm"].dims == ("altitude_m",)
        assert np.all(ds[f"{mode}_median_radius_nm"].values > 0)
        assert np.all(ds[f"{mode}_reference_extinction_per_m"].values >= 0)
