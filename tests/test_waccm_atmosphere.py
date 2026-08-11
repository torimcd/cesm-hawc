from __future__ import annotations

import numpy as np

from cesm_hawc.waccm import WACCMAtmosphere


def test_get_column_profiles_shape(synthetic_waccm_file):
    atm = WACCMAtmosphere(str(synthetic_waccm_file), alt_grid_km=np.arange(0.0, 65.001, 1.0))
    profiles = atm.get_column_profiles(lat=30.6, lon=180.0, time_index=0)

    expected_keys = {
        "altitudes_m", "pressure_pa", "temperature_k", "specific_humidity",
        "vmr_o3", "vmr_no2", "vmr_h2o", "vmr_so2", "n_air_cm3",
        "sulfate_a1_N_cm3", "sulfate_a1_r_um", "sulfate_a1_sigma",
        "sulfate_a3_N_cm3", "sulfate_a3_r_um", "sulfate_a3_sigma",
    }
    assert expected_keys <= profiles.keys()
    n_alt = len(profiles["altitudes_m"])
    assert profiles["pressure_pa"].shape == (n_alt,)
    assert profiles["temperature_k"].shape == (n_alt,)
    assert np.all(profiles["pressure_pa"] > 0)
    assert np.all(profiles["temperature_k"] > 0)


def test_longitude_normalization_bug_fix(synthetic_waccm_file):
    """A negative longitude must select the same column as its 0-360
    equivalent (lon=-90 == lon=270), not silently snap to lon=0."""
    atm = WACCMAtmosphere(str(synthetic_waccm_file), alt_grid_km=np.arange(0.0, 65.001, 1.0))

    profiles_neg = atm.get_column_profiles(lat=30.6, lon=-90.0, time_index=0)
    profiles_pos = atm.get_column_profiles(lat=30.6, lon=270.0, time_index=0)
    profiles_zero = atm.get_column_profiles(lat=30.6, lon=0.0, time_index=0)

    np.testing.assert_array_equal(profiles_neg["temperature_k"], profiles_pos["temperature_k"])
    np.testing.assert_array_equal(profiles_neg["pressure_pa"], profiles_pos["pressure_pa"])
    # and it must NOT have collapsed to the lon=0 column
    assert not np.array_equal(profiles_neg["temperature_k"], profiles_zero["temperature_k"])


def test_missing_required_vars_raises(tmp_path):
    import xarray as xr

    ds = xr.Dataset(
        data_vars={"T": (("time", "lat", "lon"), np.ones((1, 2, 2)))},
        coords={"time": [0], "lat": [0.0, 30.0], "lon": [0.0, 180.0]},
    )
    path = tmp_path / "incomplete.nc"
    ds.to_netcdf(path)

    try:
        WACCMAtmosphere(str(path))
        assert False, "expected ValueError for missing required variables"
    except ValueError as e:
        assert "Missing required WACCM variables" in str(e)


def test_sulfate_column_burden(synthetic_waccm_file):
    atm = WACCMAtmosphere(str(synthetic_waccm_file), alt_grid_km=np.arange(0.0, 65.001, 1.0))
    burden = atm.sulfate_column_burden(lat=30.6, lon=180.0, time_index=0)
    assert burden["burden_mg_m2"] >= 0.0
    assert burden["dominant_mode"] in ("a1", "a3", "none")


def test_save_and_extract_cesm_extinction_roundtrip(tmp_path, synthetic_waccm_file):
    atm = WACCMAtmosphere(str(synthetic_waccm_file), alt_grid_km=np.arange(0.0, 65.001, 1.0))
    out_path = tmp_path / "column.nc"
    atm.save_column_profiles(30.6, 180.0, str(out_path), time_index=0)
    assert out_path.exists()

    # extract_cesm_extinction should not error even when EXTINCT* vars are
    # absent (synthetic fixture doesn't define them) -- returns {} cleanly.
    extracted = atm.extract_cesm_extinction(30.6, 180.0, 0, np.arange(0.0, 65001.0, 1000.0))
    assert extracted == {}
