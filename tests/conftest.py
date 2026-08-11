from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

EXAMPLE_FIXTURE_DIR = Path(__file__).parent.parent / "src" / "cesm_hawc" / "data"
ALT_GRID_M = np.arange(0.0, 65001.0, 1000.0)


@pytest.fixture
def example_background_column_path() -> Path:
    path = EXAMPLE_FIXTURE_DIR / "example_column_background.nc"
    if not path.exists():
        pytest.skip("bundled example fixture not found")
    return path


@pytest.fixture
def example_injection_column_path() -> Path:
    path = EXAMPLE_FIXTURE_DIR / "example_column_injection.nc"
    if not path.exists():
        pytest.skip("bundled example fixture not found")
    return path


def load_profiles_dict(nc_path) -> dict:
    """Reconstruct a ``WACCMAtmosphere.get_column_profiles()``-shaped dict
    from a NetCDF written by ``WACCMAtmosphere.save_column_profiles()``."""
    ds = xr.open_dataset(nc_path)
    profiles = {"altitudes_m": ds["altitude_m"].values}
    for name, da in ds.data_vars.items():
        profiles[name] = da.values
    profiles["sulfate_a1_sigma"] = float(ds.attrs.get("sigma_a1", 1.8))
    profiles["sulfate_a3_sigma"] = float(ds.attrs.get("sigma_a3", 1.2))
    ds.close()
    return profiles


@pytest.fixture
def synthetic_waccm_file(tmp_path) -> Path:
    """A small, self-contained, in-memory-built WACCM-h0-shaped NetCDF file
    for exercising ``WACCMAtmosphere`` without needing real (large) CESM
    output or any heavy optional dependency."""
    nlev = 5
    lat = np.array([-45.0, 0.0, 30.6, 60.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    shape4 = (1, nlev, len(lat), len(lon))
    shape3 = (1, len(lat), len(lon))

    hyam = np.array([0.01, 0.05, 0.2, 0.5, 0.9])
    hybm = np.zeros(nlev)
    # T varies distinctly per longitude index (not just per level) so tests
    # can tell whether a query actually selected the right column, rather
    # than every column being physically identical by construction.
    lon_offset = (np.arange(len(lon)) * 5.0).reshape(1, 1, 1, len(lon))
    T = np.linspace(220.0, 290.0, nlev).reshape(1, nlev, 1, 1) * np.ones(shape4) + lon_offset
    Q = np.linspace(1e-6, 1e-3, nlev).reshape(1, nlev, 1, 1) * np.ones(shape4)
    PS = np.full(shape3, 101325.0)
    O3 = np.linspace(1e-6, 1e-7, nlev).reshape(1, nlev, 1, 1) * np.ones(shape4)
    NO2 = np.full(shape4, 1e-10)
    H2O = np.linspace(1e-6, 3e-6, nlev).reshape(1, nlev, 1, 1) * np.ones(shape4)
    SO2 = np.full(shape4, 1e-11)
    so4_a1 = np.full(shape4, 1e-10)
    so4_a3 = np.full(shape4, 5e-11)
    num_a1 = np.full(shape4, 1e6)
    num_a3 = np.full(shape4, 5e5)

    ds = xr.Dataset(
        data_vars={
            "T": (("time", "lev", "lat", "lon"), T),
            "Q": (("time", "lev", "lat", "lon"), Q),
            "PS": (("time", "lat", "lon"), PS),
            "O3": (("time", "lev", "lat", "lon"), O3),
            "NO2": (("time", "lev", "lat", "lon"), NO2),
            "H2O": (("time", "lev", "lat", "lon"), H2O),
            "SO2": (("time", "lev", "lat", "lon"), SO2),
            "so4_a1": (("time", "lev", "lat", "lon"), so4_a1),
            "so4_a3": (("time", "lev", "lat", "lon"), so4_a3),
            "num_a1": (("time", "lev", "lat", "lon"), num_a1),
            "num_a3": (("time", "lev", "lat", "lon"), num_a3),
        },
        coords={
            "time": [0],
            "lev": np.arange(nlev),
            "lat": lat,
            "lon": lon,
            "hyam": ("lev", hyam),
            "hybm": ("lev", hybm),
            "P0": 100_000.0,
        },
    )
    path = tmp_path / "synthetic.cam.h0.2030-01.nc"
    ds.to_netcdf(path)
    return path
