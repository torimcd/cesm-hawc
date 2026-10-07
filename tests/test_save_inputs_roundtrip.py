"""A file written by save_column_inputs reproduces the same simulation when
its constituents are rebuilt with native sasktran2 calls alone."""

from __future__ import annotations

import pytest

sk = pytest.importorskip("sasktran2")
pytest.importorskip("hawcsimulator")

import numpy as np
import pandas as pd
import xarray as xr
from hawcsimulator.ali.configurations.ideal_dolp_imager import IdealALISimulator
from hawcsimulator.noise import ALINoiseModel

from cesm_hawc.noise import default_noise_model
from cesm_hawc.orbit_files import l1b_image_to_dataset
from cesm_hawc.save_inputs import save_column_inputs
from cesm_hawc.simulation import FORWARD_PRODUCTS, run_ali_simulation_from_profiles
from conftest import ALT_GRID_M, load_profiles_dict

WAVELENGTHS_NM = np.array([470.0, 745.0, 1020.0])
NOISE_SEED = 0
SIM_GEOMETRY = {
    "tangent_latitude": 30.6,
    "tangent_longitude": 180.0,
    "tangent_solar_zenith_angle": 60.0,
    "tangent_solar_azimuth_angle": 0.0,
    "altitude_grid": ALT_GRID_M,
    "polarization_states": ["I", "dolp"],
    "sample_wavelengths": WAVELENGTHS_NM,
    "time": pd.Timestamp("2030-01-07T12:00:00Z"),
}


class _Column:
    """Stands in for WACCMAtmosphere, serving fixed profiles."""

    def __init__(self, profiles: dict):
        self.profiles = profiles

    def get_column_profiles(self, lat, lon, time_index=0):
        return self.profiles


def _native_constituents(path) -> dict:
    """Rebuild the constituents from a saved file with sasktran2 only, as on
    the documentation's "Using saved inputs" page."""
    ds = xr.open_dataset(path)
    alt_m = ds["altitude_m"].values

    def mode_constituent(mode: str):
        width_key = "mode_width_accum" if mode == "aerosol_accum" else "mode_width_coarse"
        mode_db = sk.database.MieDatabase(
            sk.mie.distribution.LogNormalDistribution().freeze(mode_width=ds.attrs[width_key]),
            sk.mie.refractive.H2SO4(),
            ds.attrs["mie_wavelength_grid_nm"],
            median_radius=ds.attrs["mie_median_radius_grid_nm"],
        )
        return sk.constituent.ExtinctionScatterer(
            mode_db,
            altitudes_m=alt_m,
            extinction_per_m=ds[f"{mode}_reference_extinction_per_m"].values,
            extinction_wavelength_nm=ds.attrs["extinction_reference_wavelength_nm"],
            median_radius=ds[f"{mode}_median_radius_nm"].values,
        )

    return {
        "o3": sk.constituent.VMRAltitudeAbsorber(
            sk.optical.O3DBM(), altitudes_m=alt_m, vmr=ds["vmr_o3"].values),
        "no2": sk.constituent.VMRAltitudeAbsorber(
            sk.optical.NO2Vandaele(), altitudes_m=alt_m, vmr=ds["vmr_no2"].values),
        "aerosol_accum": mode_constituent("aerosol_accum"),
        "aerosol_coarse": mode_constituent("aerosol_coarse"),
    }


def test_saved_inputs_reproduce_forward_model(tmp_path, example_column_path):
    profiles = load_profiles_dict(example_column_path)
    saved = tmp_path / "column.nc"
    save_column_inputs(_Column(profiles), 30.6, 180.0, str(saved), 0,
                       ALT_GRID_M, WAVELENGTHS_NM)

    data = run_ali_simulation_from_profiles(
        profiles, ALT_GRID_M, SIM_GEOMETRY, products=FORWARD_PRODUCTS,
        noise_model=default_noise_model(seed=NOISE_SEED),
    )
    direct = IdealALISimulator().run(list(FORWARD_PRODUCTS), {
        **SIM_GEOMETRY,
        "constituents": _native_constituents(saved),
        "l1b_cfg": {"noise_model": ALINoiseModel(straylight_fraction=0.0, seed=NOISE_SEED)},
    })

    expected = l1b_image_to_dataset(data["l1b"], WAVELENGTHS_NM)
    actual = l1b_image_to_dataset(direct["l1b"], WAVELENGTHS_NM)

    # cesm-hawc clamps the Mie databases' single-scattering albedo below 1;
    # the native rebuild doesn't, so the two agree closely but not exactly.
    for var in ("radiance", "dolp"):
        np.testing.assert_allclose(actual[var].values, expected[var].values,
                                   rtol=1e-3, atol=1e-12, err_msg=var)
