from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cesm_hawc.orbit_files import l1b_image_to_dataset

ALT_GRID_M = np.arange(0.0, 65001.0, 1000.0)


class _FakeSpectra:
    def __init__(self, ds):
        self.ds = ds


class _FakeL1bImage:
    def __init__(self, spectra):
        self.spectra = spectra


def _fake_l1b(wavelengths_nm, tangent_altitudes_m):
    """A minimal stand-in for hawcsimulator's L1bImage, carrying just the
    fields l1b_image_to_dataset() actually reads."""
    n_wl, n_alt = len(wavelengths_nm), len(tangent_altitudes_m)
    shape = (n_wl, n_alt)

    def make_ds(seed):
        rng = np.random.default_rng(seed)
        ds = xr.Dataset(
            data_vars={
                "radiance": (("wavelength", "los"), rng.random(shape)),
                "radiance_noise": (("wavelength", "los"), rng.random(shape) * 0.01),
            },
            coords={
                "wavelength": wavelengths_nm,
                "tangent_altitude": ("los", tangent_altitudes_m),
                "tangent_latitude": ("los", np.full(n_alt, 30.6)),
                "tangent_longitude": ("los", np.full(n_alt, 180.0)),
                "solar_zenith_angle": ("los", np.full(n_alt, 60.0)),
            },
        )
        return ds.assign_coords(time=pd.Timestamp("2035-01-01T12:00:00"))

    return _FakeL1bImage({"I": _FakeSpectra(make_ds(0)), "dolp": _FakeSpectra(make_ds(1))})


def test_l1b_image_to_dataset_basic():
    wavelengths_nm = np.array([470.0, 745.0, 1020.0])
    tangent_alt = np.arange(10000.0, 40000.0, 500.0)
    l1b = _fake_l1b(wavelengths_nm, tangent_alt)

    ds = l1b_image_to_dataset(l1b, wavelengths_nm)
    assert ds["radiance"].dims == ("wavelength", "altitude_m")
    assert ds["dolp"].shape == (len(wavelengths_nm), len(tangent_alt))


def test_l1b_image_to_dataset_with_true_extinction(synthetic_waccm_file):
    """Regression test: build_waccm_constituents(..., return_extinction=True)
    returns per-altitude-only {name}_reference_extinction_per_m /
    {name}_median_radius_nm entries alongside the 2D [wavelength, altitude]
    {name}_extinction_per_m truth arrays -- l1b_image_to_dataset must not
    choke on that shape mix (it did: ValueError on the 1D entries)."""
    pytest.importorskip("sasktran2")
    from cesm_hawc.constituents import build_waccm_constituents
    from cesm_hawc.waccm import WACCMAtmosphere

    wavelengths_nm = np.array([470.0, 745.0, 1020.0])
    tangent_alt = np.arange(10000.0, 40000.0, 500.0)
    l1b = _fake_l1b(wavelengths_nm, tangent_alt)

    waccm = WACCMAtmosphere(str(synthetic_waccm_file), alt_grid_km=ALT_GRID_M / 1e3)
    profiles = waccm.get_column_profiles(30.6, 180.0, time_index=0)
    _, true_ext = build_waccm_constituents(
        profiles, ALT_GRID_M, return_extinction=True, truth_wavelengths_nm=wavelengths_nm
    )

    ds = l1b_image_to_dataset(l1b, wavelengths_nm, true_ext, ALT_GRID_M)

    assert ds["aerosol_accum_extinction_per_m"].dims == ("wavelength", "altitude_m")
    assert ds["aerosol_accum_extinction_per_m_atm"].dims == ("wavelength", "atm_altitude_m")
    assert ds["aerosol_coarse_extinction_per_m"].dims == ("wavelength", "altitude_m")
    # the per-altitude-only entries aren't wavelength-resolved and must not
    # have been attached here (they're for cesm_hawc.save_inputs instead)
    assert "aerosol_accum_reference_extinction_per_m" not in ds.data_vars
    assert "aerosol_accum_median_radius_nm" not in ds.data_vars
