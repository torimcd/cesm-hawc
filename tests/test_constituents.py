from __future__ import annotations

import pytest

sk = pytest.importorskip("sasktran2")

import numpy as np

from cesm_hawc.constituents import build_waccm_constituents
from conftest import ALT_GRID_M, load_profiles_dict


def test_build_waccm_constituents_from_example_fixture(example_background_column_path):
    profiles = load_profiles_dict(example_background_column_path)
    constituents = build_waccm_constituents(profiles, ALT_GRID_M)
    assert set(constituents.keys()) == {"o3", "no2", "aerosol_accum", "aerosol_coarse"}


def test_build_waccm_constituents_return_extinction(example_background_column_path):
    profiles = load_profiles_dict(example_background_column_path)
    wavelengths = np.array([470.0, 745.0, 1020.0])
    constituents, true_ext = build_waccm_constituents(
        profiles, ALT_GRID_M, return_extinction=True, truth_wavelengths_nm=wavelengths
    )
    assert "aerosol_accum_extinction_per_m" in true_ext
    assert "aerosol_coarse_extinction_per_m" in true_ext
    assert true_ext["aerosol_accum_extinction_per_m"].shape == (len(wavelengths), len(ALT_GRID_M))
    assert np.all(true_ext["aerosol_accum_extinction_per_m"] >= 0.0)

    # reconstruction-ready fields: the exact extinction_per_m/median_radius
    # args ExtinctionScatterer was built with, exposed so a saved column
    # can rebuild an equivalent constituent without calling this function
    # again (see cesm_hawc.save_inputs).
    for mode in ("aerosol_accum", "aerosol_coarse"):
        ref_ext = true_ext[f"{mode}_reference_extinction_per_m"]
        radius = true_ext[f"{mode}_median_radius_nm"]
        assert ref_ext.shape == (len(ALT_GRID_M),)
        assert radius.shape == (len(ALT_GRID_M),)
        assert np.all(ref_ext >= 0.0)
        assert np.all(radius > 0.0)


def test_get_mode_mie_database():
    from cesm_hawc.constituents import _MODE_WIDTHS, get_mode_mie_database

    db = get_mode_mie_database(_MODE_WIDTHS["aerosol_accum"])
    assert db is not None
