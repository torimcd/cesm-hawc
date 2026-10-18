from __future__ import annotations

import pytest

pytest.importorskip("sasktran2")
pytest.importorskip("hawcsimulator")

import numpy as np
import pandas as pd

from cesm_hawc.noise import default_noise_model
from cesm_hawc.simulation import DEFAULT_PRODUCTS, run_ali_simulation_from_profiles
from conftest import ALT_GRID_M, load_profiles_dict


def test_run_ali_simulation_from_profiles_smoke(example_column_path):
    """End-to-end forward-model + L2 retrieval smoke test against the
    bundled example fixture.

    Explicit noise_model is required: IdealALISimulator (ideal_dolp_imager)
    has no noiseless fallback, omitting it raises AttributeError."""
    profiles = load_profiles_dict(example_column_path)
    sim_geometry = {
        "tangent_latitude": 30.6,
        "tangent_longitude": 180.0,
        "tangent_solar_zenith_angle": 60.0,
        "tangent_solar_azimuth_angle": 0.0,
        "altitude_grid": ALT_GRID_M,
        "polarization_states": ["I", "dolp"],
        "sample_wavelengths": np.array([470.0, 745.0, 1020.0]),
        "time": pd.Timestamp("2030-01-07T12:00:00Z"),
    }

    data = run_ali_simulation_from_profiles(
        profiles, ALT_GRID_M, sim_geometry, products=DEFAULT_PRODUCTS,
        noise_model=default_noise_model(),
    )

    assert "l2" in data
    assert "l1b" in data
    assert "stratospheric_aerosol_extinction_per_m" in data["l2"]


@pytest.fixture
def retrieval_range(monkeypatch):
    import cesm_hawc.simulation as simulation

    monkeypatch.setattr(simulation, "retrieval_wavelength_range", lambda: (470.0, 1650.0))
    return simulation


def test_check_retrieval_wavelengths_accepts_wavelengths_in_range(retrieval_range):
    retrieval_range.check_retrieval_wavelengths([470.0, 745.0, 1650.0])


def test_check_retrieval_wavelengths_rejects_wavelengths_out_of_range(retrieval_range):
    with pytest.raises(ValueError, match=r"2000\.0"):
        retrieval_range.check_retrieval_wavelengths([745.0, 2000.0])


def test_l2_run_rejects_out_of_range_wavelength_before_simulating(retrieval_range, example_column_path):
    profiles = load_profiles_dict(example_column_path)
    sim_geometry = {"sample_wavelengths": np.array([745.0, 2000.0])}
    with pytest.raises(ValueError, match="outside the L2 retrieval's range"):
        retrieval_range.run_ali_simulation_from_profiles(
            profiles, ALT_GRID_M, sim_geometry, products=DEFAULT_PRODUCTS,
            noise_model=default_noise_model(),
        )


def test_retrieval_wavelength_range_covers_reference_wavelength():
    from cesm_hawc.simulation import retrieval_wavelength_range

    low, high = retrieval_wavelength_range()
    assert low <= 745.0 <= high
