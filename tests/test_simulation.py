from __future__ import annotations

import pytest

pytest.importorskip("sasktran2")
pytest.importorskip("hawcsimulator")

import numpy as np
import pandas as pd

from cesm_hawc.simulation import DEFAULT_PRODUCTS, run_ali_simulation_from_profiles
from conftest import ALT_GRID_M, load_profiles_dict


def test_run_ali_simulation_from_profiles_smoke(example_background_column_path):
    """End-to-end forward-model + L2 retrieval smoke test against the
    bundled example fixture. This is the migration target for the old
    scripts/test_one_day_multicase.py manual smoke check."""
    profiles = load_profiles_dict(example_background_column_path)
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
    )

    assert "l2" in data
    assert "l1b" in data
    assert "stratospheric_aerosol_extinction_per_m" in data["l2"]
