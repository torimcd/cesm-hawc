from __future__ import annotations

import numpy as np

from cesm_hawc.waccm import (
    blend_h2o,
    hybrid_to_pressure,
    mam4_lognormal,
    pressure_to_altitude,
)


def test_hybrid_to_pressure_monotonic():
    hyam = np.array([0.01, 0.05, 0.2, 0.5, 0.9])
    hybm = np.zeros(5)
    p = hybrid_to_pressure(hyam, hybm, p0=100_000.0, ps=101_325.0)
    assert np.all(np.diff(p) > 0)
    assert p[0] > 0


def test_pressure_to_altitude_decreasing_with_pressure():
    pressure = np.array([1000.0, 5000.0, 20000.0, 50000.0, 90000.0])
    temperature = np.full(5, 250.0)
    alt = pressure_to_altitude(pressure, temperature, ps=101_325.0)
    # higher pressure (further down the array) -> lower altitude
    assert np.all(np.diff(alt) < 0)
    assert alt[-1] < alt[0]


def test_blend_h2o_endpoints():
    pressure = np.array([100.0, 1000.0, 10000.0, 50000.0, 100000.0])
    q_vmr = np.full(5, 1.0)
    chem_h2o = np.full(5, 2.0)
    merged = blend_h2o(q_vmr, chem_h2o, pressure, join_pa=10000.0)
    # low pressure (index 0, stratosphere): chemistry H2O dominates
    assert merged[0] == 2.0
    # high pressure (last index, near-surface troposphere): dynamics Q dominates
    assert merged[-1] == 1.0


def test_mam4_lognormal_positive_and_shape():
    so4_mmr = np.full(5, 1e-10)
    num_per_kg = np.full(5, 1e6)
    n_air_cm3 = np.full(5, 1e13)
    r_um, N_cm3 = mam4_lognormal(so4_mmr, num_per_kg, n_air_cm3, sigma_g=1.8)
    assert r_um.shape == (5,)
    assert N_cm3.shape == (5,)
    assert np.all(r_um > 0)
    assert np.all(N_cm3 > 0)


def test_mam4_lognormal_zero_mass_gives_floor_radius():
    so4_mmr = np.zeros(5)
    num_per_kg = np.full(5, 1e6)
    n_air_cm3 = np.full(5, 1e13)
    r_um, _ = mam4_lognormal(so4_mmr, num_per_kg, n_air_cm3, sigma_g=1.8)
    assert np.all(r_um > 0)  # floored at 1nm, never zero/negative
