"""
cesm_hawc.constituents
======================
Build the sasktran2 constituents dict that the HAWC ALI simulator needs.

The IdealALISimulator uses a Hamilton DAG. The atmosphere step
(``hawcsimulator.steps.atmosphere.atmosphere__default``) starts from:

    - Rayleigh scattering
    - MIPAS O₃ climatology
    - Solar irradiance
    - Lambertian surface albedo (0.3)

and then merges whatever is in the ``constituents`` dict on top. This module
provides ``build_waccm_constituents()`` which returns a dict containing WACCM
O₃, NO₂, and bimodal MAM4 sulfate aerosol — enough to represent the SAI
atmospheric state.

Usage
-----
Pass the returned dict to ``simulator.run()`` via the ``constituents`` key::

    from cesm_hawc.constituents import build_waccm_constituents

    constituents = build_waccm_constituents(profiles, alt_grid_m)
    data = simulator.run(
        ["l2", "sk2_atmosphere"],
        {**sim_input, "constituents": constituents},
    )

Do **not** wrap it in ``Atmosphere(constituents=...)`` — that bypasses the
Hamilton DAG and the aerosol will be silently dropped.
"""

from __future__ import annotations

import numpy as np

try:
    import sasktran2 as sk
    from aliprocessing.l2.optical import aerosol_median_radius_db
except ImportError as e:
    raise ImportError(
        "sasktran2 and hawcsimulator must be installed. "
        "See environment.yml for the correct install method."
    ) from e


def _extinction_from_number_density(N_cm3: np.ndarray, r_um: np.ndarray,
                                     mie_db) -> np.ndarray:
    """
    Convert number density [cm⁻³] and lognormal median radius [μm] to
    extinction [m⁻¹] at 745 nm using an analytic Mie approximation.

    The analytic approximation (Q_ext = min(2, (8/3)x⁴) for x < 1, else 2)
    is used because the ALI Mie database (``aerosol_median_radius_db``) does
    not expose a per-level cross-section query API. The approximation is
    adequate for computing the prior extinction used to seed the retrieval.

    Parameters
    ----------
    N_cm3  : [cm⁻³]  number concentration per altitude level
    r_um   : [μm]    lognormal median radius per altitude level
    mie_db : MieDatabase  ALI optical property database

    Returns
    -------
    extinction_per_m : [m⁻¹]
    """
    WL_REF = 745e-9   # reference wavelength [m]
    ext_m  = np.zeros(len(N_cm3))

    for k in range(len(N_cm3)):
        if N_cm3[k] <= 0.0:
            continue
        r_eff_m = r_um[k] * 1e-6                    # median radius → m
        x       = 2.0 * np.pi * r_eff_m / WL_REF
        q_ext   = min(2.0, (8.0 / 3.0) * x ** 4) if x < 1.0 else 2.0
        C_ext   = q_ext * np.pi * r_eff_m ** 2      # [m²]
        ext_m[k] = C_ext * N_cm3[k] * 1e6           # cm⁻³ → m⁻³

    return ext_m


def build_waccm_constituents(profiles: dict, alt_m: np.ndarray) -> dict:
    """
    Build the sasktran2 constituents dict from WACCM column profiles.

    This is the primary entry point for feeding CESM/WACCM data into the
    HAWC ALI simulator. The returned dict should be passed to
    ``simulator.run()`` via the ``constituents`` key (see module docstring).

    The function uses the simulator's own ``aerosol_median_radius_db()`` Mie
    database, ensuring that the forward model and L2 retrieval use the same
    optical properties.

    Parameters
    ----------
    profiles : dict
        Output of ``WACCMAtmosphere.get_column_profiles()``.
    alt_m : np.ndarray
        Altitude grid [m], must match the ``altitudes_m`` key in profiles
        and the ``altitude_grid`` key in ``sim_input``.

    Returns
    -------
    dict
        sasktran2 constituents dict with keys:
        ``o3``, ``no2``, ``aerosol_accum``, ``aerosol_coarse``.

        The simulator's default atmosphere adds ``rayleigh``,
        ``solar_irradiance``, and ``albedo`` automatically.

    Notes
    -----
    Both MAM4 modes are included:

    - ``aerosol_accum``  (so4_a1, σ_g = 1.8): fresh SO₂ injection signal
    - ``aerosol_coarse`` (so4_a3, σ_g = 1.2): aged sulfate, dominates ALI
      extinction after ~2 weeks post-injection
    """
    mie_db = aerosol_median_radius_db()
    ds     = mie_db.load_ds()
    r_min  = float(ds.median_radius.min())
    r_max  = float(ds.median_radius.max())

    constituents: dict = {}

    # ── Override MIPAS O₃ with WACCM O₃ ──────────────────────────────────
    constituents["o3"] = sk.constituent.VMRAltitudeAbsorber(
        sk.optical.O3DBM(),
        altitudes_m=alt_m,
        vmr=profiles["vmr_o3"],
    )

    # ── NO₂ (zeros if not in file — negligible at ALI wavelengths) ────────
    constituents["no2"] = sk.constituent.VMRAltitudeAbsorber(
        sk.optical.NO2Vandaele(),
        altitudes_m=alt_m,
        vmr=profiles["vmr_no2"],
    )

    # ── MAM4 bimodal stratospheric sulfate ────────────────────────────────
    for name, N_key, r_key in [
        ("aerosol_accum",  "sulfate_a1_N_cm3", "sulfate_a1_r_um"),
        ("aerosol_coarse", "sulfate_a3_N_cm3", "sulfate_a3_r_um"),
    ]:
        N_cm3 = profiles[N_key]
        r_um  = profiles[r_key]

        ext_m    = _extinction_from_number_density(N_cm3, r_um, mie_db)
        r_nm_raw = r_um * 1e3

        # Zero out sub-database-floor levels (negligible aerosol loading)
        ext_safe = np.where(r_nm_raw < r_min, 0.0, ext_m)
        r_nm     = np.clip(r_nm_raw, r_min, r_max)

        constituents[name] = sk.constituent.ExtinctionScatterer(
            mie_db,
            altitudes_m              = alt_m,
            extinction_per_m         = ext_safe,
            extinction_wavelength_nm = 745.0,
            median_radius            = r_nm,
        )

    return constituents
