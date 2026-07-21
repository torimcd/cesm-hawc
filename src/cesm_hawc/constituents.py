"""
cesm_hawc.constituents
======================
Build the sasktran2 constituents dict that the HAWC ALI simulator needs.

The IdealALISimulator uses a Hamilton DAG. The atmosphere step
(``hawcsimulator.steps.atmosphere.atmosphere__default``) starts from:

    - Rayleigh scattering
    - MIPAS O3 climatology
    - Solar irradiance
    - Lambertian surface albedo (0.3)

and then merges whatever is in the ``constituents`` dict on top. This module
provides ``build_waccm_constituents()`` which returns a dict containing WACCM
O3, NO2, and bimodal MAM4 sulfate aerosol -- enough to represent the SAI
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

Do **not** wrap it in ``Atmosphere(constituents=...)`` -- that bypasses the
Hamilton DAG and the aerosol will be silently dropped.

Extinction calculation
-----------------------
Extinction is computed from each level's number density and lognormal
median radius using ``xs_total`` (total cross section, [m^2]) from a
sasktran2 MieDatabase built with the CORRECT mode_width (geometric standard
deviation, sigma_g) for each MAM4 mode:

    - aerosol_accum  (so4_a1, sigma_g = 1.8)
    - aerosol_coarse (so4_a3, sigma_g = 1.2)

This is distinct from ``aliprocessing.l2.optical.aerosol_median_radius_db()``,
which is still used for phase-function matching in ``ExtinctionScatterer``,
but which bakes in a single fixed mode_width=1.6 for its extinction
calculation -- not correct for either MAM4 mode. Using the mode-matched
databases here ensures the extinction magnitude reflects the actual size
distribution width of each mode, since sasktran2 builds xs_total as a
distribution-weighted sum over per-particle Mie cross-sections
(see sasktran2/mie/distribution.py), i.e. the lognormal integration is done
correctly inside sasktran2 for whatever mode_width is specified -- we just
need to specify the right one per mode, which the shared aliprocessing
database does not.
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


# ── Mode-specific Mie databases ─────────────────────────────────────────────
_MODE_WIDTHS = {"aerosol_accum": 1.8, "aerosol_coarse": 1.2}
_WAVELENGTHS_NM = np.array([470, 525, 745, 1020, 1230, 1450, 1500])
_MEDIAN_RADIUS_NM = np.arange(10, 600, 10.0)

_mode_dbs: dict = {}


def _get_mode_db(mode_width: float):
    """
    Lazily build (and cache in-process) a MieDatabase for a given
    mode_width. Building triggers a real Mie calculation the first time it's
    called for a given mode_width; sasktran2's MieDatabase caches the result
    to disk internally (like the shared aliprocessing database does), and
    this dict caches the in-memory handle to avoid rebuilding within a
    single process.
    """
    if mode_width not in _mode_dbs:
        refrac = sk.mie.refractive.H2SO4()
        dist = sk.mie.distribution.LogNormalDistribution().freeze(mode_width=mode_width)
        db = sk.database.MieDatabase(
            dist, refrac, _WAVELENGTHS_NM, median_radius=_MEDIAN_RADIUS_NM,
        )
        db.path()  # triggers build/cache-to-disk if not already present
        _mode_dbs[mode_width] = db
    return _mode_dbs[mode_width]


def warm_mode_databases() -> None:
    """
    Pre-build both mode-specific Mie databases once, before any parallel
    dispatch. Call this from the main process before spawning workers --
    mirrors the calibration_database pre-warm pattern used elsewhere in
    this project. Building once, serially, up front avoids relying on
    unconfirmed concurrent-build safety in sasktran2's MieDatabase.
    """
    for mode_width in set(_MODE_WIDTHS.values()):
        _get_mode_db(mode_width)


def _extinction_from_xs_total(N_cm3: np.ndarray, r_um: np.ndarray,
                               mode_width: float,
                               wavelength_nm=745.0) -> np.ndarray:
    """
    Convert number density [cm^-3] and lognormal median radius [um] to
    extinction [m^-1] using the mode-width-matched Mie database's xs_total
    [m^2] (a distribution-weighted total cross section), rather than an
    analytic single-radius approximation.

    extinction [m^-1] = N [m^-3] * xs_total [m^2]

    Parameters
    ----------
    N_cm3         : [cm^-3]  number concentration per altitude level
    r_um          : [um]     lognormal median radius per altitude level
    mode_width    : float    geometric standard deviation (sigma_g) of this mode
    wavelength_nm : float or array-like
        Wavelength(s) to evaluate xs_total at. A scalar (default 745.0,
        matching ExtinctionScatterer's reference wavelength) returns a 1D
        array [altitude]. An array of wavelengths returns a 2D array
        [wavelength, altitude].

    Returns
    -------
    extinction_per_m : np.ndarray
        [m^-1], shape [altitude] for scalar wavelength_nm, or
        [wavelength, altitude] for array-like wavelength_nm.
    """
    db = _get_mode_db(mode_width)
    ds = db._database

    xs_at_wl = ds["xs_total"].sel(wavelength_nm=wavelength_nm, method="nearest")

    r_nm = r_um * 1e3
    r_nm_clipped = np.clip(
        r_nm, float(ds.median_radius.min()), float(ds.median_radius.max())
    )
    xs_interp = xs_at_wl.interp(median_radius=r_nm_clipped).to_numpy()  # [m^2]

    N_m3 = N_cm3 * 1e6  # cm^-3 -> m^-3
    return N_m3 * xs_interp  # [m^-1], broadcasts correctly for both scalar and array wavelength_nm


def build_waccm_constituents(profiles: dict, alt_m: np.ndarray,
                              return_extinction: bool = False,
                              truth_wavelengths_nm=None):
    """
    Build the sasktran2 constituents dict from WACCM column profiles.

    This is the primary entry point for feeding CESM/WACCM data into the
    HAWC ALI simulator. The returned dict should be passed to
    ``simulator.run()`` via the ``constituents`` key (see module docstring).

    Parameters
    ----------
    profiles : dict
        Output of ``WACCMAtmosphere.get_column_profiles()``.
    alt_m : np.ndarray
        Altitude grid [m], must match the ``altitudes_m`` key in profiles
        and the ``altitude_grid`` key in ``sim_input``.
    return_extinction : bool, optional
        If True, also return the true per-mode extinction profiles [m^-1]
        that were computed. Default False (keeps prior return signature
        unchanged for existing callers).
    truth_wavelengths_nm : array-like, optional
        Wavelength(s) [nm] to evaluate truth extinction at, when
        return_extinction=True. Should match the wavelengths the
        simulator is being run at (e.g. the caller's ALI_WAVELENGTHS), so
        radiance and truth extinction can be directly compared per
        wavelength. Defaults to [745.0] (the ExtinctionScatterer
        reference wavelength) if not given.

    Returns
    -------
    dict
        sasktran2 constituents dict with keys:
        ``o3``, ``no2``, ``aerosol_accum``, ``aerosol_coarse``.

        The simulator's default atmosphere adds ``rayleigh``,
        ``solar_irradiance``, and ``albedo`` automatically.

    dict, optional
        If ``return_extinction=True``, also returns a second dict with
        keys ``aerosol_accum_extinction_per_m`` and
        ``aerosol_coarse_extinction_per_m``, each [m^-1] with shape
        [wavelength, altitude] (matching ``truth_wavelengths_nm`` order),
        plus ``extinction_wavelength_nm`` giving the wavelength grid used.
        Note: this is the true extinction at each requested wavelength —
        distinct from the single 745 nm reference extinction that always
        drives ExtinctionScatterer internally, regardless of
        truth_wavelengths_nm.

    Notes
    -----
    Both MAM4 modes are included:

    - ``aerosol_accum``  (so4_a1, sigma_g = 1.8): fresh SO2 injection signal
    - ``aerosol_coarse`` (so4_a3, sigma_g = 1.2): aged sulfate, dominates ALI
      extinction after ~2 weeks post-injection
    """
    # still used for phase-function (p11/p12/p33/etc) matching in
    # ExtinctionScatterer -- only the extinction magnitude comes from the
    # mode-matched databases above
    mie_db = aerosol_median_radius_db()
    ds     = mie_db.load_ds()
    r_min  = float(ds.median_radius.min())
    r_max  = float(ds.median_radius.max())

    if truth_wavelengths_nm is None:
        truth_wavelengths_nm = np.array([745.0])
    else:
        truth_wavelengths_nm = np.asarray(truth_wavelengths_nm, dtype=float)

    constituents: dict = {}
    true_extinction: dict = {}

    # ── Override MIPAS O3 with WACCM O3 ──────────────────────────────────
    constituents["o3"] = sk.constituent.VMRAltitudeAbsorber(
        sk.optical.O3DBM(),
        altitudes_m=alt_m,
        vmr=profiles["vmr_o3"],
    )

    # ── NO2 (zeros if not in file -- negligible at ALI wavelengths) ────────
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
        r_nm_raw = r_um * 1e3

        # reference extinction at 745 nm — this always drives
        # ExtinctionScatterer, independent of truth_wavelengths_nm
        ext_m_ref = _extinction_from_xs_total(
            N_cm3, r_um, _MODE_WIDTHS[name], wavelength_nm=745.0
        )
        ext_ref_safe = np.where(r_nm_raw < r_min, 0.0, ext_m_ref)
        r_nm = np.clip(r_nm_raw, r_min, r_max)

        constituents[name] = sk.constituent.ExtinctionScatterer(
            mie_db,
            altitudes_m              = alt_m,
            extinction_per_m         = ext_ref_safe,
            extinction_wavelength_nm = 745.0,
            median_radius            = r_nm,
        )

        if return_extinction:
            # multi-wavelength truth extinction, shape [wavelength, altitude]
            ext_multi = _extinction_from_xs_total(
                N_cm3, r_um, _MODE_WIDTHS[name], wavelength_nm=truth_wavelengths_nm
            )
            ext_multi_safe = np.where(r_nm_raw[None, :] < r_min, 0.0, ext_multi)
            true_extinction[f"{name}_extinction_per_m"] = ext_multi_safe

    if return_extinction:
        true_extinction["extinction_wavelength_nm"] = truth_wavelengths_nm
        return constituents, true_extinction
    return constituents
