"""Build the ``sasktran2`` constituents that represent a WACCM column.

``hawcsimulator``'s default atmosphere contains Rayleigh scattering, MIPAS
climatological ozone, solar irradiance and a Lambertian surface with albedo
0.3. :func:`build_waccm_constituents` returns WACCM ozone (replacing the
MIPAS ozone), WACCM NO2 and the two MAM4 sulfate modes, to be passed to
``simulator.run()`` under the ``constituents`` key::

    constituents = build_waccm_constituents(profiles, alt_grid_m)
    data = simulator.run(products, {**sim_input, "constituents": constituents})

Do not wrap the dict in ``Atmosphere(constituents=...)``: that bypasses the
simulator's atmosphere step and the aerosol is silently dropped.

Each sulfate mode is an ``ExtinctionScatterer`` backed by its own Mie
database, built for a lognormal size distribution with that mode's width
(``sigma_g`` = 1.6 for accumulation, 1.2 for coarse) and the H2SO4
refractive index. ``sasktran2`` integrates over the size distribution, and
the same database supplies extinction and phase function at every
wavelength. The L2 retrieval separately assumes a single mode with
``sigma_g`` = 1.6, via ``aliprocessing.l2.optical.aerosol_median_radius_db``.

Requires the ``[sim]`` extra.
"""

from __future__ import annotations

import numpy as np

try:
    import sasktran2 as sk
except ImportError as e:
    raise ImportError("sasktran2 must be installed: pip install cesm-hawc[sim]") from e


_MODE_WIDTHS = {"aerosol_accum": 1.6, "aerosol_coarse": 1.2}
_WAVELENGTHS_NM = np.array([470, 525, 745, 869, 1020, 1230, 1450, 1500, 1560,
                             1750, 2000, 2250, 2500])
_MEDIAN_RADIUS_NM = np.arange(10, 600, 10.0)

_mode_dbs: dict = {}


def _get_mode_db(mode_width: float):
    """Build, or return the cached, Mie database for one mode width.

    The first build runs a Mie calculation, which ``sasktran2`` caches on
    disk; the in-memory database is also cached for this process. Applies
    the same single-scattering-albedo clamp as ``aliprocessing`` (values
    >= 1 set to 0.99999, with ``xs_scattering`` recomputed to match).
    """
    if mode_width not in _mode_dbs:
        refrac = sk.mie.refractive.H2SO4()
        dist = sk.mie.distribution.LogNormalDistribution().freeze(mode_width=mode_width)
        db = sk.database.MieDatabase(
            dist, refrac, _WAVELENGTHS_NM, median_radius=_MEDIAN_RADIUS_NM,
        )
        db.path()  # triggers build/cache-to-disk if not already present

        # mirror aliprocessing's SSA clamp
        ssa = db._database["xs_scattering"] / db._database["xs_total"]
        ssa.to_numpy()[ssa.to_numpy() >= 1] = 0.99999
        db._database["xs_scattering"] = ssa * db._database["xs_total"]

        _mode_dbs[mode_width] = db
    return _mode_dbs[mode_width]


def get_mode_mie_database(mode_width: float):
    """Return the Mie database for one sulfate mode.

    Use this to rebuild an ``sk.constituent.ExtinctionScatterer`` yourself
    from a saved column's ``{mode}_reference_extinction_per_m`` and
    ``{mode}_median_radius_nm`` (see :mod:`cesm_hawc.save_inputs`).

    Parameters
    ----------
    mode_width : float
        Geometric standard deviation of the mode: 1.6 for accumulation,
        1.2 for coarse.

    Returns
    -------
    sasktran2.database.MieDatabase
        The database, built on first use.
    """
    return _get_mode_db(mode_width)


def warm_mode_databases() -> None:
    """Build both sulfate-mode Mie databases.

    Call this in the main process before starting worker processes, so the
    databases are built once rather than concurrently by every worker.
    """
    for mode_width in set(_MODE_WIDTHS.values()):
        _get_mode_db(mode_width)


def _extinction_from_xs_total(N_cm3: np.ndarray, r_um: np.ndarray,
                               mode_width: float,
                               wavelength_nm=745.0) -> np.ndarray:
    """Extinction [m⁻¹] from number density and median radius.

    Computes ``N * xs_total(r, wavelength)``, where ``xs_total`` [m²] is the
    distribution-weighted total cross-section from the mode's Mie database.
    Radii are clipped to the database range and wavelengths snap to the
    nearest database wavelength.

    Parameters
    ----------
    N_cm3 : numpy.ndarray
        Number concentration per level [cm⁻³].
    r_um : numpy.ndarray
        Lognormal median radius per level [μm].
    mode_width : float
        Geometric standard deviation of the mode.
    wavelength_nm : float or array-like, optional
        Wavelength(s) [nm]. Default 745.

    Returns
    -------
    numpy.ndarray
        Extinction [m⁻¹], shape (altitude,) for a scalar wavelength or
        (wavelength, altitude) for several.
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
    return N_m3 * xs_interp  # [m^-1]


def build_waccm_constituents(profiles: dict, alt_m: np.ndarray,
                              return_extinction: bool = False,
                              truth_wavelengths_nm=None):
    """Build the ``sasktran2`` constituents for one WACCM column.

    Parameters
    ----------
    profiles : dict
        Result of :meth:`cesm_hawc.waccm.WACCMAtmosphere.get_column_profiles`.
    alt_m : numpy.ndarray
        Altitude grid [m]; must match ``profiles["altitudes_m"]`` and the
        simulator's ``altitude_grid``.
    return_extinction : bool, optional
        Also return the truth extinction. Default False.
    truth_wavelengths_nm : array-like, optional
        Wavelengths [nm] for the truth extinction; normally the simulated
        wavelengths. Default ``[745.0]``.

    Returns
    -------
    constituents : dict
        ``o3`` and ``no2`` (``VMRAltitudeAbsorber``) and ``aerosol_accum``
        and ``aerosol_coarse`` (``ExtinctionScatterer``, from ``so4_a1`` and
        ``so4_a3``).
    true_extinction : dict
        Only if ``return_extinction`` is True. For each mode
        (``aerosol_accum``, ``aerosol_coarse``):

        - ``{mode}_extinction_per_m``: truth extinction [m⁻¹], shape
          (wavelength, altitude), at ``truth_wavelengths_nm``;
        - ``{mode}_reference_extinction_per_m``: extinction at 745 nm
          [m⁻¹], shape (altitude,);
        - ``{mode}_median_radius_nm``: clipped median radius [nm], shape
          (altitude,);

        plus ``extinction_wavelength_nm``. The reference extinction and
        median radius are exactly the arguments used to build that mode's
        ``ExtinctionScatterer``, so it can be rebuilt from them with
        :func:`get_mode_mie_database`.

    Notes
    -----
    Median radii are clipped to the Mie database range (10–590 nm), and
    extinction is set to zero where the radius is below 10 nm.
    """
    r_min = float(_MEDIAN_RADIUS_NM.min())
    r_max = float(_MEDIAN_RADIUS_NM.max())

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

        mode_db = _get_mode_db(_MODE_WIDTHS[name])

        # reference extinction at 745 nm — drives ExtinctionScatterer's
        # number-density conversion; the RT solver then queries mode_db
        # (correctly mode-width-matched) at every other wavelength too
        ext_m_ref = _extinction_from_xs_total(
            N_cm3, r_um, _MODE_WIDTHS[name], wavelength_nm=745.0
        )
        ext_ref_safe = np.where(r_nm_raw < r_min, 0.0, ext_m_ref)
        r_nm = np.clip(r_nm_raw, r_min, r_max)

        constituents[name] = sk.constituent.ExtinctionScatterer(
            mode_db,
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
            # same values just passed to ExtinctionScatterer above,
            # so a saved column can be turned back into an equivalent
            # ExtinctionScatterer without re-deriving them from N/r.
            true_extinction[f"{name}_reference_extinction_per_m"] = ext_ref_safe
            true_extinction[f"{name}_median_radius_nm"] = r_nm

    if return_extinction:
        true_extinction["extinction_wavelength_nm"] = truth_wavelengths_nm
        return constituents, true_extinction
    return constituents