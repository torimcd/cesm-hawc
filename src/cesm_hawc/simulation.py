"""Run the HAWC ALI simulator on WACCM columns.

Uses the ``ideal_dolp_imager`` configuration of ``hawcsimulator``'s
``IdealALISimulator``. Requires the ``[sim]`` extra.

Attributes
----------
DEFAULT_PRODUCTS : tuple of str
    Simulator products for a run with the L2 retrieval.
FORWARD_PRODUCTS : tuple of str
    Simulator products for a forward-model-only run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cesm_hawc.waccm import WACCMAtmosphere
from cesm_hawc.constituents import build_waccm_constituents

try:
    from hawcsimulator.ali.configurations.ideal_dolp_imager import IdealALISimulator
    from hawcsimulator.noise import ALINoiseModel
except ImportError as e:
    raise ImportError("hawcsimulator must be installed: pip install cesm-hawc[sim]") from e


DEFAULT_PRODUCTS = ("l2", "sk2_atmosphere", "front_end_radiance", "l1b")
FORWARD_PRODUCTS = ("front_end_radiance", "l1b")


def products_for(run_l2: bool) -> tuple:
    """Choose which simulator products to request.

    Parameters
    ----------
    run_l2 : bool
        Whether the L2 retrieval should run.

    Returns
    -------
    tuple of str
        :data:`DEFAULT_PRODUCTS` if ``run_l2``, else :data:`FORWARD_PRODUCTS`.
    """
    return DEFAULT_PRODUCTS if run_l2 else FORWARD_PRODUCTS


def run_ali_simulation_from_profiles(
    profiles: dict,
    alt_m: np.ndarray,
    sim_geometry: dict,
    *,
    simulator: "IdealALISimulator | None" = None,
    products: tuple = DEFAULT_PRODUCTS,
    noise_model: ALINoiseModel,
    return_extinction: bool = False,
    truth_wavelengths_nm: np.ndarray | None = None,
):
    """Run the simulator on profiles that have already been extracted.

    Use this when running many observations from one file: open one
    :class:`~cesm_hawc.waccm.WACCMAtmosphere` (and optionally one
    ``IdealALISimulator``) and reuse them, rather than calling
    :func:`run_ali_simulation`, which reopens the file each time.

    Parameters
    ----------
    profiles : dict
        Result of :meth:`cesm_hawc.waccm.WACCMAtmosphere.get_column_profiles`.
    alt_m : numpy.ndarray
        Altitude grid [m], matching ``profiles["altitudes_m"]``.
    sim_geometry : dict
        Inputs for ``simulator.run()``: ``tangent_latitude``,
        ``tangent_longitude``, ``altitude_grid``, ``polarization_states``,
        ``sample_wavelengths`` and ``time``, plus either
        ``observer_latitude``/``observer_longitude``/``observer_altitude``
        or ``tangent_solar_zenith_angle``/``tangent_solar_azimuth_angle``.
        Must not include ``constituents`` or ``l1b_cfg``, which this
        function sets.
    simulator : IdealALISimulator, optional
        Simulator to reuse. A new one is constructed if omitted.
    products : tuple of str, optional
        Products to request. Default :data:`DEFAULT_PRODUCTS`.
    noise_model : hawcsimulator.noise.ALINoiseModel
        Required. Use :func:`cesm_hawc.noise.default_noise_model`.
    return_extinction : bool, optional
        Also return the truth extinction. Default False.
    truth_wavelengths_nm : array-like, optional
        Wavelengths [nm] for the truth extinction. Default ``[745.0]``.

    Returns
    -------
    data : dict
        The ``simulator.run()`` result, keyed by product name.
    true_extinction : dict
        Only if ``return_extinction`` is True; see
        :func:`cesm_hawc.constituents.build_waccm_constituents`.

    Raises
    ------
    ValueError
        If ``noise_model`` is None. The ``ideal_dolp_imager`` instrument
        model has no noiseless mode.
    """
    if noise_model is None:
        raise ValueError(
            "noise_model is required -- IdealALISimulator (ideal_dolp_imager) "
            "has no noiseless fallback. Pass cesm_hawc.noise.default_noise_model()."
        )
    if simulator is None:
        simulator = IdealALISimulator()

    sim_input = dict(sim_geometry)
    sim_input["l1b_cfg"] = {"noise_model": noise_model}

    if return_extinction:
        constituents, true_ext = build_waccm_constituents(
            profiles, alt_m, return_extinction=True,
            truth_wavelengths_nm=truth_wavelengths_nm,
        )
        data = simulator.run(list(products), {**sim_input, "constituents": constituents})
        return data, true_ext

    constituents = build_waccm_constituents(profiles, alt_m)
    data = simulator.run(list(products), {**sim_input, "constituents": constituents})
    return data


def run_ali_simulation(
    waccm_file: str,
    *,
    lat: float,
    lon: float,
    time_index: int = 0,
    sza_deg: float = 60.0,
    saa_deg: float = 0.0,
    obs_time: str | pd.Timestamp = "2035-01-01T12:00:00Z",
    wavelengths_nm: np.ndarray | None = None,
    alt_grid_m: np.ndarray | None = None,
    run_l2: bool = False,
    noise_model: ALINoiseModel,
) -> dict:
    """Simulate one column of a WACCM history file at a fixed geometry.

    Parameters
    ----------
    waccm_file : str
        CAM history file (any stream, e.g. h0 or h2).
    lat, lon : float
        Tangent point [degrees].
    time_index : int, optional
        Time slice within the file. Default 0.
    sza_deg, saa_deg : float, optional
        Solar zenith and azimuth angles at the tangent point [degrees].
        Defaults 60 and 0.
    obs_time : str or pandas.Timestamp, optional
        Observation time.
    wavelengths_nm : array-like, optional
        Simulated wavelengths [nm]. Default ``[470, 745, 1020]``.
    alt_grid_m : array-like, optional
        Altitude grid [m]. Default 0–65 km every 1 km.
    run_l2 : bool, optional
        Also run the L2 retrieval. Default False (forward model only).
    noise_model : hawcsimulator.noise.ALINoiseModel
        Required. Use :func:`cesm_hawc.noise.default_noise_model`.

    Returns
    -------
    dict
        ``data``: the ``simulator.run()`` result (``l1b``, plus ``l2`` when
        ``run_l2``). ``true_extinction``: truth extinction at
        ``wavelengths_nm`` (see
        :func:`cesm_hawc.constituents.build_waccm_constituents`).
        ``burden``: the column's sulfate burden (see
        :meth:`cesm_hawc.waccm.WACCMAtmosphere.sulfate_column_burden`).

    Raises
    ------
    ValueError
        If ``noise_model`` is None.

    Examples
    --------
    >>> from cesm_hawc.noise import default_noise_model
    >>> result = run_ali_simulation(
    ...     "case.cam.h0.2035-02.nc", lat=30.6, lon=180.0,
    ...     run_l2=True, noise_model=default_noise_model(),
    ... )
    >>> result["data"]["l2"]["stratospheric_aerosol_extinction_per_m"]
    """
    if noise_model is None:
        raise ValueError(
            "noise_model is required -- IdealALISimulator (ideal_dolp_imager) "
            "has no noiseless fallback. Pass cesm_hawc.noise.default_noise_model()."
        )
    if wavelengths_nm is None:
        wavelengths_nm = np.array([470.0, 745.0, 1020.0])
    if alt_grid_m is None:
        alt_grid_m = np.arange(0.0, 65001.0, 1000.0)
    if not isinstance(obs_time, pd.Timestamp):
        obs_time = pd.Timestamp(obs_time)

    sim_geometry = {
        "tangent_latitude":            lat,
        "tangent_longitude":           lon,
        "tangent_solar_zenith_angle":  sza_deg,
        "tangent_solar_azimuth_angle": saa_deg,
        "altitude_grid":               alt_grid_m,
        "polarization_states":         ["I", "dolp"],
        "sample_wavelengths":          wavelengths_nm,
        "time":                        obs_time,
    }

    waccm = WACCMAtmosphere(waccm_file, alt_grid_km=alt_grid_m / 1e3)
    profiles = waccm.get_column_profiles(lat, lon, time_index)
    data, true_ext = run_ali_simulation_from_profiles(
        profiles, alt_grid_m, sim_geometry,
        products=products_for(run_l2), noise_model=noise_model,
        return_extinction=True, truth_wavelengths_nm=wavelengths_nm,
    )
    return {
        "data": data,
        "true_extinction": true_ext,
        "burden": waccm.sulfate_column_burden(lat, lon, time_index),
    }
