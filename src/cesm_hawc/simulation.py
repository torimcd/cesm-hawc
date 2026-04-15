"""
cesm_hawc.simulation
====================
High-level wrapper for running the HAWC ALI simulator on WACCM output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.cesm_hawc.waccm import WACCMAtmosphere
from src.cesm_hawc.constituents import build_waccm_constituents

try:
    from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator
except ImportError as e:
    raise ImportError("hawcsimulator must be installed: pip install hawcsimulator") from e


def run_ali_simulation(
    background_file: str,
    injection_file: str | None = None,
    lat: float = 15.0,
    lon: float = 0.0,
    time_index: int = 0,
    sza_deg: float = 60.0,
    saa_deg: float = 0.0,
    obs_time: str | pd.Timestamp = "2035-01-01T12:00:00Z",
    wavelengths_nm: np.ndarray | None = None,
    alt_grid_m: np.ndarray | None = None,
) -> dict:
    """
    Run the HAWC IdealALISimulator on a WACCM background and optional
    injection scenario, and return a summary of retrieved quantities.

    Parameters
    ----------
    background_file : str
        Path to WACCM h0 file for the no-injection (reference) run.
    injection_file : str, optional
        Path to WACCM h0 file for the SAI injection run.
    lat, lon : float
        Observation tangent point coordinates [degrees].
    time_index : int
        Time slice index within the file (0-based).
    sza_deg, saa_deg : float
        Solar zenith and azimuth angles [degrees].
    obs_time : str or pd.Timestamp
        Observation time (used for solar position in the simulator).
    wavelengths_nm : array, optional
        ALI sample wavelengths [nm].
        Default: [470, 745, 1020] nm (quickstart channels).
    alt_grid_m : array, optional
        Altitude grid [m]. Default: 0–65 km in 1 km steps.

    Returns
    -------
    dict with keys:
        data_bg                    : simulator output for background
        data_inj                   : simulator output for injection (or None)
        burden_bg                  : sulfate column burden dict (background)
        burden_inj                 : sulfate column burden dict (injection)
        peak_extinction_anomaly_m  : peak Δ extinction [m⁻¹] above 15 km
        peak_radius_anomaly_nm     : peak Δ median radius [nm] above 15 km
        delta_burden_mg_m2         : Δ SO₄ column burden [mg m⁻²]

    Examples
    --------
    >>> result = run_ali_simulation(
    ...     "background.cam.h0.2035-02.nc",
    ...     "injection.cam.h0.2035-02.nc",
    ...     lat=30.6, lon=180.0,
    ... )
    >>> print(f"Peak extinction anomaly: {result['peak_extinction_anomaly_m']:.2e} m⁻¹")
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.array([470.0, 745.0, 1020.0])
    if alt_grid_m is None:
        alt_grid_m = np.arange(0.0, 65001.0, 1000.0)
    if not isinstance(obs_time, pd.Timestamp):
        obs_time = pd.Timestamp(obs_time)

    simulator = IdealALISimulator()

    sim_input = {
        "tangent_latitude":           lat,
        "tangent_longitude":          lon,
        "tangent_solar_zenith_angle":  sza_deg,
        "tangent_solar_azimuth_angle": saa_deg,
        "altitude_grid":               alt_grid_m,
        "polarization_states":         ["I", "dolp"],
        "sample_wavelengths":          wavelengths_nm,
        "time":                        obs_time,
    }

    # ── Background ────────────────────────────────────────────────────────
    waccm_bg   = WACCMAtmosphere(background_file, alt_grid_km=alt_grid_m / 1e3)
    profiles_bg = waccm_bg.get_column_profiles(lat, lon, time_index)
    data_bg    = simulator.run(
        ["l2", "sk2_atmosphere"],
        {**sim_input, "constituents": build_waccm_constituents(profiles_bg, alt_grid_m)},
    )
    burden_bg = waccm_bg.sulfate_column_burden(lat, lon, time_index)

    # ── Injection (optional) ──────────────────────────────────────────────
    data_inj, burden_inj, waccm_inj = None, None, None
    if injection_file is not None:
        waccm_inj    = WACCMAtmosphere(injection_file, alt_grid_km=alt_grid_m / 1e3)
        profiles_inj = waccm_inj.get_column_profiles(lat, lon, time_index)
        data_inj     = simulator.run(
            ["l2", "sk2_atmosphere"],
            {**sim_input,
             "constituents": build_waccm_constituents(profiles_inj, alt_grid_m)},
        )
        burden_inj = waccm_inj.sulfate_column_burden(lat, lon, time_index)

    # ── Derived anomaly quantities ─────────────────────────────────────────
    peak_ext_anom = None
    peak_r_anom   = None
    delta_burden  = None

    if data_inj is not None:
        ext_bg  = data_bg["l2"]["stratospheric_aerosol_extinction_per_m"]
        ext_inj = data_inj["l2"]["stratospheric_aerosol_extinction_per_m"]
        r_bg    = data_bg["l2"]["stratospheric_aerosol_median_radius"]
        r_inj   = data_inj["l2"]["stratospheric_aerosol_median_radius"]

        strat = ext_bg.altitude.values > 15000  # above 15 km

        peak_ext_anom = float((ext_inj - ext_bg).values[strat].max())
        peak_r_anom   = float((r_inj - r_bg).values[strat].max())   # already nm
        delta_burden  = burden_inj["burden_mg_m2"] - burden_bg["burden_mg_m2"]

    return {
        "data_bg":                   data_bg,
        "data_inj":                  data_inj,
        "burden_bg":                 burden_bg,
        "burden_inj":                burden_inj,
        "peak_extinction_anomaly_m": peak_ext_anom,
        "peak_radius_anomaly_nm":    peak_r_anom,
        "delta_burden_mg_m2":        delta_burden,
    }
