"""
cesm_hawc.save_inputs
======================
Save a WACCM column as a simulator-ready input file.

``save_column_inputs()`` always saves the same WACCM-derived profile
fields as ``WACCMAtmosphere.save_column_profiles()``. When ``sasktran2``
is importable, it additionally saves the constituents-level data an
external ``hawcsimulator``/``sasktran2`` user needs to reconstruct the
simulator's aerosol/gas constituent objects directly -- without calling
back into this package at all. See the README's "Consuming saved inputs
externally" section for a complete example.
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr


def save_column_inputs(waccm, lat: float, lon: float, output_path: str,
                        time_index: int, alt_m: np.ndarray,
                        wavelengths_nm=None, profiles_only: bool = False,
                        obs_time=None) -> None:
    """
    Extract one WACCM column and save it as a simulator-ready input file.

    Parameters
    ----------
    waccm : cesm_hawc.waccm.WACCMAtmosphere
    lat, lon : float
        Column coordinates [degrees].
    output_path : str
        Output NetCDF path.
    time_index : int
        Time slice index within the source file.
    alt_m : np.ndarray
        Altitude grid [m] -- must match ``waccm``'s own grid.
    wavelengths_nm : array-like, optional
        Wavelengths [nm] to save truth extinction at, when sasktran2 is
        available. Defaults to [745.0] if not given (see
        ``build_waccm_constituents``).
    profiles_only : bool, optional
        If True, skip the constituents computation even when sasktran2 is
        available (e.g. for minimal-footprint massive batch runs).
        Default False.
    obs_time : optional
        Real observation timestamp (e.g. a ``pd.Timestamp``), if known --
        saved as the ``time`` attr so a later simulator run from this file
        can match the original observation's time instead of guessing.
        Not saved if omitted (e.g. `single`/`batch` modes without a real
        per-observation time).

    Notes
    -----
    Always saves the WACCM profile fields (same shape as
    ``WACCMAtmosphere.save_column_profiles()``). If
    ``cesm_hawc.constituents`` imports successfully (``sasktran2``
    present), also saves, per mode (``aerosol_accum``, ``aerosol_coarse``):

    - ``{mode}_extinction_per_m``            [wavelength_nm, altitude_m]  truth extinction
    - ``{mode}_reference_extinction_per_m``  [altitude_m]  745 nm reference extinction
    - ``{mode}_median_radius_nm``            [altitude_m]  clipped median radius

    plus attrs describing exactly how to rebuild an equivalent Mie
    database from raw ``sasktran2`` calls (``mie_refractive_index``,
    ``mie_wavelength_grid_nm``, ``mie_median_radius_grid_nm``,
    ``mode_width_accum``, ``mode_width_coarse``,
    ``extinction_reference_wavelength_nm``) -- an external
    ``hawcsimulator``/``sasktran2`` user can reconstruct
    ``sk.constituent.ExtinctionScatterer``/``VMRAltitudeAbsorber`` objects
    from these fields with no ``cesm_hawc`` import at all. The file's
    ``includes_constituents`` attr records which shape it has.

    Falls back to profiles-only (``includes_constituents=False``) when
    ``sasktran2`` isn't importable -- this function never itself requires
    the ``[sim]`` extra.
    """
    profiles = waccm.get_column_profiles(lat, lon, time_index)

    data_vars = {
        k: ("altitude_m", v) for k, v in profiles.items()
        if not np.isscalar(v) and k != "altitudes_m"
    }
    coords = {"altitude_m": alt_m}
    attrs = {
        "latitude": float(lat),
        "longitude": float(lon),
        "time_index": int(time_index),
        "sigma_a1": profiles.get("sulfate_a1_sigma", 1.6),
        "sigma_a3": profiles.get("sulfate_a3_sigma", 1.2),
        "description": "WACCM column profile + simulator constituents input from cesm-hawc",
    }
    if obs_time is not None:
        attrs["time"] = str(obs_time)

    if profiles_only:
        _constituents_available = False
    else:
        try:
            from cesm_hawc.constituents import (
                _MEDIAN_RADIUS_NM,
                _MODE_WIDTHS,
                _WAVELENGTHS_NM,
                build_waccm_constituents,
            )
        except ImportError:
            _constituents_available = False
        else:
            _constituents_available = True

    # NetCDF attrs can't hold a Python bool -- store as int 0/1.
    attrs["includes_constituents"] = int(_constituents_available)

    if _constituents_available:
        _, true_ext = build_waccm_constituents(
            profiles, alt_m, return_extinction=True, truth_wavelengths_nm=wavelengths_nm,
        )
        coords["wavelength_nm"] = true_ext["extinction_wavelength_nm"]
        for name in _MODE_WIDTHS:
            data_vars[f"{name}_extinction_per_m"] = (
                ("wavelength_nm", "altitude_m"), true_ext[f"{name}_extinction_per_m"]
            )
            data_vars[f"{name}_reference_extinction_per_m"] = (
                "altitude_m", true_ext[f"{name}_reference_extinction_per_m"]
            )
            data_vars[f"{name}_median_radius_nm"] = (
                "altitude_m", true_ext[f"{name}_median_radius_nm"]
            )
        attrs.update({
            "extinction_reference_wavelength_nm": 745.0,
            "mode_width_accum": _MODE_WIDTHS["aerosol_accum"],
            "mode_width_coarse": _MODE_WIDTHS["aerosol_coarse"],
            "mie_refractive_index": "H2SO4",
            "mie_wavelength_grid_nm": _WAVELENGTHS_NM.tolist(),
            "mie_median_radius_grid_nm": _MEDIAN_RADIUS_NM.tolist(),
        })

    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(output_path)

    size_kb = os.path.getsize(output_path) / 1e3
    tag = " (with constituents)" if attrs.get("includes_constituents") else ""
    print(f"Saved {output_path}  ({size_kb:.0f} KB){tag}")
