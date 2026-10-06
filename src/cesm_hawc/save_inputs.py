"""Save WACCM columns as simulator-ready NetCDF files.

The files can drive ``hawcsimulator`` directly, with native ``sasktran2``
calls and no ``cesm_hawc`` import; the documentation's "Using saved inputs"
page has a complete example.
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr


def save_column_inputs(waccm, lat: float, lon: float, output_path: str,
                        time_index: int, alt_m: np.ndarray,
                        wavelengths_nm=None, profiles_only: bool = False,
                        obs_time=None, extra_attrs: dict | None = None) -> None:
    """Extract one WACCM column and save it as a simulator-ready file.

    Parameters
    ----------
    waccm : cesm_hawc.waccm.WACCMAtmosphere
        The opened history file.
    lat, lon : float
        Column location [degrees].
    output_path : str
        NetCDF file to write.
    time_index : int
        Time slice within the history file.
    alt_m : numpy.ndarray
        Altitude grid [m]; must match ``waccm.alt_grid_m``.
    wavelengths_nm : array-like, optional
        Wavelengths [nm] for the truth extinction, when constituents are
        saved. Default ``[745.0]``.
    profiles_only : bool, optional
        Save only the WACCM profiles, even if ``sasktran2`` is installed.
        Default False.
    obs_time : optional
        Observation time (e.g. a ``pandas.Timestamp``), saved as the
        ``time`` attribute. Not saved if omitted.
    extra_attrs : dict, optional
        Further global attributes, e.g. ``observer_latitude``,
        ``observer_longitude`` and ``observer_altitude``.

    Notes
    -----
    The file always holds the profiles from
    :meth:`~cesm_hawc.waccm.WACCMAtmosphere.get_column_profiles` on the
    ``altitude_m`` dimension. When ``sasktran2`` is installed and
    ``profiles_only`` is False, it also holds, for each mode
    (``aerosol_accum``, ``aerosol_coarse``):

    - ``{mode}_extinction_per_m`` (``wavelength_nm``, ``altitude_m``):
      truth extinction [m⁻¹];
    - ``{mode}_reference_extinction_per_m`` (``altitude_m``): extinction at
      745 nm [m⁻¹];
    - ``{mode}_median_radius_nm`` (``altitude_m``): clipped median radius
      [nm];

    plus the attributes needed to rebuild the Mie databases:
    ``extinction_reference_wavelength_nm``, ``mode_width_accum``,
    ``mode_width_coarse``, ``mie_refractive_index``,
    ``mie_wavelength_grid_nm`` and ``mie_median_radius_grid_nm``. The
    ``includes_constituents`` attribute (0 or 1) records which kind of file
    it is.
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
    if extra_attrs:
        attrs.update(extra_attrs)

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

    # NetCDF attrs can't hold a Python bool; store as int 0/1.
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
