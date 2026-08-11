"""
cesm_hawc.orbit_files
======================
Real HAWC orbit-track NetCDF file handling: reading observation geometry
(time/lat/lon/observer position) and indexing files by calendar day.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

_ORBIT_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def load_orbit_files(orbit_dir: str, pattern: str = "orbit_*.nc") -> list[str]:
    """Return sorted list of orbit file paths matching ``pattern``."""
    files = sorted(glob.glob(os.path.join(orbit_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No orbit files matching '{pattern}' in {orbit_dir}")
    return files


def orbit_file_start_time(path: str) -> pd.Timestamp:
    """Read an orbit file's ``start_time`` global attribute."""
    ds = xr.open_dataset(path, decode_times=False)
    try:
        return pd.Timestamp(ds.attrs["start_time"])
    finally:
        ds.close()


def orbit_file_date(path: str) -> str:
    """Return the ``YYYY-MM-DD`` calendar date of an orbit file, from its
    ``start_time`` attribute, falling back to a date parsed from the
    filename if the attribute is missing."""
    try:
        return str(orbit_file_start_time(path).date())
    except (KeyError, ValueError):
        m = _ORBIT_DATE_RE.search(os.path.basename(path))
        if m:
            return m.group(1)
        raise ValueError(f"Cannot determine calendar date for orbit file: {path}")


def collect_orbit_files_by_date(orbit_dir: str, pattern: str = "orbit_*.nc"
                                 ) -> dict[str, list[str]]:
    """Return ``{"YYYY-MM-DD": [orbit_path, ...]}`` grouped by calendar
    date (an orbit file may cover only part of a day, so a date can map to
    more than one file)."""
    grouped: dict[str, list[str]] = {}
    for p in load_orbit_files(orbit_dir, pattern):
        grouped.setdefault(orbit_file_date(p), []).append(p)
    return grouped


def _orbit_files_fingerprint(orbit_files: list[str]) -> str:
    """Cheap fingerprint (paths + mtimes + sizes) of an orbit file set, used
    to detect when a cached day-index is stale."""
    parts = [f"{f}:{os.stat(f).st_mtime_ns}:{os.stat(f).st_size}" for f in orbit_files]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def build_orbit_day_index(orbit_files: list[str], epoch: pd.Timestamp,
                           cache_path: str | os.PathLike | None = None
                           ) -> dict[int, list[str]]:
    """
    Map orbit-calendar day-of-sequence (0-indexed from ``epoch``) to the
    list of orbit file paths covering that day.

    Reading every file's ``start_time`` attribute can take minutes for a
    large file set, so if ``cache_path`` is given the resulting mapping is
    cached to disk as JSON and only rebuilt when the file set's fingerprint
    (paths/mtimes/sizes) changes.
    """
    fingerprint = _orbit_files_fingerprint(orbit_files)
    cache_path = Path(cache_path) if cache_path is not None else None

    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("fingerprint") == fingerprint:
                return {int(k): v for k, v in cached["day_index"].items()}
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning("Orbit day index cache unreadable, rebuilding: %s", e)

    day_index: dict[int, list[str]] = {}
    epoch_date = epoch.normalize()
    for f in orbit_files:
        t = orbit_file_start_time(f)
        day = (t.normalize() - epoch_date).days
        day_index.setdefault(day, []).append(f)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"fingerprint": fingerprint, "day_index": day_index}, f)
        except OSError as e:
            log.warning("Could not write orbit day index cache: %s", e)

    return day_index


def extract_observations(
    orbit_files: list[str],
    sim_date: pd.Timestamp,
    cadence_s: float,
    center_pixel: int,
    epoch: pd.Timestamp,
) -> list[dict]:
    """
    Extract subsampled observations from one day's worth of orbit files.

    For each file: read time/lat/lon/observer position at ``center_pixel``,
    subsample to every ``cadence_s`` seconds, and replace the orbit epoch's
    calendar date with ``sim_date`` while keeping the real time-of-day and
    real satellite geometry.

    Returns a list of dicts: ``{time, lat, lon, observer_lat, observer_lon,
    observer_alt}``.
    """
    observations: list[dict] = []
    epoch_date = epoch.normalize()

    for f in sorted(orbit_files):
        # decode_times=False: "time" is treated as raw integer seconds since
        # `epoch` below, not as an absolute CF-decoded datetime -- whether
        # xarray auto-decodes this variable depends on exactly which time
        # attrs happen to be present on a given orbit file, so this must be
        # explicit rather than relying on the file's own metadata.
        ds = xr.open_dataset(f, decode_times=False)
        time_s = ds["time"].values
        lats = ds["latitude"].values[:, 0, center_pixel]
        lons = ds["longitude"].values[:, 0, center_pixel]
        obs_lats = ds["observer_latitude"].values
        obs_lons = ds["observer_longitude"].values
        obs_alts = ds["observer_altitude"].values
        ds.close()

        orbit_times = [epoch_date + pd.Timedelta(seconds=int(t)) for t in time_s]

        prev_idx = -cadence_s  # ensure first point is always included
        for i, (t_orbit, lat, lon) in enumerate(zip(orbit_times, lats, lons)):
            if i - prev_idx < cadence_s:
                continue
            time_of_day = t_orbit - t_orbit.normalize()
            sim_time = sim_date.normalize() + time_of_day
            observations.append({
                "time": sim_time,
                "lat": float(lat),
                "lon": float(lon),
                "observer_lat": float(obs_lats[i]),
                "observer_lon": float(obs_lons[i]),
                "observer_alt": float(obs_alts[i]),
            })
            prev_idx = i

    return observations


def l1b_image_to_dataset(l1b, wavelengths_nm, true_extinction: dict | None = None,
                          alt_grid_m=None) -> xr.Dataset:
    """
    Combine an ``L1bImage``'s 'I' and 'dolp' spectra into a single
    ``xr.Dataset`` with dims ``(wavelength, altitude_m)``, suitable for
    ``xr.concat`` across observations along a new ``along_track`` dimension.

    If ``true_extinction`` is given (the second dict returned by
    ``build_waccm_constituents(..., return_extinction=True)``), its
    per-mode extinction fields are attached both on their native
    ``atm_altitude_m`` grid (``alt_grid_m``, exact) and interpolated onto
    the instrument's own ``altitude_m`` grid for direct point-by-point
    comparison against radiance/dolp.
    """
    I_ds = l1b.spectra["I"].ds
    dolp_ds = l1b.spectra["dolp"].ds

    ds = xr.Dataset(
        data_vars={
            "radiance": (("wavelength", "altitude_m"), I_ds["radiance"].values),
            "radiance_noise": (("wavelength", "altitude_m"), I_ds["radiance_noise"].values),
            "dolp": (("wavelength", "altitude_m"), dolp_ds["radiance"].values),
            "dolp_noise": (("wavelength", "altitude_m"), dolp_ds["radiance_noise"].values),
        },
        coords={
            "wavelength": np.asarray(wavelengths_nm),
            "altitude_m": I_ds["tangent_altitude"].values,
            "tangent_latitude": ("altitude_m", I_ds["tangent_latitude"].values),
            "tangent_longitude": ("altitude_m", I_ds["tangent_longitude"].values),
            "solar_zenith_angle": ("altitude_m", I_ds["solar_zenith_angle"].values),
        },
    )
    ds.attrs["time"] = str(I_ds["time"].values)

    if true_extinction and alt_grid_m is not None:
        instrument_alt = ds["altitude_m"].values
        for ext_key, ext_vals in true_extinction.items():
            if ext_key == "extinction_wavelength_nm":
                continue
            ds[f"{ext_key}_atm"] = (("wavelength", "atm_altitude_m"), ext_vals)
            interp_vals = np.array([
                np.interp(instrument_alt, alt_grid_m, ext_vals[i, :])
                for i in range(ext_vals.shape[0])
            ])
            ds[ext_key] = (("wavelength", "altitude_m"), interp_vals)
        ds = ds.assign_coords(atm_altitude_m=alt_grid_m)

    return ds
