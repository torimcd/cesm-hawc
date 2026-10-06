"""
cesm_hawc.orbit_files
======================
HAWC orbit NetCDF file handling: reading observation geometry
(time/lat/lon/observer position), indexing files by day since the orbit
epoch, and converting simulator L1b output to a dataset.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


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
    files_sorted = sorted(orbit_files)
    n_files = len(files_sorted)
    cadence_int = int(round(cadence_s))

    for file_idx, f in enumerate(files_sorted):
        # decode_times=False: "time" is treated as raw integer seconds since
        # `epoch` below, not as an absolute CF-decoded datetime. Whether
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

        n = len(time_s)
        if n == 0:
            continue

        # Rotate this file's starting phase instead of always starting at 0.
        # Vectorized (index arithmetic instead of a per-sample
        # Python loop) since this now runs once per file per
        # simulated date across a full production run.
        phase_shift = (file_idx * cadence_s / n_files) % cadence_s
        first_idx = int(round(phase_shift))
        idx = np.arange(first_idx, n, cadence_int)

        selected_time_s = time_s[idx].astype(int)
        orbit_times = epoch_date + pd.to_timedelta(selected_time_s, unit="s")
        sim_times = sim_date.normalize() + (orbit_times - orbit_times.normalize())

        for j, i in enumerate(idx):
            observations.append({
                "time": sim_times[j],
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "observer_lat": float(obs_lats[i]),
                "observer_lon": float(obs_lons[i]),
                "observer_alt": float(obs_alts[i]),
            })

    return observations


def l1b_image_to_dataset(l1b, wavelengths_nm, true_extinction: dict | None = None,
                          alt_grid_m=None) -> xr.Dataset:
    """
    Combine an ``L1bImage``'s 'I' and 'dolp' spectra into a single
    ``xr.Dataset`` with dims ``(wavelength, altitude_m)``, suitable for
    ``xr.concat`` across observations along a new ``along_track`` dimension.

    If ``true_extinction`` is given (the second dict returned by
    ``build_waccm_constituents(..., return_extinction=True)``), its
    per-mode multi-wavelength ``{name}_extinction_per_m`` truth-extinction
    fields (shape [wavelength, altitude]) are attached both on their
    native ``atm_altitude_m`` grid (``alt_grid_m``, exact) and interpolated
    onto the instrument's own ``altitude_m`` grid for direct point-by-point
    comparison against radiance/dolp. The dict's other, per-altitude-only
    entries (``{name}_reference_extinction_per_m``, ``{name}_median_radius_nm``,
    ``extinction_wavelength_nm``) aren't wavelength-resolved and are skipped
    here.
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
            if np.asarray(ext_vals).ndim != 2:
                continue
            ds[f"{ext_key}_atm"] = (("wavelength", "atm_altitude_m"), ext_vals)
            interp_vals = np.array([
                np.interp(instrument_alt, alt_grid_m, ext_vals[i, :])
                for i in range(ext_vals.shape[0])
            ])
            ds[ext_key] = (("wavelength", "altitude_m"), interp_vals)
        ds = ds.assign_coords(atm_altitude_m=alt_grid_m)

    return ds
