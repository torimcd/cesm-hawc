"""Read observation geometry from HAWC orbit files.

Also converts the simulator's L1b output to an ``xarray.Dataset``. The
expected orbit file contents are described in the documentation's "Orbit
files" page.
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
    """List the orbit files in a directory.

    Parameters
    ----------
    orbit_dir : str
        Directory to search.
    pattern : str, optional
        Glob pattern. Default ``"orbit_*.nc"``.

    Returns
    -------
    list of str
        Matching paths, sorted.

    Raises
    ------
    FileNotFoundError
        If nothing matches.
    """
    files = sorted(glob.glob(os.path.join(orbit_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No orbit files matching '{pattern}' in {orbit_dir}")
    return files


def orbit_file_start_time(path: str) -> pd.Timestamp:
    """Read an orbit file's start time.

    Parameters
    ----------
    path : str
        Orbit file.

    Returns
    -------
    pandas.Timestamp
        The file's ``start_time`` global attribute.

    Raises
    ------
    KeyError
        If the file has no ``start_time`` attribute.
    """
    ds = xr.open_dataset(path, decode_times=False)
    try:
        return pd.Timestamp(ds.attrs["start_time"])
    finally:
        ds.close()


def _orbit_files_fingerprint(orbit_files: list[str]) -> str:
    """Fingerprint an orbit file set by paths, modification times and sizes.

    Used to tell when a cached day index is out of date.
    """
    parts = [f"{f}:{os.stat(f).st_mtime_ns}:{os.stat(f).st_size}" for f in orbit_files]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def build_orbit_day_index(orbit_files: list[str], epoch: pd.Timestamp,
                           cache_path: str | os.PathLike | None = None
                           ) -> dict[int, list[str]]:
    """Group orbit files by day since the orbit epoch.

    Parameters
    ----------
    orbit_files : list of str
        Orbit files to index.
    epoch : pandas.Timestamp
        Time origin of the orbit file set; day 0 is ``epoch``'s date.
    cache_path : str or os.PathLike, optional
        JSON file to cache the index in. Reading every file's start time can
        take minutes for a large set, so the cached index is reused until a
        file is added, removed or modified.

    Returns
    -------
    dict of int to list of str
        ``{day: [path, ...]}``, where ``day`` is the number of days from
        ``epoch`` to each file's ``start_time``. A day can have several
        files.
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
    """Sample observations from one day's orbit files onto a model date.

    Parameters
    ----------
    orbit_files : list of str
        The orbit files for one orbit day.
    sim_date : pandas.Timestamp
        Model date to place the observations on.
    cadence_s : float
        Spacing between sampled observations [s].
    center_pixel : int
        Across-track pixel used as the tangent point.
    epoch : pandas.Timestamp
        Time origin of the orbit files; their ``time`` values are seconds
        since this date.

    Returns
    -------
    list of dict
        One dict per observation with ``time`` (``pandas.Timestamp`` on
        ``sim_date``), ``lat``, ``lon`` (tangent point [degrees]),
        ``observer_lat``, ``observer_lon`` [degrees] and ``observer_alt``
        [m].

    Notes
    -----
    Each observation keeps its real time of day and satellite geometry;
    only its date changes. The starting offset of the sampling is rotated
    from file to file. Each file is about one orbit of a sun-synchronous
    satellite, so with the same offset every file would sample nearly the
    same latitudes.
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

        # Rotate this file's starting offset instead of always starting at 0
        # (see Notes above).
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
    """Convert simulator L1b output to an ``xarray.Dataset``.

    Parameters
    ----------
    l1b : aliprocessing.l1b.data.L1bImage
        The ``"l1b"`` product from ``simulator.run()``.
    wavelengths_nm : array-like
        The simulated wavelengths [nm].
    true_extinction : dict, optional
        Truth extinction from
        :func:`cesm_hawc.constituents.build_waccm_constituents` with
        ``return_extinction=True``. Its wavelength-resolved
        ``{mode}_extinction_per_m`` entries are added to the dataset.
    alt_grid_m : array-like, optional
        The model altitude grid [m] that ``true_extinction`` is on. Required
        for ``true_extinction`` to be added.

    Returns
    -------
    xarray.Dataset
        ``radiance``, ``radiance_noise``, ``dolp`` and ``dolp_noise`` on
        (``wavelength``, ``altitude_m``), with tangent latitude, longitude
        and solar zenith angle as coordinates along ``altitude_m`` and the
        observation time as the ``time`` attribute. With truth extinction,
        each ``{mode}_extinction_per_m`` is added interpolated onto the
        instrument's ``altitude_m`` grid, and as ``{mode}_extinction_per_m_atm``
        on the model grid (``atm_altitude_m``). Datasets from several
        observations can be joined with ``xarray.concat``.
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
