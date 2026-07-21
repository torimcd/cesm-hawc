#!/usr/bin/env python
"""
run_orbit_daily.py
==================
Simulate HAWCSat ALI observations along the real HAWC orbital ground track,
using daily CESM h2 output files. Each day's observations use that day's
atmospheric snapshot. Solar geometry is computed automatically from the
observation time and tangent-point position via sasktran2's
SolarGeometryHandlerAstropy.

Orbit files are mapped to simulation dates by day-of-year offset:
  orbit day 1 (2019-08-01) -> simulation day 1 (e.g. 2030-01-01)
  orbit day 2 (2019-08-02) -> simulation day 2
  ... repeating the 3-month orbit pattern as needed.

Observations are subsampled to one per OBS_CADENCE_S seconds (~12 min)
using the center cross-track pixel (index 256) as the tangent point.

Runs forward-only simulation (no l2 retrieval) for speed.
Multiple injection cases can be run in the same pass.

Edit config.toml at the project root, then run:
    python scripts/run_orbit_daily.py
or submit via SLURM (parallel across days):
    sbatch slurm/submit_orbit_daily.sh

Output layout
-------------
OUT_DIR/
  background/
    YYYY-MM-DD/
      curtain.nc          # dims [along_track, altitude_m]; coords lat, lon, time
      orbit_track.csv     # time, lat, lon per observation
  sai_1.0Tg/
    YYYY-MM-DD/
      curtain.nc
      orbit_track.csv
  sai_0.1Tg/
    ...
  sai_0.01Tg/
    ...
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        sys.exit("Python < 3.11 requires tomli: pip install tomli")

import numpy as np
import pandas as pd
import xarray as xr

# Suppress non-fatal "unsendable, dropped on another thread" RuntimeErrors.
# These come from sasktran2's Rust core objects being garbage-collected on a
# different thread than they were created on, inside Dask's threaded
# scheduler. They're reported via sys.unraisablehook (raised during __del__ /
# GC, not through the normal call stack), so they cannot be caught with
# try/except. They don't affect the correctness of completed results — just
# terminal noise — so filter just these out while still surfacing anything
# genuinely unexpected.
import sys as _sys

_original_unraisablehook = _sys.unraisablehook


def _filtered_unraisablehook(unraisable):
    msg = str(unraisable.exc_value) if unraisable.exc_value else ""
    if "unsendable" in msg and "_core_rust" in msg:
        return  # known non-fatal noise, silently ignore
    _original_unraisablehook(unraisable)


_sys.unraisablehook = _filtered_unraisablehook

# Prevent astropy from downloading IERS Earth-orientation data on every run
# (or every worker process). This data is used internally for precise solar
# geometry, but SZA calculations don't need arcsecond-level Earth-orientation
# precision, and compute nodes typically don't have internet access anyway.
# Must be set before any code that triggers astropy's solar position calcs.
#
# auto_max_age is also disabled: our simulation dates (2030) are years beyond
# any real IERS predictive data (which only covers the recent past + a short
# forward window from the actual current date). UT1-UTC drift is
# millisecond-scale and irrelevant to SZA/day-night determination at these
# timescales, so extrapolating past the data window is safe here.
from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.auto_max_age = None

from cesm_hawc.waccm import WACCMAtmosphere
from cesm_hawc.constituents import build_waccm_constituents
from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator
from hawcsimulator.noise import ALINoiseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG = Path(__file__).parent.parent / "config.toml"
if not _CONFIG.exists():
    sys.exit(
        f"config.toml not found at {_CONFIG}\n"
        "Copy config.example.toml -> config.toml and fill in your paths."
    )

with open(_CONFIG, "rb") as _f:
    _cfg = tomllib.load(_f)

_o   = _cfg["orbit_daily"]
_ins = _cfg["instrument"]

# Orbit files
ORBIT_DIR       = os.path.expanduser(_o["orbit_dir"])
ORBIT_PATTERN   = _o.get("orbit_pattern", "orbit_*.nc")
ORBIT_EPOCH     = pd.Timestamp(_o.get("orbit_epoch", "2019-08-01"))
CENTER_PIXEL    = int(_o.get("center_pixel", 256))
OBS_CADENCE_S   = int(_o.get("obs_cadence_s", 720))

# CESM data
WACCM_DATA_DIR  = os.path.expanduser(_o["waccm_data_dir"])
H2_PATTERN      = _o.get("h2_pattern", "*.cam.h2.*.nc")
SIM_START       = pd.Timestamp(_o["sim_start"])   # e.g. "2030-01-01"

# Cases: background + injection scenarios
BACKGROUND_CASE  = _o["background_case"]   # e.g. "sai_background_2030_001"
INJECTION_CASES  = _o.get("injection_cases", [])
# e.g. ["sai_1.0Tg_2030_001", "sai_0.1Tg_2030_001", "sai_0.01Tg_2030_001"]

# Output
OUT_DIR   = os.path.expanduser(_o["out_dir"])
N_WORKERS = int(_o.get("n_workers", 1))

# Instrument
ALI_WAVELENGTHS = np.array(_ins["wavelengths_nm"])
ALT_GRID_M = np.arange(
    _ins["alt_grid_start_m"],
    _ins["alt_grid_stop_m"] + _ins["alt_grid_step_m"],
    _ins["alt_grid_step_m"],
)
_noise_frac = _ins.get("noise_straylight_fraction", "")
NOISE_MODEL: Optional[ALINoiseModel] = (
    ALINoiseModel(straylight_fraction=float(_noise_frac))
    if _noise_frac != "" else None
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress Hamilton's per-node error boxes (SLACK_ERROR_MESSAGE + "Node
# inputs" dump). These print via logger.error() on the "hamilton.*" logger
# namespace for EVERY node exception, including ones we deliberately catch
# and handle (e.g. the SZA night-side skip in process_day, which fires ~78
# times per day). We still get full tracebacks for genuine day failures via
# our own log.error() calls below, so this is safe to silence.
logging.getLogger("hamilton").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Calibration database guard
# ---------------------------------------------------------------------------
# hawcsimulator's calibration_database() unconditionally rewrites its cached
# .nc file (clobber=True) every time it's called, including internally
# whenever an IdealALISimulator is constructed. Under many workers/days
# hitting the same NFS-mounted cache file concurrently, this produces
# PermissionError / KeyError races (multiple processes racing to open the
# same file for writing).
#
# We patch it here so it's idempotent: if the cache file already exists on
# disk, skip the rewrite and trust it. This is safe because the cache
# content only depends on the (name, version) pair, which is fixed per run.

from hawcsimulator.ali import calibration as _cal_mod

_orig_calibration_database = _cal_mod.calibration_database


def _cache_file_path(name: str, version: str) -> str:
    # Matches the path seen in tracebacks:
    # ~/.local/share/hawc-simulator/ali/calibration/{name}_{version}.nc
    cache_dir = os.path.expanduser("~/.local/share/hawc-simulator/ali/calibration")
    return os.path.join(cache_dir, f"{name}_{version}.nc")


def _safe_calibration_database(name: str, version: str):
    cache_file = _cache_file_path(name, version)
    if os.path.exists(cache_file):
        return cache_file
    return _orig_calibration_database(name, version)


# Patch the module attribute (covers any code that does
# hawcsimulator.ali.calibration.calibration_database(...) directly)...
_cal_mod.calibration_database = _safe_calibration_database

# ...but IdealALISimulator's _initialize_data() does NOT use that path. It
# did `from hawcsimulator.ali.calibration import calibration_database` at
# ITS OWN import time, which binds a separate name inside
# ideal_spectrograph's own module namespace, pointing at the ORIGINAL
# function. Patching calibration.calibration_database afterward has no
# effect on that already-bound reference. We must patch the name where
# _initialize_data() actually looks it up: inside ideal_spectrograph's own
# globals. This is the binding that matters for every simulator.run() call.
from hawcsimulator.ali.configurations import ideal_spectrograph as _ideal_spectrograph_mod

_ideal_spectrograph_mod.calibration_database = _safe_calibration_database

# ---------------------------------------------------------------------------
# Orbit file utilities
# ---------------------------------------------------------------------------

def load_orbit_files() -> list[str]:
    """Return sorted list of orbit file paths."""
    files = sorted(glob.glob(os.path.join(ORBIT_DIR, ORBIT_PATTERN)))
    if not files:
        raise FileNotFoundError(
            f"No orbit files matching '{ORBIT_PATTERN}' in {ORBIT_DIR}"
        )
    log.info("Found %d orbit files in %s", len(files), ORBIT_DIR)
    return files


def orbit_file_start_time(path: str) -> pd.Timestamp:
    """Get start time of an orbit file from its global attribute."""
    ds = xr.open_dataset(path)
    t = pd.Timestamp(ds.attrs["start_time"])
    ds.close()
    return t


_ORBIT_CACHE_DIR = Path(__file__).parent.parent / ".cache"
_ORBIT_CACHE_FILE = _ORBIT_CACHE_DIR / "orbit_day_index.json"


def _orbit_files_fingerprint(orbit_files: list[str]) -> str:
    """
    Cheap fingerprint of the orbit file set (paths + mtimes + sizes) used to
    detect when the cached day-index is stale and needs rebuilding. This is
    fast (just filesystem stat calls) even though the actual index build is
    slow (opens each file to read its start_time attribute).
    """
    parts = []
    for f in orbit_files:
        st = os.stat(f)
        parts.append(f"{f}:{st.st_mtime_ns}:{st.st_size}")
    fingerprint_str = "\n".join(parts)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()


def build_orbit_day_index(orbit_files: list[str]) -> dict[int, list[str]]:
    """
    Map orbit-calendar day-of-sequence (0-indexed from ORBIT_EPOCH) to
    list of orbit file paths covering that day.

    Cached to disk: reading each orbit file's start_time attribute (1415
    files) takes several minutes via xr.open_dataset, so the resulting
    mapping is cached and only rebuilt if the orbit file set changes
    (detected via a fingerprint of paths/mtimes/sizes, checked via cheap
    os.stat calls rather than reopening every file).

    Returns dict: {orbit_day_index: [file, file, ...]}
    """
    fingerprint = _orbit_files_fingerprint(orbit_files)

    if _ORBIT_CACHE_FILE.exists():
        try:
            with open(_ORBIT_CACHE_FILE) as f:
                cached = json.load(f)
            if cached.get("fingerprint") == fingerprint:
                day_index = {int(k): v for k, v in cached["day_index"].items()}
                log.info("Using cached orbit day index (%d files, %d days)",
                          len(orbit_files), len(day_index))
                return day_index
            else:
                log.info("Orbit file set changed, rebuilding day index...")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning("Orbit day index cache unreadable, rebuilding: %s", e)

    log.info("Building orbit day index from %d files (reads start_time from "
              "each file, this can take several minutes on first run)...",
              len(orbit_files))
    day_index: dict[int, list[str]] = {}
    for f in orbit_files:
        t = orbit_file_start_time(f)
        day = (t.normalize() - ORBIT_EPOCH.normalize()).days
        day_index.setdefault(day, []).append(f)

    try:
        _ORBIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_ORBIT_CACHE_FILE, "w") as f:
            json.dump({"fingerprint": fingerprint, "day_index": day_index}, f)
        log.info("Cached orbit day index to %s", _ORBIT_CACHE_FILE)
    except OSError as e:
        log.warning("Could not write orbit day index cache: %s", e)

    return day_index


def extract_observations(
    orbit_files: list[str],
    sim_date: pd.Timestamp,
    cadence_s: int,
    center_pixel: int,
) -> list[dict]:
    """
    Extract subsampled observations from a list of orbit files covering one day.

    For each orbit file:
      - Read time, lat, lon at center_pixel
      - Subsample to every cadence_s seconds
      - Replace orbit epoch date with sim_date, keeping time-of-day

    Returns list of dicts: {time: pd.Timestamp, lat: float, lon: float}
    """
    observations = []
    orbit_epoch_date = ORBIT_EPOCH.normalize()

    for f in sorted(orbit_files):
        ds = xr.open_dataset(f)

        # time in seconds since orbit epoch
        time_s = ds["time"].values  # int64 seconds since 2019-08-01
        lats   = ds["latitude"].values[:, 0, center_pixel]
        lons   = ds["longitude"].values[:, 0, center_pixel]
        # satellite position (not the tangent point) — required by
        # hawcsimulator's time-based solar geometry handler
        obs_lats = ds["observer_latitude"].values
        obs_lons = ds["observer_longitude"].values
        obs_alts = ds["observer_altitude"].values
        ds.close()

        # convert orbit times to timestamps
        orbit_times = [
            orbit_epoch_date + pd.Timedelta(seconds=int(t))
            for t in time_s
        ]

        # subsample at cadence_s intervals
        prev_idx = -cadence_s  # ensure first point is always included
        for i, (t_orbit, lat, lon) in enumerate(zip(orbit_times, lats, lons)):
            if i - prev_idx < cadence_s:
                continue
            # replace date with simulation date, keep time-of-day
            time_of_day = t_orbit - t_orbit.normalize()
            sim_time = sim_date.normalize() + time_of_day

            observations.append({
                "time": sim_time,
                "lat":  float(lat),
                "lon":  float(lon),
                "observer_lat": float(obs_lats[i]),
                "observer_lon": float(obs_lons[i]),
                "observer_alt": float(obs_alts[i]),
            })
            prev_idx = i

    return observations


# ---------------------------------------------------------------------------
# CESM file utilities
# ---------------------------------------------------------------------------

def build_h2_index(case: str) -> dict[str, str]:
    """
    Return {YYYY-MM-DD: filepath} for all h2 files for a given case.
    """
    case_dir = os.path.join(WACCM_DATA_DIR, case, "atm", "hist")
    pattern  = os.path.join(case_dir, H2_PATTERN)
    files    = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No h2 files matching '{H2_PATTERN}' in {case_dir}"
        )
    index = {}
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})-\d+\.nc$", os.path.basename(f))
        if m:
            index[m.group(1)] = f
    log.info("  %s: %d h2 files", case, len(index))
    return index


# ---------------------------------------------------------------------------
# L1bImage -> xr.Dataset conversion
# ---------------------------------------------------------------------------
# simulator.run(...)["l1b"] returns an aliprocessing.l1b.data.L1bImage, not a
# plain xarray object. Its `.spectra` attribute is a dict keyed by
# polarization state ("I", "dolp"), each value an L1bSpectra wrapping its own
# xr.Dataset with dims (wavelength, los). "los" here is the line-of-sight /
# altitude dimension and matches ALT_GRID_M; "I" and "dolp" share identical
# geometry (tangent_altitude/lat/lon, time, solar_zenith_angle, etc.) so we
# only need to pull that metadata from one of them.

def _l1b_image_to_dataset(l1b) -> xr.Dataset:
    """
    Combine an L1bImage's 'I' and 'dolp' spectra into a single xr.Dataset
    with dims (wavelength, altitude_m), suitable for xr.concat across
    observations along a new "along_track" dimension.
    """
    I_ds    = l1b.spectra["I"].ds
    dolp_ds = l1b.spectra["dolp"].ds

    combined = xr.Dataset(
        data_vars={
            "radiance":       (("wavelength", "altitude_m"), I_ds["radiance"].values),
            "radiance_noise": (("wavelength", "altitude_m"), I_ds["radiance_noise"].values),
            "dolp":           (("wavelength", "altitude_m"), dolp_ds["radiance"].values),
            "dolp_noise":     (("wavelength", "altitude_m"), dolp_ds["radiance_noise"].values),
        },
        coords={
            "wavelength":          ALI_WAVELENGTHS,
            "altitude_m":          I_ds["tangent_altitude"].values,
            "tangent_latitude":    ("altitude_m", I_ds["tangent_latitude"].values),
            "tangent_longitude":   ("altitude_m", I_ds["tangent_longitude"].values),
            "solar_zenith_angle":  ("altitude_m", I_ds["solar_zenith_angle"].values),
        },
    )
    combined.attrs["time"] = str(I_ds["time"].values)
    return combined


# ---------------------------------------------------------------------------
# Per-worker simulator (one instance reused across all days a worker handles)
# ---------------------------------------------------------------------------

_SIMULATOR: Optional[IdealALISimulator] = None


def _get_simulator() -> IdealALISimulator:
    """
    Lazily construct IdealALISimulator once per worker process and reuse it
    across all process_day() calls dispatched to that worker. Avoids
    redundant calibration-database access (and construction overhead) on
    every single day.
    """
    global _SIMULATOR
    if _SIMULATOR is None:
        _SIMULATOR = IdealALISimulator()
    return _SIMULATOR


# ---------------------------------------------------------------------------
# Per-day simulation
# ---------------------------------------------------------------------------

def process_day(
    sim_date_str: str,
    observations: list[dict],
    h2_files: dict[str, str],   # {case_label: filepath}
    out_root: str,
) -> str:
    """
    Run forward ALI simulations for all observations on one day, for all cases.

    h2_files maps a short label (e.g. "background", "sai_1.0Tg") to the
    h2 file path for that case on this date.

    Returns short status string.
    """
    try:
        simulator = _get_simulator()
        waccm_cache: dict[str, WACCMAtmosphere] = {}

        def get_waccm(path: str) -> WACCMAtmosphere:
            if path not in waccm_cache:
                waccm_cache[path] = WACCMAtmosphere(
                    path, alt_grid_km=ALT_GRID_M / 1e3
                )
            return waccm_cache[path]

        # results[label] = list of l1b datasets
        results: dict[str, list[xr.Dataset]] = {label: [] for label in h2_files}
        successful_obs: list[dict] = []  # obs that passed the daytime/SZA check
        n_skipped_night = 0

        for obs in observations:
            t   = obs["time"]
            lat = obs["lat"]
            lon = obs["lon"]

            sim_input = {
                "tangent_latitude":    float(lat),
                "tangent_longitude":   float(lon),
                "observer_latitude":   obs["observer_lat"],
                "observer_longitude":  obs["observer_lon"],
                "observer_altitude":   obs["observer_alt"],
                "altitude_grid":       ALT_GRID_M,
                "polarization_states": ["I", "dolp"],
                "sample_wavelengths":  ALI_WAVELENGTHS,
                "time":                t,
                # SZA/SAA omitted -> computed automatically by astropy
                # from time + observer position
            }
            if NOISE_MODEL is not None:
                sim_input["l1b_cfg"] = {"noise_model": NOISE_MODEL}

            # SZA depends only on time + geometry, not on which case/h2 file
            # is used, so if one case is night-side, all cases are — check
            # once per observation and skip the whole point if so, rather
            # than letting it crash the whole day.
            obs_l1b: dict[str, xr.Dataset] = {}
            skip_obs = False
            for label, h2_path in h2_files.items():
                waccm  = get_waccm(h2_path)
                profiles = waccm.get_column_profiles(lat, lon, time_index=0)
                try:
                    data = simulator.run(
                        ["front_end_radiance", "l1b"],   # forward only — no l2
                        {**sim_input,
                         "constituents": build_waccm_constituents(profiles, ALT_GRID_M)},
                    )
                except ValueError as e:
                    if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                        # night-side tangent point — physically unobservable,
                        # not an error condition. Skip this observation.
                        skip_obs = True
                        break
                    raise
                obs_l1b[label] = _l1b_image_to_dataset(data["l1b"])

            if skip_obs:
                n_skipped_night += 1
                continue

            for label, l1b in obs_l1b.items():
                results[label].append(l1b)
            successful_obs.append(obs)

        if n_skipped_night:
            log.info(
                "[%s] skipped %d/%d observations (night-side, SZA too large)",
                sim_date_str, n_skipped_night, len(observations),
            )

        # save curtain per case — coordinates must come from successful_obs
        # only, so they stay aligned with the along_track dim of the data
        lats  = [o["lat"]  for o in successful_obs]
        lons  = [o["lon"]  for o in successful_obs]
        times = [o["time"] for o in successful_obs]

        for label, l1b_list in results.items():
            if not l1b_list:
                continue
            case_out = os.path.join(out_root, label, sim_date_str)
            os.makedirs(case_out, exist_ok=True)

            curtain = xr.concat(l1b_list, dim="along_track")
            curtain = curtain.assign_coords(
                lat=("along_track", lats),
                lon=("along_track", lons),
                time=("along_track", times),
            )
            curtain.to_netcdf(os.path.join(case_out, "curtain.nc"))

        # save orbit track csv (same for all cases) — daytime obs only
        track_df = pd.DataFrame({
            "time": [o["time"].isoformat() for o in successful_obs],
            "lat":  lats,
            "lon":  lons,
        })
        # write to background dir (canonical reference)
        bg_out = os.path.join(out_root, "background", sim_date_str)
        os.makedirs(bg_out, exist_ok=True)
        track_df.to_csv(os.path.join(bg_out, "orbit_track.csv"), index=False)

        log.info("[%s] %d/%d obs (daytime), %d cases",
                  sim_date_str, len(successful_obs), len(observations), len(h2_files))
        return f"OK   {sim_date_str}  ({len(successful_obs)}/{len(observations)} obs)"

    except Exception:
        tb = traceback.format_exc()
        log.error("[%s] FAILED:\n%s", sim_date_str, tb)
        return f"FAIL {sim_date_str}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # load orbit files and build day index
    log.info("Loading orbit file index...")
    orbit_files   = load_orbit_files()
    orbit_day_idx = build_orbit_day_index(orbit_files)
    n_orbit_days  = max(orbit_day_idx.keys()) + 1
    log.info("Orbit pattern spans %d days", n_orbit_days)

    # build h2 file indices for all cases
    log.info("Building h2 file indices...")
    all_cases = {
        "background": BACKGROUND_CASE,
        **{c.split("_")[1]: c for c in INJECTION_CASES}
        # e.g. "1.0Tg" -> "sai_1.0Tg_2030_001"
        # override labels below if this auto-naming doesn't suit
    }
    # use cleaner labels
    case_labels = {"background": BACKGROUND_CASE}
    for c in INJECTION_CASES:
        # extract rate from name e.g. sai_1.0Tg_2030_001 -> sai_1.0Tg
        m = re.match(r"(sai_[\d.]+Tg)", c)
        label = m.group(1) if m else c
        case_labels[label] = c

    h2_indices: dict[str, dict[str, str]] = {}
    for label, case in case_labels.items():
        h2_indices[label] = build_h2_index(case)

    # determine simulation dates from background h2 files
    bg_dates = sorted(h2_indices["background"].keys())
    log.info("Simulation covers %d days: %s to %s",
             len(bg_dates), bg_dates[0], bg_dates[-1])

    # build job list
    jobs = []
    for i, date_str in enumerate(bg_dates):
        # map simulation day to orbit day (repeating pattern)
        orbit_day = i % n_orbit_days
        if orbit_day not in orbit_day_idx:
            log.warning("No orbit files for orbit day %d, skipping %s",
                        orbit_day, date_str)
            continue

        sim_date = pd.Timestamp(date_str)
        obs = extract_observations(
            orbit_day_idx[orbit_day], sim_date, OBS_CADENCE_S, CENTER_PIXEL
        )
        if not obs:
            log.warning("No observations for %s, skipping", date_str)
            continue

        # collect h2 file paths for this date across all cases
        h2_for_day: dict[str, str] = {}
        for label in case_labels:
            if date_str in h2_indices[label]:
                h2_for_day[label] = h2_indices[label][date_str]
            else:
                log.warning("Missing h2 file for case '%s' on %s", label, date_str)

        if not h2_for_day:
            continue

        jobs.append((date_str, obs, h2_for_day, OUT_DIR))

    log.info("Processing %d days with %d cases each",
             len(jobs), len(case_labels))

    # pre-warm calibration database (idempotent — safe to call again
    # even if a worker also triggers it via _get_simulator())
    log.info("Pre-warming calibration database...")
    try:
        _cal_mod.calibration_database("ideal_spectrograph", "v1")
        log.info("Calibration database ready.")
    except Exception as e:
        log.warning("Could not pre-warm calibration database: %s", e)

    # pre-warm mode-specific Mie databases (accumulation/coarse extinction).
    # First build triggers a real Mie calculation per mode_width; doing
    # this once, serially, before dispatching workers avoids any
    # unknown-safety concurrent-build behavior in sasktran2's MieDatabase.
    log.info("Pre-warming mode-specific Mie databases...")
    try:
        from cesm_hawc.constituents import warm_mode_databases
        warm_mode_databases()
        log.info("Mie databases ready.")
    except Exception as e:
        log.warning("Could not pre-warm Mie databases: %s", e)
    
    # dispatch
    results: list[str] = []
    if N_WORKERS <= 1:
        for job in jobs:
            results.append(process_day(*job))
    else:
        workers = min(N_WORKERS, len(jobs))
        log.info("Dispatching to %d worker processes...", workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_day, *job): job[0] for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                log.info("Completed: %s", result)

    # final report
    ok   = [r for r in results if r.startswith("OK")]
    fail = [r for r in results if r.startswith("FAIL")]
    log.info("\n-- Run complete --")
    log.info("  Succeeded: %d / %d", len(ok), len(results))
    if fail:
        log.warning("  Failed days:")
        for f in fail:
            log.warning("    %s", f)
    log.info("  Output root: %s", OUT_DIR)


if __name__ == "__main__":
    main()