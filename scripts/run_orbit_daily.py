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

By default runs forward-only simulation (no L2 retrieval) for speed.
Set run_l2 = true in config.toml to additionally run full L2 retrieval
per observation (see the [orbit_daily] section below for cost caveats --
this is the same code path validated in benchmark_l2_retrieval.py, which
measured 100-600s per profile for the retrieval step). Run
extrapolate_to_full_run() from that script against your real benchmark
CSV before submitting a large L2 production job.

Multiple injection cases can be run in the same pass.

Edit config.toml at the project root, then run:
    python scripts/run_orbit_daily.py
or submit via SLURM (parallel across days):
    sbatch slurm/submit_orbit_daily.sh

RESUMABILITY (L2 mode): a full day of L2 retrieval can take on the order
of 15 hours (measured ~60 daytime obs/day x 4 cases x 100-600s/profile),
and the full 6-month run will almost certainly need multiple sequential
sbatch submissions to fit within walltime caps. Two layers of resume
protection:
  - Day-level: main() skips building a job for a date whose expected
    outputs already exist, so re-submitting doesn't reprocess completed
    days from scratch.
  - Profile-level (within a day): l2_diagnostics.csv is written
    incrementally (one row per profile, flushed immediately) rather than
    only at day-end, and each L2 profile's retrieval output is saved to
    its own small file immediately after that profile completes. A day
    that gets killed partway through resumes from where it left off
    rather than redoing already-completed profiles, and a single
    profile's unexpected failure is logged and skipped rather than
    discarding the rest of the day's already-completed work.

Output layout
-------------
OUT_DIR/
  background/
    YYYY-MM-DD/
      curtain.nc            # dims [along_track, altitude_m]; coords lat, lon, time
      orbit_track.csv       # time, lat, lon per observation
      l2_retrieval.nc       # only if run_l2=true; dims [along_track, ...]
      l2_diagnostics.csv    # only if run_l2=true; convergence/timing per obs, all cases
      l2_profiles/          # only if run_l2=true; one small .nc per (case,time)
        <case_label>_<time>.nc
  sai_1.0Tg/
    YYYY-MM-DD/
      curtain.nc
      orbit_track.csv
      l2_retrieval.nc        # only if run_l2=true
  sai_0.1Tg/
    ...
  sai_0.01Tg/
    ...
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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

# Optional restriction to a subset of the available date range, e.g. to run
# a job in stages (first N months now, remainder later). If unset, all
# available dates are processed. Dates are "YYYY-MM-DD" strings, compared
# lexicographically (which works correctly for ISO date format).
RUN_START_DATE = _o.get("run_start_date")  # e.g. "2030-01-01", or None
RUN_END_DATE   = _o.get("run_end_date")    # e.g. "2030-05-31", or None

# Full L2 retrieval toggle. Off by default -- forward-only is the fast path
# and is what production has run to date. When true, every observation
# additionally runs skretrieval's optimal-estimation retrieval (100-600s/
# profile measured in benchmark_l2_retrieval.py), producing data["l2"] in
# addition to the existing front_end_radiance/l1b forward output. Product
# list matches FULL_L2_PRODUCTS in benchmark_l2_retrieval.py exactly --
# that's the call signature that was actually confirmed to work and profiled;
# don't change the product list without re-validating against that script.
RUN_L2 = bool(_o.get("run_l2", False))
L2_PRODUCTS = ["l2", "sk2_atmosphere", "front_end_radiance", "l1b"]
FORWARD_ONLY_PRODUCTS = ["front_end_radiance", "l1b"]

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
# L2 convergence diagnostics
# ---------------------------------------------------------------------------
# Ported directly from benchmark_l2_retrieval.py, where these were validated
# against real output -- confirmed that hawcsimulator/skretrieval's l2
# dataset .attrs carry no diagnostics at all, and that scipy.optimize.
# least_squares' verbose=2 stdout output is the reliable source for
# converged/not-converged status (a real background-case non-convergence
# was caught this way; the old attrs-based approach silently reported it
# as unknown). num_iterations/cost come straight off the l2 Dataset itself
# as a cross-check, per the same script.

_SCIPY_CONVERGED_PATTERNS = [
    (re.compile(r"`ftol` termination condition is satisfied"), "ftol"),
    (re.compile(r"`xtol` termination condition is satisfied"), "xtol"),
    (re.compile(r"`gtol` termination condition is satisfied"), "gtol"),
]
_SCIPY_NOT_CONVERGED_PATTERNS = [
    (re.compile(r"maximum number of function evaluations is exceeded", re.IGNORECASE), "max_nfev"),
    (re.compile(r"maximum number of iterations is exceeded", re.IGNORECASE), "max_iter"),
]
_SCIPY_NFEV_PATTERN = re.compile(r"Function evaluations (\d+)")


def _parse_scipy_convergence(captured_stdout: str) -> dict:
    """Parse scipy.optimize.least_squares' verbose=2 output for the real
    convergence status and function-evaluation count. Returns
    {converged, termination_reason, n_function_evaluations}, with None
    values if no recognized message was found."""
    result = {"converged": None, "termination_reason": None, "n_function_evaluations": None}

    for pattern, reason in _SCIPY_CONVERGED_PATTERNS:
        if pattern.search(captured_stdout):
            result["converged"] = True
            result["termination_reason"] = reason
            break
    else:
        for pattern, reason in _SCIPY_NOT_CONVERGED_PATTERNS:
            if pattern.search(captured_stdout):
                result["converged"] = False
                result["termination_reason"] = reason
                break

    m = _SCIPY_NFEV_PATTERN.search(captured_stdout)
    if m:
        result["n_function_evaluations"] = int(m.group(1))

    return result


def _extract_l2_native_diagnostics(l2_obj) -> dict:
    """Pull num_iterations/cost directly from the l2 Dataset, as a
    cross-check against the stdout-parsed convergence info."""
    if l2_obj is None:
        return {"l2_num_iterations": None, "l2_final_cost": None}
    try:
        n_iter = int(l2_obj["num_iterations"].values) if "num_iterations" in l2_obj else None
    except Exception:
        n_iter = None
    try:
        cost = float(l2_obj["cost"].values) if "cost" in l2_obj else None
    except Exception:
        cost = None
    return {"l2_num_iterations": n_iter, "l2_final_cost": cost}


# ---------------------------------------------------------------------------
# L2 per-day resumability: incremental diagnostics CSV + per-profile saves
# ---------------------------------------------------------------------------
# A full day of L2 retrieval can take ~15 hours (measured ~60 daytime obs/day
# x 4 cases x 100-600s/profile in benchmark_l2_retrieval.py). Without this,
# a walltime kill or one bad profile partway through a day would discard
# everything computed that day, forward output included. This mirrors the
# resume machinery already validated in benchmark_l2_retrieval.py.

L2_DIAG_FIELDNAMES = [
    "case_label", "time", "lat", "lon", "elapsed_s",
    "converged", "termination_reason", "n_function_evaluations",
    "l2_num_iterations", "l2_final_cost", "status", "error",
]


def _safe_time_str(t) -> str:
    """Filesystem-safe representation of an observation time, used as part
    of per-profile filenames and as the resume-matching key alongside
    case_label."""
    return str(pd.Timestamp(t)).replace(" ", "T").replace(":", "")


def _l2_diag_csv_path(out_root: str, sim_date_str: str) -> str:
    bg_out = os.path.join(out_root, "background", sim_date_str)
    return os.path.join(bg_out, "l2_diagnostics.csv")


def _l2_profile_path(out_root: str, case_label: str, sim_date_str: str, obs_time) -> str:
    case_out = os.path.join(out_root, case_label, sim_date_str, "l2_profiles")
    return os.path.join(case_out, f"{case_label}_{_safe_time_str(obs_time)}.nc")


def _load_completed_l2_keys(csv_path: str) -> set[tuple[str, str]]:
    """Return {(case_label, time_str)} already present in an existing
    l2_diagnostics.csv, so a resumed day skips re-running those profiles'
    L2 retrieval. If the file's schema doesn't match L2_DIAG_FIELDNAMES
    (e.g. left over from an earlier version of this script), back it up
    and start fresh rather than risk corrupting it with inconsistent
    columns -- same protection added to benchmark_l2_retrieval.py after a
    real schema-drift corruption there."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()

    with open(csv_path) as f:
        existing_header = f.readline().strip().split(",")
    if existing_header != L2_DIAG_FIELDNAMES:
        backup_path = csv_path + ".schema_mismatch.bak"
        log.warning(
            "%s has a different column schema than the current script "
            "(existing: %s | current: %s). Backing up to %s and starting "
            "fresh rather than risking corruption.",
            csv_path, existing_header, L2_DIAG_FIELDNAMES, backup_path,
        )
        os.rename(csv_path, backup_path)
        return set()

    try:
        existing = pd.read_csv(csv_path, usecols=["case_label", "time"])
    except Exception as e:
        log.warning("Could not read %s for resume (%s); treating as no prior progress.",
                    csv_path, e)
        return set()
    return {(str(r.case_label), str(pd.Timestamp(r.time))) for r in existing.itertuples(index=False)}


def _append_l2_diag_row(csv_path: str, row: dict) -> None:
    """Append one L2 diagnostics row immediately, flushed to disk, rather
    than accumulating in memory for a single end-of-day write."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header_needed = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    row_df = pd.DataFrame([row], columns=L2_DIAG_FIELDNAMES)
    row_df.to_csv(csv_path, mode="a", index=False, header=header_needed)


def _save_l2_profile(l2_obj, out_path: str) -> tuple[bool, str | None]:
    """Save one profile's L2 output immediately after retrieval, so a
    day-level kill preserves already-completed L2 profiles instead of
    losing them when the end-of-day xr.concat step never runs. Returns
    (saved, error)."""
    if l2_obj is None:
        return False, "l2 object is None"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        l2_obj.to_netcdf(out_path)
        return True, None
    except Exception as e:
        return False, f"to_netcdf failed: {type(e).__name__}: {e}"


def _load_saved_l2_profile(path: str) -> xr.Dataset | None:
    try:
        return xr.open_dataset(path).load()
    except Exception as e:
        log.warning("Could not reload saved L2 profile %s (%s) -- it will be "
                    "excluded from this day's l2_retrieval.nc curtain.", path, e)
        return None


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
# Day-level resume (main() skips days already fully completed)
# ---------------------------------------------------------------------------

def _day_already_done(date_str: str, case_labels: dict[str, str], out_root: str) -> bool:
    """
    True if this date's expected outputs already exist for every case, so
    main() can skip re-submitting it. Coarse-grained (day-level) -- a day
    that's only partially done (e.g. killed mid-L2) is NOT considered done
    here and will be resubmitted; process_day()'s profile-level resume
    (via l2_diagnostics.csv / saved l2_profiles) then picks up only the
    remaining work for that day rather than redoing it all.
    """
    for label in case_labels:
        case_dir = os.path.join(out_root, label, date_str)
        if not os.path.exists(os.path.join(case_dir, "curtain.nc")):
            return False
        if RUN_L2 and not os.path.exists(os.path.join(case_dir, "l2_retrieval.nc")):
            return False
    return True


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
    Run ALI simulations for all observations on one day, for all cases.
    Forward-only (front_end_radiance/l1b) by default; additionally runs
    full L2 retrieval per observation when RUN_L2 is true.

    h2_files maps a short label (e.g. "background", "sai_1.0Tg") to the
    h2 file path for that case on this date.

    L2 mode is resumable within a day: already-completed (case_label,
    time) profiles (per l2_diagnostics.csv) are skipped, and their saved
    per-profile .nc files are reloaded for inclusion in this day's final
    l2_retrieval.nc curtain. A single profile's unexpected failure is
    logged and recorded in l2_diagnostics.csv with status="error" rather
    than aborting the rest of the day.

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

        products = L2_PRODUCTS if RUN_L2 else FORWARD_ONLY_PRODUCTS

        # results[label] = list of l1b datasets (forward output, unchanged)
        results: dict[str, list[xr.Dataset]] = {label: [] for label in h2_files}
        # l2_results[label] = list of raw l2 xr.Dataset objects, in the same
        # order as successful_obs, so the end-of-day concat lines up with
        # the forward curtain's along_track coordinates. Populated either
        # from a fresh retrieval this run or reloaded from a previous run's
        # saved l2_profiles/ file (profile-level resume).
        l2_results: dict[str, list[xr.Dataset]] = {label: [] for label in h2_files}

        l2_diag_csv_path = _l2_diag_csv_path(out_root, sim_date_str)
        completed_l2_keys: set[tuple[str, str]] = set()
        if RUN_L2:
            completed_l2_keys = _load_completed_l2_keys(l2_diag_csv_path)
            if completed_l2_keys:
                log.info("[%s] Resuming L2: %d (case, time) profiles already done",
                          sim_date_str, len(completed_l2_keys))

        successful_obs: list[dict] = []  # obs that passed the daytime/SZA check
        n_skipped_night = 0

        n_l2_done_this_run = 0
        l2_day_t0 = time.perf_counter()

        for obs in observations:
            t   = obs["time"]
            lat = obs["lat"]
            lon = obs["lon"]
            time_key = str(pd.Timestamp(t))

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
                # capture the true per-wavelength extinction profiles
                # alongside the constituents dict — computed inside
                # build_waccm_constituents() using the same mode-matched
                # Mie databases that drive the forward model, requested at
                # the same wavelengths (ALI_WAVELENGTHS) the simulator
                # itself runs at, so radiance and truth extinction can be
                # directly compared per wavelength in Phase 3.
                constituents, true_ext = build_waccm_constituents(
                    profiles, ALT_GRID_M, return_extinction=True,
                    truth_wavelengths_nm=ALI_WAVELENGTHS,
                )

                already_done = RUN_L2 and (label, time_key) in completed_l2_keys

                l2_stdout = io.StringIO()
                obs_t0 = time.perf_counter()
                try:
                    if already_done:
                        # Skip the expensive L2 retrieval entirely, but we
                        # still need a forward (l1b) result for the day's
                        # curtain.nc -- forward-only is cheap (~seconds),
                        # so just recompute it rather than also persisting
                        # l1b state from the earlier run.
                        data = simulator.run(
                            FORWARD_ONLY_PRODUCTS,
                            {**sim_input, "constituents": constituents},
                        )
                    elif RUN_L2:
                        with contextlib.redirect_stdout(l2_stdout):
                            data = simulator.run(
                                products,
                                {**sim_input, "constituents": constituents},
                            )
                    else:
                        data = simulator.run(
                            products,
                            {**sim_input, "constituents": constituents},
                        )
                except ValueError as e:
                    if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                        # night-side tangent point — physically unobservable,
                        # not an error condition. Skip this observation.
                        skip_obs = True
                        break
                    raise
                except Exception:
                    # An unexpected failure on THIS profile (not the known
                    # night-side case) must not discard everything already
                    # completed today. Log it, record it in the diagnostics
                    # CSV as an error row, and move on to the next
                    # observation/case rather than propagating up to the
                    # top-level except that would return "FAIL" for the
                    # whole day.
                    tb = traceback.format_exc()
                    log.error(
                        "[%s] %s at %s FAILED (unexpected exception), skipping "
                        "this profile only:\n%s",
                        sim_date_str, label, time_key, tb,
                    )
                    if RUN_L2:
                        _append_l2_diag_row(l2_diag_csv_path, {
                            "case_label": label, "time": time_key,
                            "lat": lat, "lon": lon, "elapsed_s": None,
                            "converged": None, "termination_reason": None,
                            "n_function_evaluations": None,
                            "l2_num_iterations": None, "l2_final_cost": None,
                            "status": "error", "error": str(tb)[-500:],
                        })
                    continue
                obs_elapsed = time.perf_counter() - obs_t0

                ds_obs = _l1b_image_to_dataset(data["l1b"])
                # Truth extinction was computed on ALT_GRID_M (the full
                # atmospheric state grid, e.g. 0-65km/1km steps) — a
                # DIFFERENT grid than the instrument's actual line-of-sight
                # sampling altitudes (ds_obs's existing altitude_m coord,
                # e.g. 10-40km/500m steps). These don't overlap point-for-
                # point, so we store both: the native truth (exact, no
                # extra interpolation) on its own atm_altitude_m dim, and
                # an interpolated version on the instrument's altitude_m
                # dim for direct point-by-point comparison against
                # radiance/dolp without needing to align grids later.
                for ext_key, ext_vals in true_ext.items():
                    if ext_key == "extinction_wavelength_nm":
                        continue  # not a per-altitude field, skip

                    # native grid, exact — new "atm_altitude_m" dimension
                    native_key = f"{ext_key}_atm"
                    ds_obs[native_key] = (("wavelength", "atm_altitude_m"), ext_vals)

                    # interpolated onto the instrument's altitude_m grid,
                    # per wavelength (linear interp; instrument grid's
                    # 10-40km range sits safely inside ALT_GRID_M's
                    # 0-65km range, so no extrapolation occurs)
                    instrument_alt = ds_obs["altitude_m"].values
                    interp_vals = np.array([
                        np.interp(instrument_alt, ALT_GRID_M, ext_vals[i, :])
                        for i in range(ext_vals.shape[0])
                    ])
                    ds_obs[ext_key] = (("wavelength", "altitude_m"), interp_vals)

                ds_obs = ds_obs.assign_coords(atm_altitude_m=ALT_GRID_M)
                obs_l1b[label] = ds_obs

                if RUN_L2:
                    if already_done:
                        # Reload the previous run's saved profile so it's
                        # included in this day's final l2_retrieval.nc
                        # curtain, in the same position as newly-computed
                        # profiles.
                        saved_path = _l2_profile_path(out_root, label, sim_date_str, t)
                        reloaded = _load_saved_l2_profile(saved_path)
                        if reloaded is not None:
                            l2_results[label].append(reloaded)
                    else:
                        l2_obj = data.get("l2")
                        diag = _parse_scipy_convergence(l2_stdout.getvalue())
                        native_diag = _extract_l2_native_diagnostics(l2_obj)

                        if l2_obj is not None:
                            l2_results[label].append(l2_obj)
                            saved_path = _l2_profile_path(out_root, label, sim_date_str, t)
                            saved_ok, save_err = _save_l2_profile(l2_obj, saved_path)
                            if not saved_ok:
                                log.warning("[%s] %s at %s: failed to save L2 "
                                            "profile (%s)", sim_date_str, label,
                                            time_key, save_err)

                        _append_l2_diag_row(l2_diag_csv_path, {
                            "case_label": label, "time": time_key,
                            "lat": lat, "lon": lon, "elapsed_s": obs_elapsed,
                            "converged": diag["converged"],
                            "termination_reason": diag["termination_reason"],
                            "n_function_evaluations": diag["n_function_evaluations"],
                            "l2_num_iterations": native_diag["l2_num_iterations"],
                            "l2_final_cost": native_diag["l2_final_cost"],
                            "status": "ok", "error": None,
                        })

                        n_l2_done_this_run += 1
                        # L2 profiles are expensive (100-600s measured) — log
                        # progress periodically so a running job isn't silent
                        # for hours.
                        if n_l2_done_this_run % 10 == 0:
                            elapsed_min = (time.perf_counter() - l2_day_t0) / 60
                            log.info(
                                "[%s] L2 progress: %d NEW profiles done this run "
                                "(%d resumed from a previous run), %.1f min elapsed, "
                                "last=%s (%.1fs, converged=%s)",
                                sim_date_str, n_l2_done_this_run, len(completed_l2_keys),
                                elapsed_min, label, obs_elapsed, diag["converged"],
                            )

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

        # save L2 retrieval output per case, if requested. Same along_track
        # coordinate convention as the forward curtain, but kept as its own
        # file since the L2 dataset's internal grid/dims are independent of
        # the forward curtain's altitude_m. Includes both profiles computed
        # this run and any reloaded from a previous run's l2_profiles/ save
        # (profile-level resume), so a day finished across multiple
        # sbatch submissions still ends up with one complete curtain.
        if RUN_L2:
            for label, l2_list in l2_results.items():
                if not l2_list:
                    continue
                case_out = os.path.join(out_root, label, sim_date_str)
                os.makedirs(case_out, exist_ok=True)
                try:
                    l2_curtain = xr.concat(l2_list, dim="along_track")
                    l2_curtain = l2_curtain.assign_coords(
                        lat=("along_track", lats),
                        lon=("along_track", lons),
                        time=("along_track", times),
                    )
                    l2_curtain.to_netcdf(os.path.join(case_out, "l2_retrieval.nc"))
                except Exception:
                    # Don't let an L2-save failure (e.g. inconsistent dims
                    # across profiles) take down the forward output for the
                    # day, which has already been written above. The
                    # per-profile l2_profiles/*.nc files are unaffected by
                    # this and remain available for a later resume/retry of
                    # just the concat step.
                    log.error(
                        "[%s] failed to concat/save l2_retrieval.nc for case %s "
                        "(per-profile l2_profiles/*.nc are still on disk and "
                        "safe):\n%s",
                        sim_date_str, label, traceback.format_exc(),
                    )

            if os.path.exists(l2_diag_csv_path):
                diag_df = pd.read_csv(l2_diag_csv_path)
                n_conv = int(diag_df["converged"].sum(skipna=True)) if "converged" in diag_df else 0
                n_conv_known = int(diag_df["converged"].notna().sum())
                n_errors = int((diag_df["status"] == "error").sum()) if "status" in diag_df else 0
                if n_conv_known and n_conv < n_conv_known:
                    log.warning(
                        "[%s] L2 convergence: %d/%d converged (%d did not)",
                        sim_date_str, n_conv, n_conv_known, n_conv_known - n_conv,
                    )
                if n_errors:
                    log.warning("[%s] L2: %d profile(s) failed with an unexpected "
                                "error (status=error in l2_diagnostics.csv)",
                                sim_date_str, n_errors)

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

        log.info("[%s] %d/%d obs (daytime), %d cases%s",
                  sim_date_str, len(successful_obs), len(observations), len(h2_files),
                  " (L2 retrieval on)" if RUN_L2 else "")
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

    if RUN_L2:
        log.warning(
            "RUN_L2 is enabled. L2 retrieval measured at 100-600s/profile in "
            "benchmark_l2_retrieval.py -- if you haven't already, run "
            "extrapolate_to_full_run() against your real l2_benchmark_results.csv "
            "to confirm walltime/CPU-hour budget before this job scales up."
        )

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

    if RUN_START_DATE or RUN_END_DATE:
        log.info("Restricting this run to date range: %s to %s (of %s to %s available)",
                  RUN_START_DATE or bg_dates[0], RUN_END_DATE or bg_dates[-1],
                  bg_dates[0], bg_dates[-1])

    # build job list
    jobs = []
    n_already_done = 0
    for i, date_str in enumerate(bg_dates):
        # map simulation day to orbit day (repeating pattern). Computed from
        # i = the date's position in the FULL bg_dates list, so that
        # filtering to a date-range subset below doesn't change which
        # orbit_day a given date maps to. This keeps staged runs (e.g. only
        # Jan-May now, June-July later) consistent with a single full run.
        orbit_day = i % n_orbit_days

        # optional date-range restriction for staged runs
        if RUN_START_DATE and date_str < RUN_START_DATE:
            continue
        if RUN_END_DATE and date_str > RUN_END_DATE:
            continue

        if orbit_day not in orbit_day_idx:
            log.warning("No orbit files for orbit day %d, skipping %s",
                        orbit_day, date_str)
            continue

        # Day-level resume: skip dates whose expected outputs already exist
        # for every case, so re-submitting a job (e.g. after a walltime
        # limit forced multiple sequential sbatch runs across the 6-month
        # period) doesn't reprocess completed days from scratch. Days that
        # are only PARTIALLY done (e.g. L2 killed partway through) are not
        # matched here and get resubmitted -- process_day()'s profile-level
        # resume then picks up only the remaining work for that day.
        if _day_already_done(date_str, case_labels, OUT_DIR):
            n_already_done += 1
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

    if n_already_done:
        log.info("Skipping %d day(s) already fully completed from a previous run",
                  n_already_done)
    log.info("Processing %d days with %d cases each",
             len(jobs), len(case_labels))

    # pre-warm calibration database (now idempotent — safe to call again
    # even if a worker also triggers it via _get_simulator())
    log.info("Pre-warming calibration database...")
    try:
        _cal_mod.calibration_database("ideal_spectrograph", "v1")
        log.info("Calibration database ready.")
    except Exception as e:
        log.warning("Could not pre-warm calibration database: %s", e)

    # pre-warm mode-specific Mie databases (accumulation/coarse extinction).
    # First build triggers a real Mie calculation per mode_width; doing
    # this once, serially, before dispatching workers avoids relying on
    # unconfirmed concurrent-build safety in sasktran2's MieDatabase.
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

        # max_tasks_per_child=1: each worker process is discarded and
        # replaced with a fresh one after finishing ONE day, rather than
        # persisting across every day dispatched to it for the life of the
        # job. This directly targets a real failure: a 90-worker RUN_L2 job
        # OOM-killed ~5 hours in despite mem-per-cpu being deliberately kept
        # under the node's total memory, most likely from memory that isn't
        # released between days within a long-lived worker (WACCMAtmosphere
        # opens h2 NetCDF files via xarray but is never explicitly closed;
        # _get_simulator()'s per-worker IdealALISimulator singleton was also
        # designed to persist across days). Restarting workers per-day caps
        # any such growth instead of letting it accumulate for hours.
        #
        # This does cost re-constructing IdealALISimulator() once per day
        # instead of once per worker-lifetime, but the calibration-database
        # patch earlier made that idempotent against the pre-warmed NFS
        # cache, so the actual overhead should be small relative to a
        # 100-600s/profile L2 day. If jobs still OOM even with this in
        # place, that points to a single day's own memory footprint being
        # the problem (not cross-day accumulation) and mem-per-cpu itself
        # needs to go up, not max_tasks_per_child down further.
        try:
            with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
                futures = {pool.submit(process_day, *job): job[0] for job in jobs}
                for fut in as_completed(futures):
                    result = fut.result()
                    results.append(result)
                    log.info("Completed: %s", result)
        except BrokenProcessPool:
            # A worker was killed abruptly (e.g. OOM by the OS/cgroup, not a
            # Python exception) -- this poisons the whole pool, so every
            # OTHER worker's in-progress day is abandoned too, even ones
            # that were fine. Nothing already written to disk is lost
            # (day-level curtain.nc/l2_retrieval.nc for completed days, and
            # profile-level l2_diagnostics.csv rows / l2_profiles/*.nc for
            # partially-done days) -- day-level and profile-level resume in
            # main()/process_day() mean simply re-submitting this exact job
            # picks up from here rather than redoing finished work. Check
            # `sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,NNodes,State`
            # for the actual peak memory reached before deciding whether to
            # also adjust --mem-per-cpu/--cpus-per-task before resubmitting.
            log.error(
                "Process pool broke, likely from a worker being killed "
                "abruptly (check for an oom_kill event in the SLURM .err "
                "log). Days already completed and profiles already saved "
                "are safe on disk. Re-submit this exact job -- day-level "
                "and profile-level resume will skip everything already "
                "done and continue from where this run stopped."
            )
            raise

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
