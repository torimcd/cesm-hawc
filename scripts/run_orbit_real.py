#!/usr/bin/env python
"""
run_orbit_real.py
=================
Simulate HAWC ALI observations using real orbit geometry files matched to
daily CESM h2 atmospheric output.

Each orbit file covers one pass (~70 min, 4218 time steps at 1 Hz). Files are
grouped by calendar date and matched to the h2 file for that day. Observer
geometry (satellite position) and tangent-point positions are read directly
from the orbit NetCDF; the atmospheric state comes from WACCM h2 output.

Edit config.toml at the project root (copy config.example.toml to start),
then run:

    python scripts/run_orbit_real.py

Parallelism
-----------
Set n_workers > 1 in [orbit_real] to process orbit files in parallel using
a process pool. Each orbit file is an independent job.
Set n_workers = 1 to run serially — useful for debugging.

Output layout
-------------
OUT_DIR/
  YYYY-MM-DD/
    orbit_0001_l2_bg.nc    # L2 background, coords: obs_time, across_idx, lat, lon
    orbit_0001_l2_inj.nc   # L2 injection (if injection dir configured)
    orbit_0002_l2_bg.nc
    ...
    summary.txt
"""

from __future__ import annotations

import glob
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

from cesm_hawc.waccm import WACCMAtmosphere
from cesm_hawc.constituents import build_waccm_constituents
from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator
from hawcsimulator.noise import ALINoiseModel

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

_CONFIG = Path(__file__).parent.parent / "config.toml"
if not _CONFIG.exists():
    sys.exit(
        f"config.toml not found at {_CONFIG}\n"
        "Copy config.example.toml → config.toml and fill in your paths."
    )
with open(_CONFIG, "rb") as _f:
    _cfg = tomllib.load(_f)

_o   = _cfg["orbit_real"]
_ins = _cfg["instrument"]

ORBIT_DIR            = os.path.expanduser(_o["orbit_dir"])
ORBIT_PATTERN        = _o["orbit_pattern"]
WACCM_BACKGROUND_DIR = os.path.expanduser(_o["waccm_background_dir"])
WACCM_INJECTION_DIR  = os.path.expanduser(_o["waccm_injection_dir"]) if _o["waccm_injection_dir"] else None
H2_PATTERN           = _o["h2_pattern"]
OUT_DIR              = os.path.expanduser(_o["out_dir"])
N_WORKERS            = int(_o.get("n_workers", 1))

# [] means all 512 pixels; [256] means center only; etc.
ACROSS_INDICES: list[int] = list(_o.get("across_indices", []))
TIME_STRIDE = int(_o.get("time_stride", 1))

ALI_WAVELENGTHS = np.array(_ins["wavelengths_nm"])
ALT_GRID_M = np.arange(
    _ins["alt_grid_start_m"],
    _ins["alt_grid_stop_m"] + _ins["alt_grid_step_m"],
    _ins["alt_grid_step_m"],
)

_noise_frac = _ins.get("noise_straylight_fraction", "")
NOISE_MODEL: ALINoiseModel | None = (
    ALINoiseModel(straylight_fraction=float(_noise_frac))
    if _noise_frac != "" else None
)

# ── END CONFIGURATION ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── File discovery ─────────────────────────────────────────────────────────────

def _parse_h2_date(filename: str) -> Optional[str]:
    """Return YYYY-MM-DD from a CESM h2 filename, or None."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})-\d{5}", filename)
    return m.group(1) if m else None


def build_h2_index(directory: str, pattern: str) -> dict[str, str]:
    """Return {YYYY-MM-DD: filepath} for all h2 files in directory."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in: {directory}"
        )
    index: dict[str, str] = {}
    for p in paths:
        date = _parse_h2_date(os.path.basename(p))
        if date is None:
            log.warning("Could not parse date from filename, skipping: %s", p)
            continue
        index[date] = p
    log.info("Found %d h2 file(s) in %s", len(index), directory)
    return index


def _orbit_file_date(orbit_path: str) -> str:
    """Return YYYY-MM-DD for the calendar date of an orbit file."""
    with xr.open_dataset(orbit_path, decode_times=False) as ds:
        start = ds.attrs.get("start_time", "")
    if start:
        return str(pd.Timestamp(start).date())
    # fallback: look for YYYY-MM-DD in the filename
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(orbit_path))
    if m:
        return m.group(1)
    raise ValueError(f"Cannot determine calendar date for orbit file: {orbit_path}")


def collect_orbit_files(directory: str, pattern: str) -> dict[str, list[str]]:
    """Return {YYYY-MM-DD: [orbit_path, ...]} grouped by calendar date."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No orbit files matching '{pattern}' found in: {directory}"
        )
    grouped: dict[str, list[str]] = {}
    for p in paths:
        date = _orbit_file_date(p)
        grouped.setdefault(date, []).append(p)
    log.info(
        "Found %d orbit file(s) spanning %d calendar day(s)",
        len(paths), len(grouped),
    )
    return grouped


# ── Per-orbit-file simulation ──────────────────────────────────────────────────

def process_orbit_file(
    orbit_path: str,
    h2_bg_path: str,
    h2_inj_path: str | None,
    out_dir: str,
) -> str:
    """
    Simulate all selected observations in one orbit file.

    Called once per orbit file, either directly (serial) or from a worker
    process (parallel). Returns a short status string.

    Module-level constants ACROSS_INDICES, TIME_STRIDE, ALT_GRID_M,
    ALI_WAVELENGTHS, and NOISE_MODEL are used directly.
    """
    orbit_name = os.path.splitext(os.path.basename(orbit_path))[0]
    try:
        orbit = xr.open_dataset(orbit_path, decode_times=True)
        n_time   = orbit.sizes["time"]
        n_across = orbit.sizes["across"]

        across_idx_list = ACROSS_INDICES if ACROSS_INDICES else list(range(n_across))
        time_idx_list   = list(range(0, n_time, TIME_STRIDE))

        waccm_bg  = WACCMAtmosphere(h2_bg_path,  alt_grid_km=ALT_GRID_M / 1e3)
        waccm_inj = (
            WACCMAtmosphere(h2_inj_path, alt_grid_km=ALT_GRID_M / 1e3)
            if h2_inj_path else None
        )
        simulator = IdealALISimulator()

        l2_bg_list:  list[xr.Dataset] = []
        l2_inj_list: list[xr.Dataset] = []
        meta: list[dict] = []

        for t_idx in time_idx_list:
            obs_lat  = float(orbit["observer_latitude"].isel(time=t_idx))
            obs_lon  = float(orbit["observer_longitude"].isel(time=t_idx))
            obs_alt  = float(orbit["observer_altitude"].isel(time=t_idx))
            obs_time = pd.Timestamp(orbit["time"].isel(time=t_idx).values)

            for ac_idx in across_idx_list:
                lat = float(orbit["latitude"].isel(time=t_idx, along=0, across=ac_idx))
                lon = float(orbit["longitude"].isel(time=t_idx, along=0, across=ac_idx))

                # Skip fill-value pixels (ocean limb, etc.)
                if not np.isfinite(lat) or not np.isfinite(lon):
                    continue

                profiles_bg = waccm_bg.get_column_profiles(lat, lon, time_index=0)

                sim_input = {
                    "tangent_latitude":    lat,
                    "tangent_longitude":   lon,
                    "observer_latitude":   obs_lat,
                    "observer_longitude":  obs_lon,
                    "observer_altitude":   obs_alt,
                    "altitude_grid":       ALT_GRID_M,
                    "polarization_states": ["I", "dolp"],
                    "sample_wavelengths":  ALI_WAVELENGTHS,
                    "time":                obs_time,
                    "constituents":        build_waccm_constituents(profiles_bg, ALT_GRID_M),
                }
                if NOISE_MODEL is not None:
                    sim_input["l1b_cfg"] = {"noise_model": NOISE_MODEL}

                data_bg = simulator.run(["l2", "l1b"], sim_input)
                l2_bg_list.append(data_bg["l2"])

                if waccm_inj is not None:
                    profiles_inj = waccm_inj.get_column_profiles(lat, lon, time_index=0)
                    data_inj = simulator.run(
                        ["l2", "l1b"],
                        {**sim_input,
                         "constituents": build_waccm_constituents(profiles_inj, ALT_GRID_M)},
                    )
                    l2_inj_list.append(data_inj["l2"])

                meta.append({
                    "time":      obs_time,
                    "across":    ac_idx,
                    "lat":       lat,
                    "lon":       lon,
                })

        os.makedirs(out_dir, exist_ok=True)

        def _save(l2_list: list[xr.Dataset], tag: str) -> None:
            curtain = xr.concat(l2_list, dim="obs")
            curtain = curtain.assign_coords(
                obs_time  =("obs", [r["time"]   for r in meta]),
                across_idx=("obs", [r["across"] for r in meta]),
                lat       =("obs", [r["lat"]    for r in meta]),
                lon       =("obs", [r["lon"]    for r in meta]),
            )
            out_path = os.path.join(out_dir, f"{orbit_name}_{tag}.nc")
            curtain.to_netcdf(out_path)
            log.info("  Wrote %s  (%d obs)", out_path, len(l2_list))

        _save(l2_bg_list, "l2_bg")
        if l2_inj_list:
            _save(l2_inj_list, "l2_inj")

        orbit.close()
        return f"OK   {orbit_name}  ({len(meta)} obs)"

    except Exception:
        tb = traceback.format_exc()
        log.error("[%s] FAILED:\n%s", orbit_name, tb)
        return f"FAIL {orbit_name}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    bg_index  = build_h2_index(WACCM_BACKGROUND_DIR, H2_PATTERN)
    inj_index: dict[str, str] = {}
    if WACCM_INJECTION_DIR:
        inj_index = build_h2_index(WACCM_INJECTION_DIR, H2_PATTERN)

    orbit_by_date = collect_orbit_files(ORBIT_DIR, ORBIT_PATTERN)

    # Build job list: (orbit_path, h2_bg_path, h2_inj_path, out_dir)
    jobs: list[tuple[str, str, str | None, str]] = []
    for date, orbit_paths in sorted(orbit_by_date.items()):
        if date not in bg_index:
            log.warning(
                "No background h2 file for %s — skipping %d orbit file(s)",
                date, len(orbit_paths),
            )
            continue
        h2_bg  = bg_index[date]
        h2_inj = inj_index.get(date)
        day_out = os.path.join(OUT_DIR, date)
        for op in orbit_paths:
            jobs.append((op, h2_bg, h2_inj, day_out))

    if not jobs:
        log.error("No jobs to run — check that orbit and h2 file dates overlap.")
        sys.exit(1)

    log.info(
        "Processing %d orbit file(s) across %d day(s)  "
        "(time_stride=%d, across=%s)",
        len(jobs),
        len({j[3] for j in jobs}),
        TIME_STRIDE,
        f"all {512}" if not ACROSS_INDICES else str(ACROSS_INDICES),
    )
    if NOISE_MODEL is not None:
        log.info("Noise model enabled  (straylight_fraction=%s)", _noise_frac)

    # Pre-warm calibration database before spawning workers
    log.info("Pre-warming calibration database...")
    try:
        from hawcsimulator.ali.calibration import calibration_database
        calibration_database("ideal_spectrograph", "v1")
        log.info("Calibration database ready.")
    except Exception as e:
        log.warning("Could not pre-warm calibration database: %s", e)

    # Dispatch jobs
    results: list[str] = []
    if N_WORKERS <= 1:
        for job in jobs:
            results.append(process_orbit_file(*job))
    else:
        workers = min(N_WORKERS, len(jobs))
        log.info("Dispatching to %d worker processes...", workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_orbit_file, *job): os.path.basename(job[0])
                for job in jobs
            }
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                log.info("Completed: %s", result)

    # Write per-day summary files
    days_results: dict[str, list[str]] = {}
    for job, result in zip(jobs, results):
        date = os.path.basename(job[3])
        days_results.setdefault(date, []).append(result)

    for date, day_results in sorted(days_results.items()):
        day_out = os.path.join(OUT_DIR, date)
        os.makedirs(day_out, exist_ok=True)
        lines = [f"Date: {date}", f"Orbit files: {len(day_results)}", ""]
        lines += day_results
        with open(os.path.join(day_out, "summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    ok   = [r for r in results if r.startswith("OK")]
    fail = [r for r in results if r.startswith("FAIL")]
    log.info("\n── Run complete ──────────────────────────────")
    log.info("  Succeeded: %d / %d", len(ok), len(results))
    if fail:
        log.warning("  Failed orbit files:")
        for f in fail:
            log.warning("    %s", f)
    log.info("  Output root: %s", OUT_DIR)


if __name__ == "__main__":
    main()
