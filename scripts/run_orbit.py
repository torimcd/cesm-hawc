#!/usr/bin/env python
"""
run_orbit.py
============
Simulate HAWCSat ALI observations along a realistic orbital ground track,
using hourly CESM h1 output files so each observation gets the
contemporaneous atmospheric state.

Each orbit observation is matched to the nearest h1 file by timestamp.
Solar geometry (SZA, SAA) is computed automatically from observation time
and tangent-point position via sasktran2's SolarGeometryHandlerAstropy —
no explicit angles needed in the config.

Edit config.toml at the project root (copy config.example.toml to start),
then run:

    python scripts/run_orbit.py

or submit via SLURM (parallel across days):

    sbatch slurm/submit.sh

Parallelism
-----------
Set n_workers > 1 to process days in parallel using a process pool.
Each day is independent (embarrassingly parallel).
Set n_workers = 1 to run serially — useful for debugging.

Output layout
-------------
OUT_DIR/
  YYYY-MM-DD/
    orbit_track.csv          # time, lat, lon, bg_file per observation
    curtain_background.nc    # dims [along_track, altitude_m]; coords lat, lon, time
    curtain_injection.nc     # same for injection case (if configured)
    summary.txt              # per-observation peak stats
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

_o   = _cfg["orbit"]
_ins = _cfg["instrument"]

WACCM_BACKGROUND_DIR = _o["waccm_background_dir"]
WACCM_INJECTION_DIR  = _o["waccm_injection_dir"] or None
H2_PATTERN           = _o["h2_pattern"]
START_TIME_STR       = _o["start_time"] or None
END_TIME_STR         = _o["end_time"]   or None

ALTITUDE_KM     = float(_o["altitude_km"])
INCLINATION_DEG = float(_o["inclination_deg"])
START_LON_DEG   = float(_o["start_lon_deg"])
OBS_CADENCE_S   = float(_o["obs_cadence_s"])
MAX_GAP_S       = float(_o["max_gap_s"])

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

OUT_DIR   = os.path.expanduser(_o["out_dir"])
N_WORKERS = int(_o["n_workers"])

# ── END CONFIGURATION ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── File discovery ─────────────────────────────────────────────────────────────

def _parse_h1_timestamp(filename: str) -> Optional[pd.Timestamp]:
    """
    Parse timestamp from a CESM h1 filename.
    Expected pattern: *.cam.h1.YYYY-MM-DD-SSSSS.nc
    where SSSSS = seconds of day (0=midnight, 3600=01:00, etc.).
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})-(\d{5})", filename)
    if m:
        return pd.Timestamp(m.group(1)) + pd.Timedelta(seconds=int(m.group(2)))
    return None


def build_file_index(directory: str, pattern: str) -> dict[pd.Timestamp, str]:
    """Return {timestamp: filepath} for all h1 files in directory."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in: {directory}"
        )
    index: dict[pd.Timestamp, str] = {}
    for p in paths:
        ts = _parse_h1_timestamp(os.path.basename(p))
        if ts is None:
            log.warning("Could not parse timestamp from filename, skipping: %s", p)
            continue
        index[ts] = p
    log.info("Found %d h1 file(s) in %s", len(index), directory)
    return index


def find_nearest_file(
    obs_time: pd.Timestamp,
    sorted_timestamps: list[pd.Timestamp],
    file_index: dict[pd.Timestamp, str],
    max_gap_s: float,
) -> Optional[str]:
    """Return the path of the h1 file nearest to obs_time, or None if too far."""
    times_s = np.array([t.timestamp() for t in sorted_timestamps])
    t0 = obs_time.timestamp()
    idx = int(np.searchsorted(times_s, t0))
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(times_s):
        candidates.append(idx)
    best = min(candidates, key=lambda i: abs(times_s[i] - t0))
    if abs(times_s[best] - t0) > max_gap_s:
        return None
    return file_index[sorted_timestamps[best]]


# ── Orbit ground track ─────────────────────────────────────────────────────────

def generate_sso_ground_track(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    cadence_s: float,
    altitude_km: float = 600.0,
    inclination_deg: float = 98.0,
    start_lon_deg: float = 0.0,
) -> pd.DataFrame:
    """
    Generate an analytical sun-synchronous orbit ground track.

    Uses circular Keplerian mechanics with Earth's rotation. Good enough for
    a representative HAWCSat orbit; replace with TLE/sgp4 propagation for
    exact ground tracks once TLE lines are available.

    Returns
    -------
    DataFrame with columns: time (pd.Timestamp), lat (degrees), lon (degrees)
    """
    RE = 6_371_000.0         # m  — Earth mean radius
    GM = 3.986004418e14      # m³ s⁻²
    a  = RE + altitude_km * 1e3
    T  = 2.0 * np.pi * np.sqrt(a ** 3 / GM)   # orbital period [s]

    omega_orb = 2.0 * np.pi / T
    omega_E   = 7.2921150e-5                   # Earth rotation rate [rad/s]
    inc       = np.radians(inclination_deg)
    raan      = np.radians(start_lon_deg)
    total_s   = (end_time - start_time).total_seconds()

    records: list[dict] = []
    t = 0.0
    while t <= total_s:
        theta = omega_orb * t
        lat   = np.degrees(np.arcsin(np.sin(inc) * np.sin(theta)))
        u     = np.arctan2(np.cos(inc) * np.sin(theta), np.cos(theta))
        lon   = np.degrees(raan + u - omega_E * t)
        lon   = ((lon + 180.0) % 360.0) - 180.0   # wrap to [-180, 180]
        records.append({
            "time": start_time + pd.Timedelta(seconds=t),
            "lat":  float(lat),
            "lon":  float(lon),
        })
        t += cadence_s

    df = pd.DataFrame(records)
    log.info(
        "Generated %d orbit observations over %.1f days "
        "(cadence %.0f s, period %.1f min)",
        len(df),
        total_s / 86400,
        cadence_s,
        T / 60,
    )
    return df


# ── Per-day simulation ─────────────────────────────────────────────────────────

def process_orbit_day(
    date_str: str,
    day_obs: list[dict],   # list of {time, lat, lon, bg_file, inj_file}
    out_root: str,
) -> str:
    """
    Simulate all orbit observations for one calendar day.

    Called once per day, either directly (serial) or from a worker process
    (parallel).  Returns a short status string.

    Each observation uses the h1 file matched to its timestamp.  Files are
    cached within the day's worker to avoid repeated opens when consecutive
    observations share the same snapshot.
    """
    try:
        day_out = os.path.join(out_root, date_str)
        os.makedirs(day_out, exist_ok=True)

        simulator = IdealALISimulator()

        # Cache open WACCMAtmosphere objects keyed by file path
        waccm_cache: dict[str, WACCMAtmosphere] = {}

        def get_waccm(path: str) -> WACCMAtmosphere:
            if path not in waccm_cache:
                waccm_cache[path] = WACCMAtmosphere(
                    path, alt_grid_km=ALT_GRID_M / 1e3
                )
            return waccm_cache[path]

        l2_bg_list:  list[xr.Dataset] = []
        l2_inj_list: list[xr.Dataset] = []
        summary_rows: list[dict] = []

        for obs in day_obs:
            t   = obs["time"]
            lat = obs["lat"]
            lon = obs["lon"]
            bg_file  = obs["bg_file"]
            inj_file = obs.get("inj_file")

            sim_input = {
                "tangent_latitude":    float(lat),
                "tangent_longitude":   float(lon),
                "altitude_grid":       ALT_GRID_M,
                "polarization_states": ["I", "dolp"],
                "sample_wavelengths":  ALI_WAVELENGTHS,
                "time":                t,
                # No sza_deg / saa_deg → SolarGeometryHandlerAstropy computes
                # solar geometry automatically from time + tangent position.
            }
            if NOISE_MODEL is not None:
                sim_input["l1b_cfg"] = {"noise_model": NOISE_MODEL}

            waccm_bg    = get_waccm(bg_file)
            profiles_bg = waccm_bg.get_column_profiles(lat, lon, time_index=0)

            data_bg = simulator.run(
                ["l2", "sk2_atmosphere", "front_end_radiance", "l1b"],
                {**sim_input,
                 "constituents": build_waccm_constituents(profiles_bg, ALT_GRID_M)},
            )

            data_inj = None
            if inj_file is not None:
                waccm_inj    = get_waccm(inj_file)
                profiles_inj = waccm_inj.get_column_profiles(lat, lon, time_index=0)
                data_inj = simulator.run(
                    ["l2", "sk2_atmosphere", "front_end_radiance", "l1b"],
                    {**sim_input,
                     "constituents": build_waccm_constituents(profiles_inj, ALT_GRID_M)},
                )

            l2_bg_list.append(data_bg["l2"])

            # Derive peak anomaly if injection was run
            peak_ext_anom = None
            peak_r_anom   = None
            if data_inj is not None:
                l2_inj_list.append(data_inj["l2"])
                ext_bg  = data_bg["l2"]["stratospheric_aerosol_extinction_per_m"]
                ext_inj = data_inj["l2"]["stratospheric_aerosol_extinction_per_m"]
                r_bg    = data_bg["l2"]["stratospheric_aerosol_median_radius"]
                r_inj   = data_inj["l2"]["stratospheric_aerosol_median_radius"]
                strat   = ext_bg.altitude.values > 15000
                peak_ext_anom = float((ext_inj - ext_bg).values[strat].max())
                peak_r_anom   = float((r_inj   - r_bg).values[strat].max())

            summary_rows.append({
                "time":           t.isoformat(),
                "lat":            lat,
                "lon":            lon,
                "peak_ext_anom_per_m": peak_ext_anom,
                "peak_r_anom_nm":      peak_r_anom,
            })

        # ── Assemble curtain datasets ─────────────────────────────────────────
        lats  = [o["lat"]  for o in day_obs]
        lons  = [o["lon"]  for o in day_obs]
        times = [o["time"] for o in day_obs]

        def make_curtain(l2_list: list[xr.Dataset]) -> xr.Dataset:
            curtain = xr.concat(l2_list, dim="along_track")
            curtain = curtain.assign_coords(
                lat=("along_track", lats),
                lon=("along_track", lons),
                time=("along_track", times),
            )
            return curtain

        curtain_bg = make_curtain(l2_bg_list)
        curtain_bg.to_netcdf(os.path.join(day_out, "curtain_background.nc"))
        log.info("[%s] curtain_background.nc  (%d obs)", date_str, len(l2_bg_list))

        if l2_inj_list:
            curtain_inj = make_curtain(l2_inj_list)
            curtain_inj.to_netcdf(os.path.join(day_out, "curtain_injection.nc"))
            log.info("[%s] curtain_injection.nc  (%d obs)", date_str, len(l2_inj_list))

        # ── orbit_track.csv ───────────────────────────────────────────────────
        track_df = pd.DataFrame([
            {"time": o["time"], "lat": o["lat"], "lon": o["lon"],
             "bg_file": o["bg_file"]}
            for o in day_obs
        ])
        track_df.to_csv(os.path.join(day_out, "orbit_track.csv"), index=False)

        # ── summary.txt ───────────────────────────────────────────────────────
        lines = [f"Date: {date_str}", f"Observations: {len(day_obs)}", ""]
        for row in summary_rows:
            line = f"  {row['time']}  lat={row['lat']:+7.2f}  lon={row['lon']:+8.2f}"
            if row["peak_ext_anom_per_m"] is not None:
                line += (f"  Δext={row['peak_ext_anom_per_m']:.2e} m⁻¹"
                         f"  Δr={row['peak_r_anom_nm']:.1f} nm")
            lines.append(line)
        summary_text = "\n".join(lines)
        with open(os.path.join(day_out, "summary.txt"), "w") as f:
            f.write(summary_text + "\n")

        return f"OK   {date_str}  ({len(day_obs)} obs)"

    except Exception:
        tb = traceback.format_exc()
        log.error("[%s] FAILED:\n%s", date_str, tb)
        return f"FAIL {date_str}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Build hourly file indices ──────────────────────────────────────────────
    bg_index  = build_file_index(WACCM_BACKGROUND_DIR, H2_PATTERN)
    inj_index: dict[pd.Timestamp, str] = {}
    if WACCM_INJECTION_DIR is not None:
        inj_index = build_file_index(WACCM_INJECTION_DIR, H2_PATTERN)

    sorted_bg_ts  = sorted(bg_index.keys())
    sorted_inj_ts = sorted(inj_index.keys())

    # ── Determine simulation time range ───────────────────────────────────────
    start_time = (pd.Timestamp(START_TIME_STR) if START_TIME_STR
                  else sorted_bg_ts[0])
    end_time   = (pd.Timestamp(END_TIME_STR)   if END_TIME_STR
                  else sorted_bg_ts[-1])
    log.info("Simulation period: %s → %s", start_time, end_time)

    # ── Generate orbit ground track ────────────────────────────────────────────
    track = generate_sso_ground_track(
        start_time=start_time,
        end_time=end_time,
        cadence_s=OBS_CADENCE_S,
        altitude_km=ALTITUDE_KM,
        inclination_deg=INCLINATION_DEG,
        start_lon_deg=START_LON_DEG,
    )

    # ── Match each observation to the nearest h1 file ─────────────────────────
    track["bg_file"] = track["time"].apply(
        lambda t: find_nearest_file(t, sorted_bg_ts, bg_index, MAX_GAP_S)
    )
    if inj_index:
        track["inj_file"] = track["time"].apply(
            lambda t: find_nearest_file(t, sorted_inj_ts, inj_index, MAX_GAP_S)
        )
    else:
        track["inj_file"] = None

    n_unmatched = track["bg_file"].isna().sum()
    if n_unmatched:
        log.warning(
            "Dropping %d observations with no h1 file within %.0f s",
            n_unmatched, MAX_GAP_S,
        )
    track = track.dropna(subset=["bg_file"]).reset_index(drop=True)
    log.info("%d observations matched to h1 files", len(track))

    # ── Partition by calendar day → build job list ────────────────────────────
    track["_date"] = track["time"].dt.strftime("%Y-%m-%d")
    jobs = []
    for date_str, day_df in track.groupby("_date"):
        day_obs = day_df.drop(columns="_date").to_dict(orient="records")
        jobs.append((date_str, day_obs, OUT_DIR))

    log.info("Processing %d day(s): %s … %s",
             len(jobs), jobs[0][0], jobs[-1][0])

    # ── Pre-warm calibration database (avoid race condition in workers) ────────
    log.info("Pre-warming calibration database...")
    try:
        from hawcsimulator.ali.calibration import calibration_database
        calibration_database("ideal_spectrograph", "v1")
        log.info("Calibration database ready.")
    except Exception as e:
        log.warning("Could not pre-warm calibration database: %s", e)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    results: list[str] = []
    if N_WORKERS <= 1:
        for job in jobs:
            results.append(process_orbit_day(*job))
    else:
        workers = min(N_WORKERS, len(jobs))
        log.info("Dispatching to %d worker processes...", workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_orbit_day, *job): job[0]
                       for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                log.info("Completed: %s", result)

    # ── Final report ──────────────────────────────────────────────────────────
    ok   = [r for r in results if r.startswith("OK")]
    fail = [r for r in results if r.startswith("FAIL")]
    log.info("\n── Run complete ──────────────────────────────")
    log.info("  Succeeded: %d / %d", len(ok), len(results))
    if fail:
        log.warning("  Failed days:")
        for f in fail:
            log.warning("    %s", f)
    log.info("  Output root: %s", OUT_DIR)


if __name__ == "__main__":
    main()
