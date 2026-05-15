#!/usr/bin/env python
"""
run_simulation.py
=================
Run the HAWC ALI simulator across ALL monthly h0 files for a background
and (optionally) an injection case.  Each file is treated as one time step;
outputs are written per-month so nothing is overwritten.

Edit the CONFIGURATION section, then run:

    python run_simulation.py

or submit via SLURM (parallel across months):

    sbatch slurm/submit.sh

Parallelism
-----------
Set N_WORKERS > 1 to process months in parallel using a process pool.
Each month is independent, so this is embarrassingly parallel.
Set N_WORKERS = 1 (or 0) to run serially — useful for debugging or when
memory is tight.

Output layout
-------------
OUT_DIR/
  background/
    YYYY-MM/
      l2_background.nc
      cesm_extinction_background.nc
      summary.txt
  injection/            (if WACCM_INJECTION_DIR is set)
    YYYY-MM/
      l2_injection.nc
      cesm_extinction_injection.nc
      summary.txt
  diff/                 (if both cases present)
    YYYY-MM/
      summary_diff.txt
"""

from __future__ import annotations

import glob
import logging
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cesm_hawc.waccm import WACCMAtmosphere
from cesm_hawc.constituents import build_waccm_constituents
from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

# Directories containing *.cam.h0.YYYY-MM.nc files.
# The script globs for all h0 files it finds here.
WACCM_BACKGROUND_DIR = "/path/to/background/atm/hist/"
WACCM_INJECTION_DIR  = "/path/to/injection/atm/hist/"   # set None to skip

# File glob pattern within each directory
H0_PATTERN = "*.cam.h0.*.nc"

# If set, only process months whose YYYY-MM string matches this list.
# Leave empty ([]) to process all available months.
MONTH_FILTER: list[str] = []   # e.g. ["2034-01", "2034-02"]

# Observation geometry — match your SO₂ injection latitude
TANGENT_LAT = 30.6    # degrees
TANGENT_LON = 180.0   # degrees
SZA_DEG     = 60.0
SAA_DEG     = 0.0

# Which ALI simulator to use: "ideal" or "full"
# "full" requires: pip install ali_l1 -f https://arg.usask.ca/wheels/
SIMULATOR = "ideal"

# ALI sample wavelengths [nm]
# For "ideal": [470, 745, 1020] for dev; [470,525,745,1020,1230,1450,1500] for production.
# For "full":  fixed 11 instrument bands (610–1560 nm), set by detector design.
ALI_WAVELENGTHS = {
    "ideal": np.array([470.0, 745.0, 1020.0]),
    "full":  np.array([610., 676., 755., 869., 950., 1022., 1080., 1225., 1360., 1450., 1560.]),
}[SIMULATOR]

# Altitude grid [m]
ALT_GRID_M = np.arange(0.0, 65001.0, 1000.0)

# Output root directory
OUT_DIR = os.path.expanduser("~/results/hawc_ali/")

# Parallelism: number of worker processes (0 or 1 = serial)
N_WORKERS = 4

# ── END CONFIGURATION ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _date_from_path(path: str) -> Optional[str]:
    """
    Extract YYYY-MM from a CESM h0 filename.
    Returns None if the pattern is not found.
    """
    m = re.search(r"\d{4}-\d{2}", os.path.basename(path))
    return m.group(0) if m else None


def _collect_files(directory: str, pattern: str,
                   month_filter: list[str]) -> dict[str, str]:
    """
    Return {YYYY-MM: filepath} for all h0 files in *directory* matching
    *pattern*.  If *month_filter* is non-empty, only those months are kept.
    """
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in: {directory}"
        )
    result = {}
    for p in paths:
        date = _date_from_path(p)
        if date is None:
            log.warning("Could not parse date from filename, skipping: %s", p)
            continue
        if month_filter and date not in month_filter:
            continue
        result[date] = p

    log.info("Found %d file(s) in %s", len(result), directory)
    return result


def _build_sim_input(obs_time_str: str) -> dict:
    """Return the geometry/instrument dict shared by all simulations."""
    sim_input = {
        "tangent_latitude":            TANGENT_LAT,
        "tangent_longitude":           TANGENT_LON,
        "tangent_solar_zenith_angle":  SZA_DEG,
        "tangent_solar_azimuth_angle": SAA_DEG,
        "altitude_grid":               ALT_GRID_M,
        "sample_wavelengths":          ALI_WAVELENGTHS,
        "time":                        pd.Timestamp(obs_time_str),
    }
    if SIMULATOR == "ideal":
        sim_input["polarization_states"] = ["I", "dolp"]
    return sim_input


def _save_cesm_extinction(profiles: dict, alt_m: np.ndarray,
                          out_dir: str, tag: str) -> None:
    """Save CESM MAM4 extinction profiles directly (no simulator needed)."""
    from cesm_hawc.constituents import _extinction_from_number_density
    from aliprocessing.l2.optical import aerosol_median_radius_db
    import xarray as xr

    mie_db     = aerosol_median_radius_db()
    ext_accum  = _extinction_from_number_density(
        profiles["sulfate_a1_N_cm3"], profiles["sulfate_a1_r_um"], mie_db
    )
    ext_coarse = _extinction_from_number_density(
        profiles["sulfate_a3_N_cm3"], profiles["sulfate_a3_r_um"], mie_db
    )
    ds = xr.Dataset(
        {
            "extinction_total":  ("altitude_m", ext_accum + ext_coarse),
            "extinction_accum":  ("altitude_m", ext_accum),
            "extinction_coarse": ("altitude_m", ext_coarse),
        },
        coords={"altitude_m": alt_m},
        attrs={"description": "CESM MAM4 extinction at 745 nm",
               "wavelength_nm": 745.0},
    )
    path = os.path.join(out_dir, f"cesm_extinction_{tag}.nc")
    ds.to_netcdf(path)
    log.info("  CESM extinction → %s", path)


def _write_summary(lines: list[str], out_dir: str, filename: str) -> None:
    text = "\n".join(lines)
    print(text)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        f.write(text + "\n")
    log.info("  Summary → %s", path)


# ── per-month worker ───────────────────────────────────────────────────────────

def process_month(
    date: str,
    bg_path: str,
    inj_path: Optional[str],
    out_root: str,
) -> str:
    """
    Run the ALI simulator for one calendar month.  Called once per month,
    either directly (serial) or from a worker process (parallel).

    Returns a short status string for progress reporting.
    """
    try:
        # Each h0 file is a single time step; always use index 0
        TIME_IDX = 0

        # Observation timestamp: use the 15th of the month as a representative
        # mid-month time for the geometry calculation
        obs_time = f"{date}-15T12:00:00Z"
        sim_input = _build_sim_input(obs_time)
        if SIMULATOR == "full":
            from hawcsimulator.ali.configurations.full_inst import ALIPhase0Simulator
            sim_obj = ALIPhase0Simulator()
        else:
            sim_obj = IdealALISimulator()

        # ── background ──────────────────────────────────────────────────────
        bg_out = os.path.join(out_root, "background", date)
        os.makedirs(bg_out, exist_ok=True)

        log.info("[%s] Loading background: %s", date, bg_path)
        waccm_bg    = WACCMAtmosphere(bg_path, alt_grid_km=ALT_GRID_M / 1e3)
        profiles_bg = waccm_bg.get_column_profiles(TANGENT_LAT, TANGENT_LON,
                                                    TIME_IDX)

        log.info("[%s] Running background simulation...", date)
        data_bg = sim_obj.run(
            ["l2", "sk2_atmosphere"],
            {**sim_input,
             "constituents": build_waccm_constituents(profiles_bg, ALT_GRID_M)},
        )
        log.info("[%s] Background converged in %s iterations (cost %.4f)",
                 date,
                 data_bg["l2"]["num_iterations"].values,
                 data_bg["l2"]["cost"].values)

        data_bg["l2"].to_netcdf(os.path.join(bg_out, "l2_background.nc"))
        _save_cesm_extinction(profiles_bg, ALT_GRID_M, bg_out, "background")

        burden_bg = waccm_bg.sulfate_column_burden(TANGENT_LAT, TANGENT_LON,
                                                    TIME_IDX)

        # ── injection (optional) ────────────────────────────────────────────
        data_inj   = None
        waccm_inj  = None
        burden_inj = None

        if inj_path is not None:
            inj_out = os.path.join(out_root, "injection", date)
            os.makedirs(inj_out, exist_ok=True)

            log.info("[%s] Loading injection: %s", date, inj_path)
            waccm_inj    = WACCMAtmosphere(inj_path, alt_grid_km=ALT_GRID_M / 1e3)
            profiles_inj = waccm_inj.get_column_profiles(TANGENT_LAT, TANGENT_LON,
                                                          TIME_IDX)

            log.info("[%s] Running injection simulation...", date)
            data_inj = sim_obj.run(
                ["l2", "sk2_atmosphere"],
                {**sim_input,
                 "constituents": build_waccm_constituents(profiles_inj,
                                                          ALT_GRID_M)},
            )
            log.info("[%s] Injection converged in %s iterations (cost %.4f)",
                     date,
                     data_inj["l2"]["num_iterations"].values,
                     data_inj["l2"]["cost"].values)

            data_inj["l2"].to_netcdf(os.path.join(inj_out, "l2_injection.nc"))
            _save_cesm_extinction(profiles_inj, ALT_GRID_M, inj_out, "injection")
            burden_inj = waccm_inj.sulfate_column_burden(TANGENT_LAT, TANGENT_LON,
                                                          TIME_IDX)

        # ── per-month summary ────────────────────────────────────────────────
        lines = [
            f"Month:             {date}",
            f"TANGENT_LAT:       {TANGENT_LAT}",
            f"TANGENT_LON:       {TANGENT_LON}",
            "",
            "Background stratospheric sulfate (15–35 km):",
        ]
        for k, v in burden_bg.items():
            lines.append(f"  {k:25s}: {v}" if isinstance(v, str)
                         else f"  {k:25s}: {v:.4g}")

        if data_inj is not None and burden_inj is not None:
            ext_bg  = data_bg["l2"]["stratospheric_aerosol_extinction_per_m"]
            ext_inj = data_inj["l2"]["stratospheric_aerosol_extinction_per_m"]
            r_bg    = data_bg["l2"]["stratospheric_aerosol_median_radius"]
            r_inj   = data_inj["l2"]["stratospheric_aerosol_median_radius"]
            strat   = ext_bg.altitude.values > 15000

            peak_ext = float((ext_inj - ext_bg).values[strat].max())
            peak_r   = float((r_inj   - r_bg).values[strat].max())
            d_burden = burden_inj["burden_mg_m2"] - burden_bg["burden_mg_m2"]

            lines += [
                "",
                "Injection stratospheric sulfate (15–35 km):",
            ]
            for k, v in burden_inj.items():
                lines.append(f"  {k:25s}: {v}" if isinstance(v, str)
                             else f"  {k:25s}: {v:.4g}")
            lines += [
                "",
                f"Peak extinction anomaly (>15 km):  {peak_ext:.3e} m⁻¹",
                f"Peak radius anomaly (>15 km):      {peak_r:.1f} nm",
                f"Δ SO₄ burden:                      {d_burden:.3f} mg m⁻²",
            ]

            diff_out = os.path.join(out_root, "diff", date)
            os.makedirs(diff_out, exist_ok=True)
            _write_summary(lines, diff_out, "summary_diff.txt")
        else:
            _write_summary(lines, bg_out, "summary.txt")

        return f"OK   {date}"

    except Exception:
        # Capture full traceback so it survives the process boundary
        tb = traceback.format_exc()
        log.error("[%s] FAILED:\n%s", date, tb)
        return f"FAIL {date}"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── collect files ─────────────────────────────────────────────────────────
    bg_files = _collect_files(WACCM_BACKGROUND_DIR, H0_PATTERN, MONTH_FILTER)

    inj_files: dict[str, str] = {}
    if WACCM_INJECTION_DIR is not None:
        inj_files = _collect_files(WACCM_INJECTION_DIR, H0_PATTERN, MONTH_FILTER)

    # Only process months present in both cases when injection is active;
    # months only in the background are still processed (injection skipped).
    all_dates = sorted(bg_files.keys())
    log.info("Processing %d month(s): %s … %s",
             len(all_dates), all_dates[0], all_dates[-1])

    # Build the work list
    jobs = [
        (date, bg_files[date], inj_files.get(date), OUT_DIR)
        for date in all_dates
    ]

    # ── dispatch ──────────────────────────────────────────────────────────────
    results = []
    if N_WORKERS <= 1:
        # Serial — easy to debug
        for job in jobs:
            results.append(process_month(*job))
    else:
        workers = min(N_WORKERS, len(jobs))
        log.info("Dispatching to %d worker processes...", workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_month, *job): job[0]
                       for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                log.info("Completed: %s", result)

    # ── final report ──────────────────────────────────────────────────────────
    ok   = [r for r in results if r.startswith("OK")]
    fail = [r for r in results if r.startswith("FAIL")]
    log.info("\n── Run complete ──────────────────────────────")
    log.info("  Succeeded: %d / %d", len(ok), len(results))
    if fail:
        log.warning("  Failed months:")
        for f in fail:
            log.warning("    %s", f)
    log.info("  Output root: %s", OUT_DIR)


if __name__ == "__main__":
    main()
