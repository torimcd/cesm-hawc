#!/usr/bin/env python
"""
test_scaling.py
================
Scaling checkpoint before a full production run: processes a handful of
days across all available cases (background + all configured perturbation
cases), dispatched via the real ProcessPoolExecutor path with multiple
workers — exercising the calibration-cache concurrency fix and per-day
simulator construction under actual parallelism for the first time.

Mirrors cesm_hawc.cli's orbit-track real_files job-building logic but
restricted to N_TEST_DAYS days, so config.toml doesn't need to be edited to
test at small scale. Requires config.toml with an [orbit] section using
track_source = "real_files". Run from the repo root:

    python scripts/test_scaling.py --config config.toml
"""
import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from cesm_hawc import file_index, orbit_files
from cesm_hawc.calibration import warm_calibration_database
from cesm_hawc.cli import _case_labels, _run_orbit_daily_case_day
from cesm_hawc.config import load_config
from cesm_hawc.constituents import warm_mode_databases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- test scale knobs ---
N_TEST_DAYS = 5
N_TEST_WORKERS = 2


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    if cfg.orbit is None or cfg.orbit.track_source != "real_files":
        raise SystemExit(
            "config.toml needs an [orbit] section with track_source = "
            "\"real_files\" for this script."
        )
    o, ins = cfg.orbit, cfg.instrument
    epoch = pd.Timestamp(o.orbit_epoch)
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)

    log.info("Loading orbit files...")
    orbit_paths = orbit_files.load_orbit_files(o.orbit_dir, o.orbit_pattern)
    cache_path = os.path.join(o.out_dir, ".orbit_day_index_cache.json")
    orbit_day_idx = orbit_files.build_orbit_day_index(orbit_paths, epoch, cache_path=cache_path)
    n_orbit_days = max(orbit_day_idx.keys()) + 1
    log.info("Orbit pattern spans %d days", n_orbit_days)

    log.info("Building h2 file indices for all cases...")
    case_labels = _case_labels(o.background_case, o.injection_cases)
    h2_indices = {
        label: file_index.index_by_date(os.path.join(o.waccm_data_dir, case, "atm", "hist"), o.h2_pattern)
        for label, case in case_labels.items()
    }

    bg_dates = sorted(h2_indices["background"].keys())
    test_dates = bg_dates[:N_TEST_DAYS]
    log.info("Testing %d days: %s", len(test_dates), test_dates)

    jobs = []
    for i, date_str in enumerate(bg_dates):
        if date_str not in test_dates:
            continue
        orbit_day = i % n_orbit_days
        if orbit_day not in orbit_day_idx:
            log.warning("No orbit files for orbit day %d, skipping %s", orbit_day, date_str)
            continue

        sim_date = pd.Timestamp(date_str)
        obs = orbit_files.extract_observations(
            orbit_day_idx[orbit_day], sim_date, o.obs_cadence_s, o.center_pixel, epoch
        )
        if not obs:
            log.warning("No observations for %s, skipping", date_str)
            continue

        h2_for_day = {label: h2_indices[label][date_str]
                      for label in case_labels if date_str in h2_indices[label]}
        if not h2_for_day:
            continue

        jobs.append((date_str, obs, h2_for_day, o.out_dir, alt_grid_m, wavelengths_nm, o.run_l2))

    log.info("Built %d jobs, %d cases each: %s",
              len(jobs), len(case_labels), list(case_labels.keys()))

    log.info("Pre-warming calibration database...")
    warm_calibration_database()
    log.info("Calibration database ready.")

    log.info("Pre-warming mode-specific Mie databases...")
    warm_mode_databases()
    log.info("Mie databases ready.")

    log.info("Dispatching %d jobs to %d worker processes...", len(jobs), N_TEST_WORKERS)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=N_TEST_WORKERS) as pool:
        futures = {pool.submit(_run_orbit_daily_case_day, *job): job[0] for job in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            log.info("Completed: %s", result)
    elapsed = time.time() - t0

    ok = [r for r in results if r.startswith("OK")]
    fail = [r for r in results if r.startswith("FAIL")]

    log.info("")
    log.info("-- Scaling test complete --")
    log.info("  Succeeded: %d / %d", len(ok), len(results))
    if fail:
        log.warning("  Failed days:")
        for f in fail:
            log.warning("    %s", f)
    log.info("  Total elapsed: %.1f sec for %d days x %d cases, %d workers",
              elapsed, len(jobs), len(case_labels), N_TEST_WORKERS)
    if jobs:
        log.info("  Approx per-day-per-case: %.1f sec",
                  elapsed / (len(jobs) * len(case_labels)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    main(args.config)
