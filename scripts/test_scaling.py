#!/usr/bin/env python
"""
test_scaling.py
================
Scaling checkpoint before the full 182-day run: processes a handful of
days across all available cases (background + all configured injection
cases), dispatched via the real ProcessPoolExecutor path with multiple
workers — exercising the calibration-cache concurrency fix and the
per-worker simulator reuse under actual parallelism for the first time.

Mirrors main()'s job-building logic but restricted to N_TEST_DAYS days,
so config.toml doesn't need to be edited to test at small scale.

Run from the scripts/ directory:
    python test_scaling.py
"""
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import run_orbit_daily as rod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- test scale knobs ---
N_TEST_DAYS = 5
N_TEST_WORKERS = 2


def main():
    log.info("Loading orbit files...")
    orbit_files = rod.load_orbit_files()
    orbit_day_idx = rod.build_orbit_day_index(orbit_files)
    n_orbit_days = max(orbit_day_idx.keys()) + 1
    log.info("Orbit pattern spans %d days", n_orbit_days)

    log.info("Building h2 file indices for all cases...")
    case_labels = {"background": rod.BACKGROUND_CASE}
    for c in rod.INJECTION_CASES:
        m = re.match(r"(sai_[\d.]+Tg)", c)
        label = m.group(1) if m else c
        case_labels[label] = c

    h2_indices = {}
    for label, case in case_labels.items():
        h2_indices[label] = rod.build_h2_index(case)

    bg_dates = sorted(h2_indices["background"].keys())
    test_dates = bg_dates[:N_TEST_DAYS]
    log.info("Testing %d days: %s", len(test_dates), test_dates)

    # build job list exactly like main(), restricted to test_dates
    jobs = []
    for i, date_str in enumerate(bg_dates):
        if date_str not in test_dates:
            continue
        orbit_day = i % n_orbit_days
        if orbit_day not in orbit_day_idx:
            log.warning("No orbit files for orbit day %d, skipping %s", orbit_day, date_str)
            continue

        sim_date = pd.Timestamp(date_str)
        obs = rod.extract_observations(
            orbit_day_idx[orbit_day], sim_date, rod.OBS_CADENCE_S, rod.CENTER_PIXEL
        )
        if not obs:
            log.warning("No observations for %s, skipping", date_str)
            continue

        h2_for_day = {}
        for label in case_labels:
            if date_str in h2_indices[label]:
                h2_for_day[label] = h2_indices[label][date_str]
            else:
                log.warning("Missing h2 file for case '%s' on %s", label, date_str)

        if not h2_for_day:
            continue

        jobs.append((date_str, obs, h2_for_day, rod.OUT_DIR))

    log.info("Built %d jobs, %d cases each: %s",
              len(jobs), len(case_labels), list(case_labels.keys()))

    # pre-warm calibration in main process before dispatching (idempotent
    # now, but still the right order of operations)
    log.info("Pre-warming calibration database...")
    rod._cal_mod.calibration_database("ideal_spectrograph", "v1")
    log.info("Calibration database ready.")

    # pre-warm mode-specific Mie databases (accumulation/coarse extinction),
    # matching what main() does — without this, workers would race to
    # build these on first concurrent use inside process_day()
    log.info("Pre-warming mode-specific Mie databases...")
    from cesm_hawc.constituents import warm_mode_databases
    warm_mode_databases()
    log.info("Mie databases ready.")

    # dispatch via real ProcessPoolExecutor, matching main()'s actual
    # multi-worker code path
    log.info("Dispatching %d jobs to %d worker processes...", len(jobs), N_TEST_WORKERS)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=N_TEST_WORKERS) as pool:
        futures = {pool.submit(rod.process_day, *job): job[0] for job in jobs}
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
    main()