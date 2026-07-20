#!/usr/bin/env python
"""
test_one_day_multicase.py
==========================
Single-day test with TWO cases (background + first injection scenario),
to exercise process_day()'s multi-case loop before committing to the full
182-day, multi-case run. Times the run so we can estimate per-case cost.

Run from the scripts/ directory:
    python test_one_day_multicase.py
"""
import logging
import time
import pandas as pd

import run_orbit_daily as rod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

if not rod.INJECTION_CASES:
    raise SystemExit(
        "No injection_cases configured in config.toml — nothing to test "
        "beyond the background-only case we've already verified."
    )

log.info("Loading orbit files...")
orbit_files = rod.load_orbit_files()
orbit_day_idx = rod.build_orbit_day_index(orbit_files)
first_orbit_day = sorted(orbit_day_idx.keys())[0]
log.info("Using orbit day %d (%d files)", first_orbit_day, len(orbit_day_idx[first_orbit_day]))

# background + first injection case only, to keep this test fast
log.info("Loading h2 index for background case...")
h2_bg = rod.build_h2_index(rod.BACKGROUND_CASE)

first_injection_case = rod.INJECTION_CASES[0]
log.info("Loading h2 index for injection case: %s", first_injection_case)
h2_inj = rod.build_h2_index(first_injection_case)

first_date_str = sorted(h2_bg.keys())[0]
sim_date = pd.Timestamp(first_date_str)
log.info("Using simulation date %s", first_date_str)

if first_date_str not in h2_inj:
    raise SystemExit(f"No h2 file for injection case on {first_date_str}")

obs = rod.extract_observations(
    orbit_day_idx[first_orbit_day], sim_date, rod.OBS_CADENCE_S, rod.CENTER_PIXEL
)
log.info("Extracted %d observations", len(obs))

rod._cal_mod.calibration_database("ideal_spectrograph", "v1")
log.info("Calibration database checked/warmed.")

# use a cleaner label matching what main() would generate, e.g. "sai_1.0Tg"
import re
m = re.match(r"(sai_[\d.]+Tg)", first_injection_case)
inj_label = m.group(1) if m else first_injection_case

h2_for_day = {
    "background": h2_bg[first_date_str],
    inj_label:    h2_inj[first_date_str],
}
log.info("Running process_day for %s with cases: %s", first_date_str, list(h2_for_day.keys()))

t0 = time.time()
result = rod.process_day(first_date_str, obs, h2_for_day, rod.OUT_DIR)
elapsed = time.time() - t0

log.info("Result: %s", result)
log.info("Elapsed: %.1f sec for %d case(s)", elapsed, len(h2_for_day))
log.info("Approx per-case time: %.1f sec", elapsed / len(h2_for_day))

if result.startswith("OK"):
    log.info("SUCCESS — multi-case single day completed cleanly.")
else:
    log.error("FAILED — see traceback above for details.")