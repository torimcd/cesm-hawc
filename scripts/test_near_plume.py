#!/usr/bin/env python
"""
find_and_test_near_plume.py
============================
Scans all orbit days for the one whose ground track passes closest to the
SAI injection location (30.6N, 180E), then runs a two-case (background +
injection) comparison on a simulation date well after January — so the
plume has had time to develop — that maps to that orbit day.

The geo scan opens every orbit file once to check tangent-point distance
to the injection location, so the first run will be slow (similar to the
original orbit day index build). Results are cached to disk afterward.

Run from the scripts/ directory:
    python find_and_test_near_plume.py
"""
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import run_orbit_daily as rod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INJECTION_LAT = 30.6
INJECTION_LON = 180.0

_GEO_CACHE_DIR = Path(__file__).parent.parent / ".cache"
_GEO_CACHE_FILE = _GEO_CACHE_DIR / "orbit_day_plume_distance.json"


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points (degrees).
    Handles longitude wraparound (e.g. 179 vs -179) correctly without
    needing to normalize inputs first."""
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_orbit_day_plume_distance(orbit_day_idx: dict) -> dict:
    """
    For each orbit day, find the minimum great-circle distance (km) from
    any center-pixel tangent point in that day's files to the injection
    location. Cached to disk since it requires opening every orbit file.
    """
    all_files = sorted(f for files in orbit_day_idx.values() for f in files)
    fingerprint = rod._orbit_files_fingerprint(all_files)

    if _GEO_CACHE_FILE.exists():
        try:
            with open(_GEO_CACHE_FILE) as f:
                cached = json.load(f)
            if cached.get("fingerprint") == fingerprint:
                log.info("Using cached plume-distance scan (%d orbit days)",
                          len(cached["distances"]))
                return {int(k): v for k, v in cached["distances"].items()}
            else:
                log.info("Orbit file set changed, rebuilding plume-distance scan...")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning("Plume-distance cache unreadable, rebuilding: %s", e)

    log.info("Scanning %d orbit days for closest approach to injection "
              "location (%.1fN, %.1fE) — opens every orbit file, this will "
              "take a while on first run...",
              len(orbit_day_idx), INJECTION_LAT, INJECTION_LON)

    distances: dict = {}
    for i, (orbit_day, files) in enumerate(sorted(orbit_day_idx.items())):
        min_dist = np.inf
        for f in files:
            ds = xr.open_dataset(f)
            lats = ds["latitude"].values[:, 0, rod.CENTER_PIXEL]
            lons = ds["longitude"].values[:, 0, rod.CENTER_PIXEL]
            ds.close()
            d = _haversine_km(lats, lons, INJECTION_LAT, INJECTION_LON)
            min_dist = min(min_dist, float(np.min(d)))
        distances[orbit_day] = min_dist
        if (i + 1) % 10 == 0:
            log.info("  scanned %d/%d orbit days...", i + 1, len(orbit_day_idx))

    try:
        _GEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_GEO_CACHE_FILE, "w") as f:
            json.dump({"fingerprint": fingerprint, "distances": distances}, f)
        log.info("Cached plume-distance scan to %s", _GEO_CACHE_FILE)
    except OSError as e:
        log.warning("Could not write plume-distance cache: %s", e)

    return distances


def pick_simulation_date(orbit_day: int, bg_dates: list, n_orbit_days: int,
                          min_index: int = 31) -> str:
    """
    Pick the earliest simulation date at or after min_index (default: day
    index 31, i.e. after January) that maps to the given orbit_day via
    i % n_orbit_days == orbit_day.
    """
    for i in range(len(bg_dates)):
        if i >= min_index and i % n_orbit_days == orbit_day:
            return bg_dates[i]
    raise ValueError(f"No simulation date >= index {min_index} maps to orbit day {orbit_day}")


def main():
    log.info("Loading orbit files...")
    orbit_files = rod.load_orbit_files()
    orbit_day_idx = rod.build_orbit_day_index(orbit_files)
    n_orbit_days = max(orbit_day_idx.keys()) + 1

    distances = build_orbit_day_plume_distance(orbit_day_idx)

    ranked = sorted(distances.items(), key=lambda kv: kv[1])
    log.info("Closest 5 orbit days to injection location:")
    for orbit_day, dist_km in ranked[:5]:
        log.info("  orbit day %d: %.1f km", orbit_day, dist_km)

    best_orbit_day, best_dist = ranked[0]
    log.info("Selected orbit day %d (%.1f km from injection point)", best_orbit_day, best_dist)

    log.info("Loading h2 index for background case...")
    h2_bg = rod.build_h2_index(rod.BACKGROUND_CASE)
    bg_dates = sorted(h2_bg.keys())

    first_injection_case = rod.INJECTION_CASES[0]
    log.info("Loading h2 index for injection case: %s", first_injection_case)
    h2_inj = rod.build_h2_index(first_injection_case)

    target_date = pick_simulation_date(best_orbit_day, bg_dates, n_orbit_days, min_index=31)
    log.info("Selected simulation date %s (orbit day %d, well after January)",
              target_date, best_orbit_day)

    if target_date not in h2_inj:
        raise SystemExit(f"No injection h2 file for {target_date}")

    sim_date = pd.Timestamp(target_date)
    obs = rod.extract_observations(
        orbit_day_idx[best_orbit_day], sim_date, rod.OBS_CADENCE_S, rod.CENTER_PIXEL
    )
    log.info("Extracted %d observations", len(obs))

    rod._cal_mod.calibration_database("ideal_spectrograph", "v1")
    log.info("Calibration database checked/warmed.")

    m = re.match(r"(sai_[\d.]+Tg)", first_injection_case)
    inj_label = m.group(1) if m else first_injection_case

    h2_for_day = {
        "background": h2_bg[target_date],
        inj_label:    h2_inj[target_date],
    }
    log.info("Running process_day for %s with cases: %s", target_date, list(h2_for_day.keys()))

    t0 = time.time()
    result = rod.process_day(target_date, obs, h2_for_day, rod.OUT_DIR)
    elapsed = time.time() - t0

    log.info("Result: %s", result)
    log.info("Elapsed: %.1f sec for %d case(s)", elapsed, len(h2_for_day))

    if result.startswith("OK"):
        log.info("SUCCESS.")
    else:
        log.error("FAILED — see traceback above.")


if __name__ == "__main__":
    main()