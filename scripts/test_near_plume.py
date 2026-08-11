#!/usr/bin/env python
"""
test_near_plume.py
===================
Scans all orbit days for the one whose ground track passes closest to the
SAI injection location (30.6N, 180E), then runs a two-case (background +
injection) comparison on a simulation date well after January — so the
plume has had time to develop — that maps to that orbit day.

The geo scan opens every orbit file once to check tangent-point distance
to the injection location, so the first run will be slow (similar to the
orbit day index build in cesm_hawc.orbit_files). Results are cached to disk
afterward.

Requires config.toml with an [orbit] section using track_source =
"real_files". Run from the repo root:

    python scripts/test_near_plume.py --config config.toml
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cesm_hawc import file_index, orbit_files
from cesm_hawc.calibration import warm_calibration_database
from cesm_hawc.cli import _case_labels, _run_orbit_daily_case_day
from cesm_hawc.config import load_config

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


def build_orbit_day_plume_distance(orbit_day_idx: dict, center_pixel: int) -> dict:
    """
    For each orbit day, find the minimum great-circle distance (km) from
    any center-pixel tangent point in that day's files to the injection
    location. Cached to disk since it requires opening every orbit file.
    """
    all_files = sorted(f for files in orbit_day_idx.values() for f in files)
    fingerprint = orbit_files._orbit_files_fingerprint(all_files)

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
            ds = xr.open_dataset(f, decode_times=False)
            lats = ds["latitude"].values[:, 0, center_pixel]
            lons = ds["longitude"].values[:, 0, center_pixel]
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

    distances = build_orbit_day_plume_distance(orbit_day_idx, o.center_pixel)

    ranked = sorted(distances.items(), key=lambda kv: kv[1])
    log.info("Closest 5 orbit days to injection location:")
    for orbit_day, dist_km in ranked[:5]:
        log.info("  orbit day %d: %.1f km", orbit_day, dist_km)

    best_orbit_day, best_dist = ranked[0]
    log.info("Selected orbit day %d (%.1f km from injection point)", best_orbit_day, best_dist)

    case_labels = _case_labels(o.background_case, o.injection_cases)
    log.info("Loading h2 index for background case...")
    h2_bg = file_index.index_by_date(os.path.join(o.waccm_data_dir, o.background_case, "atm", "hist"),
                                      o.h2_pattern)
    bg_dates = sorted(h2_bg.keys())

    first_injection_case = o.injection_cases[0]
    inj_label = next(label for label, case in case_labels.items() if case == first_injection_case)
    log.info("Loading h2 index for injection case: %s", first_injection_case)
    h2_inj = file_index.index_by_date(os.path.join(o.waccm_data_dir, first_injection_case, "atm", "hist"),
                                       o.h2_pattern)

    target_date = pick_simulation_date(best_orbit_day, bg_dates, n_orbit_days, min_index=31)
    log.info("Selected simulation date %s (orbit day %d, well after January)",
              target_date, best_orbit_day)

    if target_date not in h2_inj:
        raise SystemExit(f"No injection h2 file for {target_date}")

    sim_date = pd.Timestamp(target_date)
    obs = orbit_files.extract_observations(
        orbit_day_idx[best_orbit_day], sim_date, o.obs_cadence_s, o.center_pixel, epoch
    )
    log.info("Extracted %d observations", len(obs))

    warm_calibration_database()
    log.info("Calibration database checked/warmed.")

    h2_for_day = {"background": h2_bg[target_date], inj_label: h2_inj[target_date]}
    log.info("Running one day for %s with cases: %s", target_date, list(h2_for_day.keys()))

    t0 = time.time()
    result = _run_orbit_daily_case_day(
        target_date, obs, h2_for_day, o.out_dir, alt_grid_m, wavelengths_nm, o.run_l2
    )
    elapsed = time.time() - t0

    log.info("Result: %s", result)
    log.info("Elapsed: %.1f sec for %d case(s)", elapsed, len(h2_for_day))

    if result.startswith("OK"):
        log.info("SUCCESS.")
    else:
        log.error("FAILED — see traceback above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    main(args.config)
