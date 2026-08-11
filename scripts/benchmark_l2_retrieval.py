#!/usr/bin/env python
"""
benchmark_l2_retrieval.py
==========================
Benchmark the cost of full ALI L2 retrieval, using REAL orbit geometry and
CESM h2 files pulled through cesm_hawc.orbit_files/file_index, to assess
whether running full L2 on a large production set is realistic, or whether
an averaging-kernel shortcut is needed.

Requires config.toml with an [orbit] section using track_source =
"real_files". Run from the repo root:

    python scripts/benchmark_l2_retrieval.py --config config.toml

front_end_radiance/l1b are byproducts of the full-L2 product list already
(FULL_L2_PRODUCTS below). Each profile is run through
simulator.run(FULL_L2_PRODUCTS, ...) exactly once, wrapped in cProfile.

L2 marginal cost is estimated from the CUMULATIVE time of a single
identified entry-point function, not an own-time keyword sum. Grepping
skretrieval (the actual retrieval package underneath hawcsimulator, via
sasktran2 for RT) showed skretrieval.retrieval.processing.Retrieval.
retrieve() as the top-level orchestrator, with Minimizer subclasses
(Rodgers, SciPyMinimizer, SciPyMinimizerGrad) doing Gauss-Newton /
Levenberg-Marquardt-style iteration underneath it -- each iteration calls
back into the forward RT model (via statevector.propagate_wf) to rebuild
the measurement vector y and jacobian K. That means most of the real cost
lives in RT functions called FROM retrieve(), not in functions literally
named "retrieve" -- an own-time keyword sum would badly undercount.
Cumulative time of the single outermost retrieve() call captures
everything nested under it, RT calls included, without needing to name
every function that runs during iteration.

ASSUMPTIONS TO VERIFY:
  - L2_ENTRY_POINT_CANDIDATES below assumes
    skretrieval.retrieval.processing.Retrieval.retrieve() is the function
    hawcsimulator actually calls to produce the "l2" product. Run
    inspect_profile_functions() on one profile first -- it prints every
    profiled function named "retrieve" (via list_retrieve_functions) and
    reports which candidate matched, so you can confirm this before
    trusting the full benchmark. If none match, update
    L2_ENTRY_POINT_CANDIDATES from that printed list.
  - Convergence / function-evaluation diagnostics are parsed from scipy's
    verbose=2 stdout output (captured via redirect_stdout during the
    profiled call) rather than guessed at from data["l2"].attrs -- see
    cesm_hawc.convergence.parse_scipy_convergence(). This was confirmed
    against real output: a background case in one run genuinely failed to
    converge ("The maximum number of function evaluations is exceeded"),
    which an attrs-based placeholder silently reported as unknown (None)
    rather than surfacing as a real non-convergence.
"""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import io
import logging
import os
import pstats
import time
import traceback

import numpy as np
import pandas as pd

from cesm_hawc import file_index, orbit_files
from cesm_hawc.cli import _case_labels
from cesm_hawc.config import load_config
from cesm_hawc.constituents import build_waccm_constituents
from cesm_hawc.convergence import extract_l2_native_diagnostics, parse_scipy_convergence
from cesm_hawc.noise import default_noise_model
from cesm_hawc.waccm import WACCMAtmosphere

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FULL_L2_PRODUCTS = ["l2", "sk2_atmosphere", "front_end_radiance", "l1b"]

# Identified from grepping skretrieval directly (see conversation notes):
# skretrieval.retrieval.processing.Retrieval.retrieve() is the top-level
# orchestrator. Underneath it, Minimizer subclasses (Rodgers,
# SciPyMinimizer, SciPyMinimizerGrad) iterate -- each iteration calls back
# into the forward RT model (via statevector.propagate_wf) to rebuild the
# measurement vector y and jacobian K, then does a Gauss-Newton/Levenberg-
# Marquardt-style update. CUMULATIVE time of the single outermost
# retrieve() call is used instead of an own-time keyword sum, since
# cumulative time already includes every nested forward-model call made
# during iteration.
L2_ENTRY_POINT_CANDIDATES = [
    ("skretrieval/retrieval/processing.py", "retrieve"),   # primary: top-level orchestrator
    ("skretrieval/retrieval/rodgers.py", "retrieve"),       # fallback: Rodgers minimizer directly
    ("skretrieval/retrieval/scipy.py", "retrieve"),         # fallback: SciPy minimizer directly
]

# ---------------------------------------------------------------------------
# Config-derived globals, populated by _load_globals() in main()
# ---------------------------------------------------------------------------

BACKGROUND_CASE = None
INJECTION_CASES = None
RUN_START_DATE = None
RUN_END_DATE = None
ALT_GRID_M = None
ALI_WAVELENGTHS = None
NOISE_MODEL = None
CENTER_PIXEL = None
OBS_CADENCE_S = None
ORBIT_EPOCH = None
N_WORKERS = None
WACCM_DATA_DIR = None
H2_PATTERN = None
ORBIT_DIR = None
ORBIT_PATTERN = None
OUT_DIR = None

_SIMULATOR = None


def _load_globals(config_path: str) -> None:
    global BACKGROUND_CASE, INJECTION_CASES, RUN_START_DATE, RUN_END_DATE
    global ALT_GRID_M, ALI_WAVELENGTHS, NOISE_MODEL, CENTER_PIXEL, OBS_CADENCE_S
    global ORBIT_EPOCH, N_WORKERS, WACCM_DATA_DIR, H2_PATTERN, ORBIT_DIR, ORBIT_PATTERN, OUT_DIR

    cfg = load_config(config_path)
    if cfg.orbit is None or cfg.orbit.track_source != "real_files":
        raise SystemExit(
            "config.toml needs an [orbit] section with track_source = "
            "\"real_files\" for this script."
        )
    o, ins = cfg.orbit, cfg.instrument
    BACKGROUND_CASE = o.background_case
    INJECTION_CASES = o.injection_cases
    RUN_START_DATE = o.run_start_date
    RUN_END_DATE = o.run_end_date
    ALT_GRID_M = ins.altitude_grid_m()
    ALI_WAVELENGTHS = np.array(ins.wavelengths_nm)
    NOISE_MODEL = default_noise_model()
    CENTER_PIXEL = o.center_pixel
    OBS_CADENCE_S = o.obs_cadence_s
    ORBIT_EPOCH = pd.Timestamp(o.orbit_epoch)
    N_WORKERS = o.n_workers
    WACCM_DATA_DIR = o.waccm_data_dir
    H2_PATTERN = o.h2_pattern
    ORBIT_DIR = o.orbit_dir
    ORBIT_PATTERN = o.orbit_pattern
    OUT_DIR = o.out_dir


def _get_simulator():
    """Lazily construct one IdealALISimulator and reuse it across every
    benchmarked observation in this process (avoids redundant calibration-
    database access on every single call)."""
    global _SIMULATOR
    if _SIMULATOR is None:
        from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator
        _SIMULATOR = IdealALISimulator()
    return _SIMULATOR


def _h2_index(case: str) -> dict:
    return file_index.index_by_date(os.path.join(WACCM_DATA_DIR, case, "atm", "hist"), H2_PATTERN)


# ---------------------------------------------------------------------------
# Real-data sampling
# ---------------------------------------------------------------------------

def build_case_labels() -> dict[str, str]:
    return _case_labels(BACKGROUND_CASE, INJECTION_CASES)


def sample_observations(n_days: int = 3, n_obs_per_day: int = 8, seed: int = 42) -> tuple[list[dict], dict]:
    """
    Pull real DAYTIME observations spread across the full simulation
    period and across each sampled day's orbit arc, using
    cesm_hawc.orbit_files' orbit-day mapping and extraction logic.

    Sample dates are chosen via a seeded random draw from n_days roughly
    equal bins across the full date range, not evenly-spaced linspace --
    for a Jan-to-Jul date range, linspace(0, len-1, n_days) lands on the
    exact endpoints, which for this dataset meant both solstices: an
    unrepresentative pair for solar-geometry variation, and confirmed by
    an earlier run where both sampled dates had unusually narrow daytime
    windows.

    Within each sampled day, EVERY candidate observation is cheaply
    checked for daytime via a forward-only (no L2) simulator call using
    the background case's atmosphere -- the night-side SZA check is
    purely geometric and doesn't depend on which case's atmosphere is
    used, so this avoids wasting expensive full-L2 calls (100-600s each,
    per observed benchmark data) on observations that would just be
    skipped anyway. Only confirmed-daytime observations are returned,
    evenly spaced across the day's daytime arc.

    Returns (samples, daytime_stats):
      samples: list of dicts {date_str, orbit_day, obs, h2_for_day}, where
        obs is a single observation dict from
        cesm_hawc.orbit_files.extract_observations() (real lat/lon/time/
        observer geometry) and h2_for_day maps case label -> h2 file path
        for that date, one entry per sampled observation.
      daytime_stats: {n_daytime_candidates, n_total_candidates,
        observed_daytime_fraction, per_day} -- the REAL daytime fraction
        measured before the pre-filter discards night-side candidates.
        Use this (not anything derived from the returned samples/benchmark
        results, which are ALL daytime by construction) for extrapolating
        to the full simulation period via estimate_total_profile_count().
    """
    orbit_paths = orbit_files.load_orbit_files(ORBIT_DIR, ORBIT_PATTERN)
    cache_path = os.path.join(OUT_DIR, ".orbit_day_index_cache.json")
    orbit_day_idx = orbit_files.build_orbit_day_index(orbit_paths, ORBIT_EPOCH, cache_path=cache_path)
    n_orbit_days = max(orbit_day_idx.keys()) + 1

    case_labels = build_case_labels()
    h2_indices = {label: _h2_index(case) for label, case in case_labels.items()}

    bg_dates = sorted(h2_indices["background"].keys())
    if RUN_START_DATE or RUN_END_DATE:
        bg_dates = [
            d for d in bg_dates
            if (not RUN_START_DATE or d >= RUN_START_DATE)
            and (not RUN_END_DATE or d <= RUN_END_DATE)
        ]

    # stratified random sample dates: split the date range into n_days
    # roughly-equal bins and draw one random date per bin, so sampling
    # still spans the full period but isn't pinned to exact endpoints.
    rng = np.random.default_rng(seed)
    bin_edges = np.linspace(0, len(bg_dates), n_days + 1, dtype=int)
    sample_idx = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        hi = max(hi, lo + 1)
        sample_idx.append(int(rng.integers(lo, min(hi, len(bg_dates)))))

    simulator = _get_simulator()

    samples = []
    daytime_stats = {"n_daytime_candidates": 0, "n_total_candidates": 0, "per_day": []}
    for i in sample_idx:
        date_str = bg_dates[i]
        orbit_day = i % n_orbit_days
        if orbit_day not in orbit_day_idx:
            continue

        sim_date = pd.Timestamp(date_str)
        day_obs = orbit_files.extract_observations(
            orbit_day_idx[orbit_day], sim_date, OBS_CADENCE_S, CENTER_PIXEL, ORBIT_EPOCH
        )
        if not day_obs:
            continue

        h2_for_day = {}
        for label in case_labels:
            if date_str in h2_indices[label]:
                h2_for_day[label] = h2_indices[label][date_str]
        if "background" not in h2_for_day:
            continue  # need it for the cheap daytime check below

        # cheap geometric daytime check against every candidate observation
        # in the day, using the background atmosphere only -- the SZA
        # check doesn't depend on which case's atmosphere is used, so one
        # atmosphere suffices to classify all of them.
        waccm = WACCMAtmosphere(h2_for_day["background"], alt_grid_km=ALT_GRID_M / 1e3)
        daytime_obs = []
        for obs in day_obs:
            profiles = waccm.get_column_profiles(obs["lat"], obs["lon"], time_index=0)
            constituents = build_waccm_constituents(profiles, ALT_GRID_M)
            sim_input = {**_build_sim_input(obs), "constituents": constituents}
            try:
                simulator.run(["front_end_radiance", "l1b"], sim_input)
                daytime_obs.append(obs)
            except ValueError as e:
                if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                    continue
                raise

        log.info("%s: %d/%d observations are daytime", date_str, len(daytime_obs), len(day_obs))
        daytime_stats["n_daytime_candidates"] += len(daytime_obs)
        daytime_stats["n_total_candidates"] += len(day_obs)
        daytime_stats["per_day"].append({
            "date_str": date_str, "n_daytime": len(daytime_obs), "n_total": len(day_obs),
        })
        if not daytime_obs:
            continue

        # evenly spaced subsample among CONFIRMED daytime observations --
        # spans the daytime arc rather than clustering
        obs_idx = np.linspace(0, len(daytime_obs) - 1, min(n_obs_per_day, len(daytime_obs)), dtype=int)
        chosen_obs = [daytime_obs[j] for j in np.unique(obs_idx)]

        for obs in chosen_obs:
            samples.append({
                "date_str": date_str,
                "orbit_day": int(orbit_day),
                "obs": obs,
                "h2_for_day": h2_for_day,
            })

    daytime_stats["observed_daytime_fraction"] = (
        daytime_stats["n_daytime_candidates"] / daytime_stats["n_total_candidates"]
        if daytime_stats["n_total_candidates"] > 0 else None
    )
    return samples, daytime_stats


def _build_sim_input(obs: dict) -> dict:
    """Matches cesm_hawc.cli's orbit-track real_files sim_input construction
    exactly -- SZA/SAA are NOT passed explicitly; they're computed
    automatically from time + observer position, same as production."""
    sim_input = {
        "tangent_latitude": float(obs["lat"]),
        "tangent_longitude": float(obs["lon"]),
        "observer_latitude": obs["observer_lat"],
        "observer_longitude": obs["observer_lon"],
        "observer_altitude": obs["observer_alt"],
        "altitude_grid": ALT_GRID_M,
        "polarization_states": ["I", "dolp"],
        "sample_wavelengths": ALI_WAVELENGTHS,
        "time": obs["time"],
        "l1b_cfg": {"noise_model": NOISE_MODEL},
    }
    return sim_input


def _find_l2_entry_cumulative_time(stats: pstats.Stats) -> tuple[float | None, str | None]:
    """Look up cumulative time (ct) of the identified L2 retrieval entry
    point -- checked in order against L2_ENTRY_POINT_CANDIDATES. Returns
    (cumulative_seconds, which_candidate_matched), or (None, None) if none
    of the candidates appear in the call graph (in which case fall back to
    list_retrieve_functions() to see what's actually there)."""
    for file_substr, funcname in L2_ENTRY_POINT_CANDIDATES:
        for (filename, _lineno, fname), (_cc, _nc, _tt, ct, _callers) in stats.stats.items():
            if file_substr in filename and fname == funcname:
                return ct, f"{file_substr}:{funcname}"
    return None, None


def list_retrieve_functions(stats: pstats.Stats) -> None:
    """Print every profiled function literally named 'retrieve', with call
    count and cumulative time. Use this if _find_l2_entry_cumulative_time
    returns None -- it means none of L2_ENTRY_POINT_CANDIDATES matched,
    and this shows what's actually in the call graph so you can update the
    candidate list."""
    print(f"{'file':<70} {'ncalls':>8} {'cumtime':>10}")
    for (filename, _lineno, fname), (cc, _nc, _tt, ct, _callers) in stats.stats.items():
        if fname == "retrieve":
            print(f"{filename:<70} {cc:>8} {ct:>10.4f}")


def _save_l2_output(data_full: dict, out_path_base: str) -> dict:
    """Save the actual L2 retrieval output to disk, not just timing/
    convergence metadata.

    Only saves data_full['l2'] -- confirmed via inspect_profile_functions()
    to be a clean xarray Dataset, so .to_netcdf() just works.
    data_full['sk2_atmosphere'] is deliberately NOT saved: it wraps a
    compiled sasktran2 extension type, not xarray -- pickling it is likely
    to fail or produce something large and fragile, and it's the forward
    model's internal RT state rather than retrieval output.

    Returns {saved, format, paths (dict, currently just {'l2': path}), error}.
    """
    result = {"saved": False, "format": None, "paths": {}, "error": None}

    obj = data_full.get("l2")
    if obj is None:
        result["error"] = "data_full['l2'] is None or missing"
        return result

    path_nc = f"{out_path_base}_l2.nc"
    try:
        obj.to_netcdf(path_nc)
        result["paths"]["l2"] = path_nc
        result["saved"] = True
        result["format"] = "netcdf"
    except Exception as e:
        result["error"] = f"l2: to_netcdf failed: {type(e).__name__}: {e}"

    return result


def _actual_sza(data: dict) -> float | None:
    """Pull the SZA the simulator actually computed for this observation
    (from the l1b dataset), rather than a value we specified -- since
    production doesn't pass SZA explicitly, this is the only way to know
    it for post-hoc stratification of benchmark results."""
    l1b = data.get("l1b", None)
    if l1b is None:
        return None
    try:
        ds = orbit_files.l1b_image_to_dataset(l1b, ALI_WAVELENGTHS)
        return float(np.mean(ds["solar_zenith_angle"].values))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def find_daytime_sample(samples: list[dict]) -> tuple[dict, str]:
    """Find the first (sample, case_label) pair that isn't a night-side
    tangent point, for use with inspect_profile_functions(). Does a cheap
    forward-only check (not the full L2 profile) to avoid wasting time on
    a full retrieval call just to find out it's night-side."""
    simulator = _get_simulator()
    for sample in samples:
        for case_label, h2_path in sample["h2_for_day"].items():
            waccm = WACCMAtmosphere(h2_path, alt_grid_km=ALT_GRID_M / 1e3)
            profiles = waccm.get_column_profiles(
                sample["obs"]["lat"], sample["obs"]["lon"], time_index=0
            )
            constituents = build_waccm_constituents(profiles, ALT_GRID_M)
            sim_input = {**_build_sim_input(sample["obs"]), "constituents": constituents}
            try:
                simulator.run(["front_end_radiance", "l1b"], sim_input)
                return sample, case_label
            except ValueError as e:
                if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                    continue
                raise
    raise RuntimeError("No daytime observation found among samples -- widen n_days/n_obs_per_day")


def _describe_l2_output(l2_obj, max_depth: int = 2, _depth: int = 0) -> None:
    """Print the real structure of whatever data_full['l2'] actually is --
    type, and for common container types (dict, xarray Dataset/DataArray),
    its keys/variables/coords/shape."""
    indent = "  " * _depth
    print(f"{indent}type: {type(l2_obj)}")

    if isinstance(l2_obj, dict):
        print(f"{indent}dict keys: {list(l2_obj.keys())}")
        if _depth < max_depth:
            for k, v in l2_obj.items():
                print(f"{indent}  ['{k}']:")
                _describe_l2_output(v, max_depth, _depth + 2)
        return

    if hasattr(l2_obj, "data_vars"):
        print(f"{indent}xarray-like Dataset")
        print(f"{indent}data_vars: {list(l2_obj.data_vars)}")
        print(f"{indent}coords: {list(l2_obj.coords)}")
        print(f"{indent}dims: {dict(l2_obj.sizes) if hasattr(l2_obj, 'sizes') else l2_obj.dims}")
        return
    if hasattr(l2_obj, "dims") and hasattr(l2_obj, "values"):
        print(f"{indent}xarray-like DataArray, shape={getattr(l2_obj, 'shape', None)}, "
              f"dims={l2_obj.dims}")
        return

    print(f"{indent}dir() (public attrs): "
          f"{[a for a in dir(l2_obj) if not a.startswith('_')]}")


def inspect_profile_functions(sample: dict, case_label: str, simulator, top_n: int = 40) -> pstats.Stats:
    """Run one real observation under cProfile and print the top functions
    by cumulative time, so you can see real function names and confirm
    L2_ENTRY_POINT_CANDIDATES before running the full benchmark. Also
    prints the real structure of data_full['l2'] so a save format can be
    chosen based on what's actually there."""
    h2_path = sample["h2_for_day"][case_label]
    waccm = WACCMAtmosphere(h2_path, alt_grid_km=ALT_GRID_M / 1e3)
    profiles = waccm.get_column_profiles(sample["obs"]["lat"], sample["obs"]["lon"], time_index=0)
    constituents = build_waccm_constituents(profiles, ALT_GRID_M)
    sim_input = {**_build_sim_input(sample["obs"]), "constituents": constituents}

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        data_full = simulator.run(FULL_L2_PRODUCTS, sim_input)
    finally:
        profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(top_n)

    print("\n--- retrieve() functions found in call graph ---")
    list_retrieve_functions(stats)

    ct, entry_point = _find_l2_entry_cumulative_time(stats)
    if entry_point is not None:
        print(f"\nL2 entry point matched: {entry_point}  (cumulative time: {ct:.4f}s)")
    else:
        print(
            "\nNo candidate in L2_ENTRY_POINT_CANDIDATES matched. Check the "
            "'retrieve() functions found' list above and update "
            "L2_ENTRY_POINT_CANDIDATES accordingly."
        )

    print("\n--- structure of data_full['l2'] (the actual retrieval output) ---")
    _describe_l2_output(data_full.get("l2"))
    print("\n--- structure of data_full['sk2_atmosphere'] (retrieved atmospheric state?) ---")
    _describe_l2_output(data_full.get("sk2_atmosphere"))

    return stats


def benchmark_observation(sample: dict, case_label: str, simulator, save_output_dir: str | None = None) -> dict:
    """Run the full-L2 simulator call once for one real (case, observation)
    pair, under cProfile, and split total time into 'retrieval' vs 'other'
    based on the identified L2 entry point's cumulative time.

    If save_output_dir is given, also persists the actual retrieval output
    (data_full['l2']) to disk via _save_l2_output(), not just timing/
    convergence metadata -- see l2_output_* result fields."""
    obs = sample["obs"]
    result = {
        "case_label": case_label,
        "date_str": sample["date_str"],
        "orbit_day": sample["orbit_day"],
        "lat": obs["lat"],
        "lon": obs["lon"],
        "time": obs["time"],
        "actual_sza_deg": None,
        "total_time_s": None,
        "l2_marginal_time_s": None,
        "l2_entry_point": None,
        "non_retrieval_time_s": None,
        "converged": None,
        "termination_reason": None,
        "n_function_evaluations": None,
        "l2_num_iterations": None,
        "l2_final_cost": None,
        "l2_output_saved": False,
        "l2_output_format": None,
        "l2_output_paths": None,
        "l2_output_save_error": None,
        "status": "ok",
        "error": None,
        "error_traceback": None,
    }

    try:
        h2_path = sample["h2_for_day"][case_label]
        waccm = WACCMAtmosphere(h2_path, alt_grid_km=ALT_GRID_M / 1e3)
        profiles = waccm.get_column_profiles(obs["lat"], obs["lon"], time_index=0)
        constituents = build_waccm_constituents(profiles, ALT_GRID_M)
        sim_input = {**_build_sim_input(obs), "constituents": constituents}

        profiler = cProfile.Profile()
        stdout_capture = io.StringIO()
        t0 = time.perf_counter()
        profiler.enable()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                data_full = simulator.run(FULL_L2_PRODUCTS, sim_input)
        finally:
            # Must always disable, even on exception (e.g. night-side SZA
            # ValueError) -- cProfile installs a single global C-level
            # hook, so skipping disable() here leaves it stuck installed
            # and the NEXT profiler.enable() call in the loop fails.
            profiler.disable()
        t1 = time.perf_counter()
        result["total_time_s"] = t1 - t0
        result["actual_sza_deg"] = _actual_sza(data_full)

        stats = pstats.Stats(profiler)
        retrieval_time, entry_point = _find_l2_entry_cumulative_time(stats)
        result["l2_marginal_time_s"] = retrieval_time
        result["l2_entry_point"] = entry_point
        if retrieval_time is not None:
            result["non_retrieval_time_s"] = result["total_time_s"] - retrieval_time

        diag = parse_scipy_convergence(stdout_capture.getvalue())
        result["converged"] = diag["converged"]
        result["termination_reason"] = diag["termination_reason"]
        result["n_function_evaluations"] = diag["n_function_evaluations"]

        native_diag = extract_l2_native_diagnostics(data_full.get("l2"))
        result["l2_num_iterations"] = native_diag["l2_num_iterations"]
        result["l2_final_cost"] = native_diag["l2_final_cost"]

        if save_output_dir is not None:
            # Save for BOTH converged and non-converged cases -- a failed
            # retrieval's output is still scientifically informative (what
            # did it converge TOWARD before hitting max_nfev?).
            os.makedirs(save_output_dir, exist_ok=True)
            safe_time = str(pd.Timestamp(obs["time"])).replace(" ", "T").replace(":", "")
            out_path_base = os.path.join(
                save_output_dir, f"{sample['date_str']}_{case_label}_{safe_time}"
            )
            save_info = _save_l2_output(data_full, out_path_base)
            result["l2_output_saved"] = save_info["saved"]
            result["l2_output_format"] = save_info["format"]
            result["l2_output_paths"] = str(save_info["paths"]) if save_info["paths"] else None
            result["l2_output_save_error"] = save_info["error"]

    except ValueError as e:
        if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
            # night-side tangent point, same skip condition as production
            result["status"] = "skipped_night"
        else:
            result["status"] = "error"
            result["error"] = f"{type(e).__name__}: {e}"
            result["error_traceback"] = traceback.format_exc()
    except Exception as e:  # noqa: BLE001 -- want every failure logged, not fatal
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["error_traceback"] = traceback.format_exc()

    return result


# Must match benchmark_observation()'s result dict keys, in order. Used to
# detect a schema mismatch against an existing out_csv before appending --
# this fixed a real bug where the same output filename got reused across
# script versions during iterative development, silently corrupting the
# CSV with inconsistent column counts across appended chunks.
_RESULT_FIELDNAMES = [
    "case_label", "date_str", "orbit_day", "lat", "lon", "time",
    "actual_sza_deg", "total_time_s", "l2_marginal_time_s", "l2_entry_point",
    "non_retrieval_time_s", "converged", "termination_reason",
    "n_function_evaluations", "l2_num_iterations", "l2_final_cost",
    "l2_output_saved", "l2_output_format",
    "l2_output_paths", "l2_output_save_error", "status", "error", "error_traceback",
]


def _load_completed_keys(csv_path: str | None) -> set[tuple[str, str, str]]:
    """Read an existing benchmark CSV (if any) and return the set of
    (date_str, case_label, time) keys already completed, so a re-run
    after a walltime kill skips them instead of redoing 100-600s of work
    that's already captured."""
    if not csv_path or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    try:
        existing = pd.read_csv(csv_path, usecols=["date_str", "case_label", "time"])
    except Exception:
        return set()
    return {
        (str(r.date_str), str(r.case_label), str(pd.Timestamp(r.time)))
        for r in existing.itertuples(index=False)
    }


def run_benchmark(samples: list[dict], out_csv: str | None = None, save_output_dir: str | None = None) -> pd.DataFrame:
    """Run the L2 benchmark across every (sample observation, case) pair.
    Uses one simulator instance reused across every call in this process.

    If save_output_dir is given, the actual retrieval output (not just
    timing/convergence metadata) is saved per profile via
    benchmark_observation()'s save_output_dir passthrough -- see
    l2_output_* columns in the results.

    Writes each result to out_csv incrementally (flushed after every row),
    not just once at the end -- L2 retrievals here run 100-600+s each with
    high variance, so a walltime kill or crash partway through a large
    sample would otherwise lose everything already computed.

    RESUMABLE: if out_csv already has rows from a previous (killed) run
    over the same samples, matching (date_str, case_label, time) combos
    are skipped rather than recomputed. Returns the FULL combined dataset
    (old + newly computed rows), read back from out_csv."""
    simulator = _get_simulator()

    csv_path = out_csv
    header_written = False
    if csv_path is not None:
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        if file_exists:
            with open(csv_path) as f:
                existing_header = f.readline().strip().split(",")
            if existing_header != _RESULT_FIELDNAMES:
                backup_path = csv_path + ".schema_mismatch.bak"
                log.warning(
                    "%s has a different column schema than the current script "
                    "(existing: %s | current: %s) -- likely reused across script "
                    "versions during development. Backing up to %s and starting "
                    "fresh rather than corrupting further appends.",
                    csv_path, existing_header, _RESULT_FIELDNAMES, backup_path,
                )
                os.rename(csv_path, backup_path)
                file_exists = False
        header_written = file_exists

    completed = _load_completed_keys(out_csv)
    if completed:
        log.info("Resuming: %d results already in %s, matching samples will be skipped",
                  len(completed), out_csv)

    n_total = sum(len(s["h2_for_day"]) for s in samples)
    n_done = 0
    n_skipped_existing = 0
    for sample in samples:
        for case_label in sample["h2_for_day"]:
            key = (sample["date_str"], case_label, str(pd.Timestamp(sample["obs"]["time"])))
            if key in completed:
                n_skipped_existing += 1
                continue

            row = benchmark_observation(sample, case_label, simulator, save_output_dir=save_output_dir)
            n_done += 1

            if csv_path is not None:
                row_df = pd.DataFrame([row])
                row_df.to_csv(csv_path, mode="a", index=False, header=not header_written)
                header_written = True

            log.info(
                "[%d/%d new, %d already done] %s %s: status=%s total=%.1fs l2_marginal=%s converged=%s (%s, nfev=%s)",
                n_done, n_total - n_skipped_existing, n_skipped_existing,
                sample["date_str"], case_label,
                row["status"],
                row["total_time_s"] if row["total_time_s"] is not None else float("nan"),
                f"{row['l2_marginal_time_s']:.1f}s" if row.get("l2_marginal_time_s") is not None else "n/a",
                row.get("converged"),
                row.get("termination_reason"),
                row.get("n_function_evaluations"),
            )

    if n_skipped_existing:
        log.info("Skipped %d already-completed samples from a previous run", n_skipped_existing)

    if csv_path is not None and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return pd.read_csv(csv_path)
    return pd.DataFrame([])


# ---------------------------------------------------------------------------
# Summary and extrapolation
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-case summary: mean/median/max marginal L2 cost, convergence
    rate, error/skip counts."""
    ok = df[df["status"] == "ok"]
    summary = ok.groupby("case_label").agg(
        n=("case_label", "count"),
        total_mean_s=("total_time_s", "mean"),
        l2_marginal_mean_s=("l2_marginal_time_s", "mean"),
        l2_marginal_median_s=("l2_marginal_time_s", "median"),
        l2_marginal_max_s=("l2_marginal_time_s", "max"),
        convergence_rate=("converged", "mean"),
    )
    for status in ("error", "skipped_night"):
        counts = df[df["status"] == status].groupby("case_label").size()
        summary[f"n_{status}"] = counts.reindex(summary.index).fillna(0).astype(int)
    return summary


def estimate_total_profile_count(daytime_fraction: float, n_cases: int) -> dict:
    """
    Rough estimate of total ALI observations across the full simulation
    period, using the actual number of simulation days from the h2 file
    index and a REAL measured daytime fraction.

    daytime_fraction must come from sample_observations()'s returned
    daytime_stats["observed_daytime_fraction"], not from the benchmark
    results df -- since sample_observations() now pre-filters to only
    return confirmed-daytime observations, every row in the benchmark
    results has status="ok" by construction, making
    (df["status"]=="ok").sum()/len(df) trivially 1.0 and silently
    doubling this estimate.
    """
    bg_dates = sorted(_h2_index(BACKGROUND_CASE).keys())
    if RUN_START_DATE or RUN_END_DATE:
        bg_dates = [
            d for d in bg_dates
            if (not RUN_START_DATE or d >= RUN_START_DATE)
            and (not RUN_END_DATE or d <= RUN_END_DATE)
        ]
    n_days = len(bg_dates)
    obs_per_day_theoretical = 86400 // OBS_CADENCE_S

    total_profiles = int(n_days * obs_per_day_theoretical * daytime_fraction * n_cases)
    return {
        "n_simulation_days": n_days,
        "obs_per_day_theoretical": obs_per_day_theoretical,
        "observed_daytime_fraction": daytime_fraction,
        "n_cases": n_cases,
        "estimated_total_profile_count": total_profiles,
    }


def extrapolate_to_full_run(
    summary: pd.DataFrame,
    total_profile_count: int,
    n_parallel_workers: int,
    case_label: str | None = None,
) -> dict:
    """Rough walltime extrapolation for a full production run.

    total_profile_count: from estimate_total_profile_count(), or your own
        number if you already track it precisely.
    n_parallel_workers: number of concurrent SLURM tasks/workers.
    case_label: which case's l2_marginal_mean_s to use; if None, uses the
        weighted mean across all cases in `summary`.
    """
    if case_label is not None:
        per_profile_s = summary.loc[case_label, "l2_marginal_mean_s"]
    else:
        per_profile_s = (summary["l2_marginal_mean_s"] * summary["n"]).sum() / summary["n"].sum()

    total_l2_cpu_seconds = per_profile_s * total_profile_count
    walltime_hours = total_l2_cpu_seconds / n_parallel_workers / 3600.0

    return {
        "per_profile_l2_marginal_s": per_profile_s,
        "total_profile_count": total_profile_count,
        "total_l2_cpu_hours": total_l2_cpu_seconds / 3600.0,
        "estimated_walltime_hours": walltime_hours,
    }


def main(config_path: str) -> None:
    _load_globals(config_path)

    # Pull real observations spread across the simulation period.
    # Start small -- 2 days x 4 obs/day x n_cases is enough to sanity-check
    # before scaling up to a bigger sample.
    samples, daytime_stats = sample_observations(n_days=2, n_obs_per_day=4)
    log.info("Sampled %d observations across %d days", len(samples),
             len({s["date_str"] for s in samples}))
    log.info("Real observed daytime fraction: %.3f (%d/%d candidates)",
             daytime_stats["observed_daytime_fraction"],
             daytime_stats["n_daytime_candidates"], daytime_stats["n_total_candidates"])

    simulator = _get_simulator()

    # Run this first, on one real daytime observation, to see actual
    # function names in the call graph and confirm L2_ENTRY_POINT_CANDIDATES.
    inspect_sample, inspect_case = find_daytime_sample(samples)
    inspect_profile_functions(inspect_sample, inspect_case, simulator)

    df = run_benchmark(samples, out_csv="l2_benchmark_results.csv", save_output_dir="l2_outputs")
    print(df[["case_label", "date_str", "actual_sza_deg", "total_time_s",
               "l2_marginal_time_s", "l2_entry_point", "converged", "status"]])

    summary = summarize(df)
    print(summary)

    n_cases = len(build_case_labels())
    count_est = estimate_total_profile_count(daytime_stats["observed_daytime_fraction"], n_cases)
    print(count_est)

    extrap = extrapolate_to_full_run(
        summary,
        total_profile_count=count_est["estimated_total_profile_count"],
        n_parallel_workers=N_WORKERS,
    )
    print(extrap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    main(args.config)
