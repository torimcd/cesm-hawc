#!/usr/bin/env python
"""
benchmark_l2_retrieval.py
==========================
Benchmark the cost of full ALI L2 retrieval, using REAL orbit geometry and
CESM h2 files pulled through the same machinery as run_orbit_daily.py, to
assess whether running full L2 on the 6-month OSSE production set is
realistic, or whether an averaging-kernel shortcut is needed.

Place this file alongside run_orbit_daily.py (same scripts/ directory, same
config.toml at the project root) -- it imports run_orbit_daily as a module
to reuse orbit-file loading, h2-file indexing, observation extraction, the
per-worker simulator singleton, and the calibration-cache / IERS / Hamilton
patches already established there. Importing run_orbit_daily executes its
module-level setup (config load, logging, monkeypatches) as a side effect,
which is what we want -- the benchmark should run under the same patched
environment as production, not a clean one.

front_end_radiance/l1b are byproducts of the full-L2 product list already
(FULL_L2_PRODUCTS below). Unlike run_orbit_daily's forward-only calls,
here we request the full L2 product list -- l2 has never been run in
production, so there's no existing "forward-only" call to diff against.
Instead each profile is run through simulator.run(FULL_L2_PRODUCTS, ...)
exactly once, wrapped in cProfile.

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

Real per-profile forward-model-only cost (for comparison against the L2
numbers here) should come from your actual Fir job logs / timing around
run_orbit_daily.py's process_day(), not from re-deriving it in this
script -- that's your real production baseline under real cluster
conditions.

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
    _parse_scipy_convergence(). This was confirmed against real output:
    the background case in one run genuinely failed to converge ("The
    maximum number of function evaluations is exceeded"), which the old
    attrs-based placeholder silently reported as unknown (None) rather
    than surfacing as a real non-convergence.
"""

from __future__ import annotations

import cProfile
import contextlib
import io
import pstats
import re
import time
import traceback

import numpy as np
import pandas as pd

# Reuses orbit loading, h2 indexing, observation extraction, the per-worker
# simulator singleton, and all the calibration/IERS/Hamilton patches from
# production. Must live in the same directory, with the same config.toml
# available, as run_orbit_daily.py.
import run_orbit_daily as rod

from cesm_hawc.constituents import build_waccm_constituents


FULL_L2_PRODUCTS = ["l2", "sk2_atmosphere", "front_end_radiance", "l1b"]

# Identified from grepping skretrieval directly (see conversation notes):
# skretrieval.retrieval.processing.Retrieval.retrieve() is the top-level
# orchestrator. Underneath it, Minimizer subclasses (Rodgers,
# SciPyMinimizer, SciPyMinimizerGrad) iterate -- each iteration calls back
# into the forward RT model (via statevector.propagate_wf) to rebuild the
# measurement vector y and jacobian K, then does a Gauss-Newton/Levenberg-
# Marquardt-style update. That means most of the real cost lives in RT
# functions called FROM retrieve(), not in functions literally named
# "retrieve" -- so a keyword sum over per-function OWN time would
# undercount badly. CUMULATIVE time of the single outermost retrieve()
# call is used instead, since cumulative time already includes every
# nested forward-model call made during iteration.
L2_ENTRY_POINT_CANDIDATES = [
    ("skretrieval/retrieval/processing.py", "retrieve"),   # primary: top-level orchestrator
    ("skretrieval/retrieval/rodgers.py", "retrieve"),       # fallback: Rodgers minimizer directly
    ("skretrieval/retrieval/scipy.py", "retrieve"),         # fallback: SciPy minimizer directly
]


# ---------------------------------------------------------------------------
# Real-data sampling (mirrors run_orbit_daily.main()'s case/date setup)
# ---------------------------------------------------------------------------

def build_case_labels() -> dict[str, str]:
    """Same case-label logic as run_orbit_daily.main(); factored out here
    since it isn't a standalone function there."""
    case_labels = {"background": rod.BACKGROUND_CASE}
    for c in rod.INJECTION_CASES:
        m = re.match(r"(sai_[\d.]+Tg)", c)
        label = m.group(1) if m else c
        case_labels[label] = c
    return case_labels


def sample_observations(n_days: int = 3, n_obs_per_day: int = 8, seed: int = 42) -> list[dict]:
    """
    Pull real DAYTIME observations spread across the full simulation
    period and across each sampled day's orbit arc, using the same
    orbit-day mapping and extraction logic as run_orbit_daily.py.

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
        obs is a single observation dict from rod.extract_observations()
        (real lat/lon/time/observer geometry) and h2_for_day maps case
        label -> h2 file path for that date, one entry per sampled
        observation.
      daytime_stats: {n_daytime_candidates, n_total_candidates,
        observed_daytime_fraction, per_day} -- the REAL daytime fraction
        measured before the pre-filter discards night-side candidates.
        Use this (not anything derived from the returned samples/benchmark
        results, which are ALL daytime by construction) for extrapolating
        to the full simulation period via estimate_total_profile_count().
    """
    orbit_files = rod.load_orbit_files()
    orbit_day_idx = rod.build_orbit_day_index(orbit_files)
    n_orbit_days = max(orbit_day_idx.keys()) + 1

    case_labels = build_case_labels()
    h2_indices = {label: rod.build_h2_index(case) for label, case in case_labels.items()}

    bg_dates = sorted(h2_indices["background"].keys())
    if rod.RUN_START_DATE or rod.RUN_END_DATE:
        bg_dates = [
            d for d in bg_dates
            if (not rod.RUN_START_DATE or d >= rod.RUN_START_DATE)
            and (not rod.RUN_END_DATE or d <= rod.RUN_END_DATE)
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

    simulator = rod._get_simulator()

    samples = []
    daytime_stats = {"n_daytime_candidates": 0, "n_total_candidates": 0, "per_day": []}
    for i in sample_idx:
        date_str = bg_dates[i]
        orbit_day = i % n_orbit_days
        if orbit_day not in orbit_day_idx:
            continue

        sim_date = pd.Timestamp(date_str)
        day_obs = rod.extract_observations(
            orbit_day_idx[orbit_day], sim_date, rod.OBS_CADENCE_S, rod.CENTER_PIXEL
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
        waccm = rod.WACCMAtmosphere(h2_for_day["background"], alt_grid_km=rod.ALT_GRID_M / 1e3)
        daytime_obs = []
        for obs in day_obs:
            profiles = waccm.get_column_profiles(obs["lat"], obs["lon"], time_index=0)
            constituents = build_waccm_constituents(profiles, rod.ALT_GRID_M)
            sim_input = {**_build_sim_input(obs), "constituents": constituents}
            try:
                simulator.run(["front_end_radiance", "l1b"], sim_input)
                daytime_obs.append(obs)
            except ValueError as e:
                if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                    continue
                raise

        rod.log.info("%s: %d/%d observations are daytime", date_str, len(daytime_obs), len(day_obs))
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
    """Matches process_day()'s sim_input construction exactly -- SZA/SAA
    are NOT passed explicitly; they're computed automatically from time +
    observer position, same as production."""
    sim_input = {
        "tangent_latitude": float(obs["lat"]),
        "tangent_longitude": float(obs["lon"]),
        "observer_latitude": obs["observer_lat"],
        "observer_longitude": obs["observer_lon"],
        "observer_altitude": obs["observer_alt"],
        "altitude_grid": rod.ALT_GRID_M,
        "polarization_states": ["I", "dolp"],
        "sample_wavelengths": rod.ALI_WAVELENGTHS,
        "time": obs["time"],
    }
    if rod.NOISE_MODEL is not None:
        sim_input["l1b_cfg"] = {"noise_model": rod.NOISE_MODEL}
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
    candidate list. Also useful as a sanity check even when a candidate
    IS found: a call count of 1 on the outermost retrieve() is expected
    (called once per profile); if the underlying Minimizer.retrieve() also
    shows cc=1, that's consistent with one retrieve() call per profile
    doing N forward-model iterations internally -- check the RT function
    call counts too (e.g. sasktran2 entry points) to confirm N > 1."""
    print(f"{'file':<70} {'ncalls':>8} {'cumtime':>10}")
    for (filename, _lineno, fname), (cc, _nc, _tt, ct, _callers) in stats.stats.items():
        if fname == "retrieve":
            print(f"{filename:<70} {cc:>8} {ct:>10.4f}")


# scipy.optimize.least_squares' verbose=2 output always ends with one of
# these termination messages (confirmed from actual run output: e.g.
# "The maximum number of function evaluations is exceeded." for the
# non-converged background case). Parsing this directly is more reliable
# than guessing at hawcsimulator/skretrieval's internal attrs, which
# turned out to carry no diagnostics at all (_extract_l2_diagnostics
# previously returned None for every single row, including the background
# case that actually failed to converge -- silently masking a real
# non-convergence as status="ok").
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
    """Parse scipy.optimize.least_squares' verbose=2 output (captured via
    stdout redirection during the profiled call) for the real convergence
    status and function-evaluation count. Returns
    {converged, termination_reason, n_function_evaluations}, with None
    values if no recognized message was found (e.g. verbose output isn't
    actually coming from this call, or the format changed)."""
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


def _actual_sza(data: dict) -> float | None:
    """Pull the SZA the simulator actually computed for this observation
    (from the l1b dataset), rather than a value we specified -- since
    production doesn't pass SZA explicitly, this is the only way to know
    it for post-hoc stratification of benchmark results."""
    l1b = data.get("l1b", None)
    if l1b is None:
        return None
    try:
        ds = rod._l1b_image_to_dataset(l1b)
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
    simulator = rod._get_simulator()
    for sample in samples:
        for case_label, h2_path in sample["h2_for_day"].items():
            waccm = rod.WACCMAtmosphere(h2_path, alt_grid_km=rod.ALT_GRID_M / 1e3)
            profiles = waccm.get_column_profiles(
                sample["obs"]["lat"], sample["obs"]["lon"], time_index=0
            )
            constituents = build_waccm_constituents(profiles, rod.ALT_GRID_M)
            sim_input = {**_build_sim_input(sample["obs"]), "constituents": constituents}
            try:
                simulator.run(["front_end_radiance", "l1b"], sim_input)
                return sample, case_label
            except ValueError as e:
                if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                    continue
                raise
    raise RuntimeError("No daytime observation found among samples -- widen n_days/n_obs_per_day")


def inspect_profile_functions(sample: dict, case_label: str, simulator, top_n: int = 40) -> pstats.Stats:
    """Run one real observation under cProfile and print the top functions
    by cumulative time, so you can see real function names and tune
    RETRIEVAL_KEYWORDS before running the full benchmark."""
    h2_path = sample["h2_for_day"][case_label]
    waccm = rod.WACCMAtmosphere(h2_path, alt_grid_km=rod.ALT_GRID_M / 1e3)
    profiles = waccm.get_column_profiles(sample["obs"]["lat"], sample["obs"]["lon"], time_index=0)
    constituents = build_waccm_constituents(profiles, rod.ALT_GRID_M)
    sim_input = {**_build_sim_input(sample["obs"]), "constituents": constituents}

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        simulator.run(FULL_L2_PRODUCTS, sim_input)
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

    return stats


def benchmark_observation(sample: dict, case_label: str, simulator) -> dict:
    """Run the full-L2 simulator call once for one real (case, observation)
    pair, under cProfile, and split total time into 'retrieval' vs 'other'
    based on RETRIEVAL_KEYWORDS matching against function names."""
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
        "status": "ok",
        "error": None,
    }

    try:
        h2_path = sample["h2_for_day"][case_label]
        waccm = rod.WACCMAtmosphere(h2_path, alt_grid_km=rod.ALT_GRID_M / 1e3)
        profiles = waccm.get_column_profiles(obs["lat"], obs["lon"], time_index=0)
        constituents = build_waccm_constituents(profiles, rod.ALT_GRID_M)
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
            # and the NEXT profiler.enable() call in the loop fails with
            # "Cannot install a profile function while another profile
            # function is being installed".
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

        diag = _parse_scipy_convergence(stdout_capture.getvalue())
        result["converged"] = diag["converged"]
        result["termination_reason"] = diag["termination_reason"]
        result["n_function_evaluations"] = diag["n_function_evaluations"]

    except ValueError as e:
        if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
            # night-side tangent point, same skip condition as process_day
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


def _load_completed_keys(csv_path: str | None) -> set[tuple[str, str, str]]:
    """Read an existing benchmark CSV (if any) and return the set of
    (date_str, case_label, time) keys already completed, so a re-run
    after a walltime kill skips them instead of redoing 100-600s of work
    that's already captured. This has now happened twice -- worth having
    real resume support rather than relying on manual dedup afterward."""
    import os
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


def run_benchmark(samples: list[dict], out_csv: str | None = None) -> pd.DataFrame:
    """Run the L2 benchmark across every (sample observation, case) pair.
    Uses the same per-worker simulator singleton as production.

    Writes each result to out_csv incrementally (flushed after every row),
    not just once at the end -- L2 retrievals here run 100-600+s each with
    high variance (some cases converge in ~25 function evals, others hit
    the 200-eval cap without converging), so a walltime kill or crash
    partway through a large sample would otherwise lose everything already
    computed.

    RESUMABLE: if out_csv already has rows from a previous (killed) run
    over the same samples, matching (date_str, case_label, time) combos
    are skipped rather than recomputed. Returns the FULL combined dataset
    (old + newly computed rows), read back from out_csv, so downstream
    summarize()/estimate_total_profile_count() see everything -- not just
    whatever this particular invocation computed."""
    simulator = rod._get_simulator()

    completed = _load_completed_keys(out_csv)
    if completed:
        rod.log.info("Resuming: %d results already in %s, matching samples will be skipped",
                      len(completed), out_csv)

    csv_path = out_csv
    header_written = False
    if csv_path is not None:
        import os
        header_written = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    n_total = sum(len(s["h2_for_day"]) for s in samples)
    n_done = 0
    n_skipped_existing = 0
    for sample in samples:
        for case_label in sample["h2_for_day"]:
            key = (sample["date_str"], case_label, str(pd.Timestamp(sample["obs"]["time"])))
            if key in completed:
                n_skipped_existing += 1
                continue

            row = benchmark_observation(sample, case_label, simulator)
            n_done += 1

            if csv_path is not None:
                row_df = pd.DataFrame([row])
                row_df.to_csv(csv_path, mode="a", index=False, header=not header_written)
                header_written = True

            rod.log.info(
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
        rod.log.info("Skipped %d already-completed samples from a previous run", n_skipped_existing)

    if csv_path is not None:
        import os
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
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
    period, using the actual number of simulation days from run_orbit_
    daily's h2 file index and a REAL measured daytime fraction.

    daytime_fraction must come from sample_observations()'s returned
    daytime_stats["observed_daytime_fraction"], not from the benchmark
    results df -- since sample_observations() now pre-filters to only
    return confirmed-daytime observations, every row in the benchmark
    results has status="ok" by construction, making
    (df["status"]=="ok").sum()/len(df) trivially 1.0 and silently
    doubling this estimate (confirmed against real data: two sampled
    days showed ~50% daytime, not the 100% the old df-derived fraction
    implied).
    """
    case_labels = build_case_labels()
    bg_dates = sorted(rod.build_h2_index(rod.BACKGROUND_CASE).keys())
    if rod.RUN_START_DATE or rod.RUN_END_DATE:
        bg_dates = [
            d for d in bg_dates
            if (not rod.RUN_START_DATE or d >= rod.RUN_START_DATE)
            and (not rod.RUN_END_DATE or d <= rod.RUN_END_DATE)
        ]
    n_days = len(bg_dates)
    obs_per_day_theoretical = 86400 // rod.OBS_CADENCE_S

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
    """Rough walltime extrapolation for the full 6-month run.

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


if __name__ == "__main__":
    # Pull real observations spread across the simulation period.
    # Start small -- 3 days x 8 obs/day x n_cases is enough to sanity-check
    # before scaling up to a bigger sample.
    samples, daytime_stats = sample_observations(n_days=3, n_obs_per_day=4)
    rod.log.info("Sampled %d observations across %d days", len(samples),
                 len({s["date_str"] for s in samples}))
    rod.log.info("Real observed daytime fraction: %.3f (%d/%d candidates)",
                 daytime_stats["observed_daytime_fraction"],
                 daytime_stats["n_daytime_candidates"], daytime_stats["n_total_candidates"])

    simulator = rod._get_simulator()

    # Run this first, on one real daytime observation, to see actual
    # function names in the call graph and confirm L2_ENTRY_POINT_CANDIDATES.
    inspect_sample, inspect_case = find_daytime_sample(samples)
    inspect_profile_functions(inspect_sample, inspect_case, simulator)

    df = run_benchmark(samples, out_csv="l2_benchmark_results.csv")
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
        n_parallel_workers=rod.N_WORKERS,
    )
    print(extrap)