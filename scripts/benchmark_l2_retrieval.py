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
  - Convergence / iteration-count diagnostics are pulled from
    data["l2"].attrs -- adjust `_extract_l2_diagnostics()` to match
    whatever sasktran2/hawcsimulator actually exposes; the field names
    here are placeholders.
"""

from __future__ import annotations

import cProfile
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


def sample_observations(n_days: int = 3, n_obs_per_day: int = 8) -> list[dict]:
    """
    Pull real observations spread across the full simulation period, using
    the same orbit-day mapping and extraction logic as run_orbit_daily.py.

    Returns a list of dicts:
        {date_str, orbit_day, obs, h2_for_day}
    where obs is a single observation dict from rod.extract_observations()
    (real lat/lon/time/observer geometry) and h2_for_day maps case label
    -> h2 file path for that date, one entry per sampled observation.
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

    # evenly spaced sample dates across the full run (captures seasonal
    # variation in solar geometry, not just one part of the run)
    sample_idx = np.linspace(0, len(bg_dates) - 1, n_days, dtype=int)

    samples = []
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

        # evenly spaced subsample within the day -- spans the full orbit
        # ground track (different latitudes/SZA) rather than clustering
        obs_idx = np.linspace(0, len(day_obs) - 1, min(n_obs_per_day, len(day_obs)), dtype=int)
        chosen_obs = [day_obs[j] for j in np.unique(obs_idx)]

        h2_for_day = {}
        for label in case_labels:
            if date_str in h2_indices[label]:
                h2_for_day[label] = h2_indices[label][date_str]

        for obs in chosen_obs:
            samples.append({
                "date_str": date_str,
                "orbit_day": int(orbit_day),
                "obs": obs,
                "h2_for_day": h2_for_day,
            })

    return samples


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


def _extract_l2_diagnostics(data: dict) -> dict:
    """Best-effort pull of retrieval diagnostics. PLACEHOLDER field names --
    check what data['l2'] actually carries (likely in .attrs or a
    companion 'retrieval_diagnostics' key) and adjust."""
    l2 = data.get("l2", None)
    if l2 is None:
        return {"converged": None, "n_iterations": None}
    attrs = getattr(l2, "attrs", {})
    return {
        "converged": attrs.get("converged", None),
        "n_iterations": attrs.get("n_iterations", None),
    }


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
    simulator.run(FULL_L2_PRODUCTS, sim_input)
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
        "n_iterations": None,
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
        t0 = time.perf_counter()
        profiler.enable()
        data_full = simulator.run(FULL_L2_PRODUCTS, sim_input)
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

        diag = _extract_l2_diagnostics(data_full)
        result["converged"] = diag["converged"]
        result["n_iterations"] = diag["n_iterations"]

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


def run_benchmark(samples: list[dict], out_csv: str | None = None) -> pd.DataFrame:
    """Run the L2 benchmark across every (sample observation, case) pair.
    Uses the same per-worker simulator singleton as production."""
    simulator = rod._get_simulator()

    rows = []
    for sample in samples:
        for case_label in sample["h2_for_day"]:
            rows.append(benchmark_observation(sample, case_label, simulator))

    df = pd.DataFrame(rows)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df


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


def estimate_total_profile_count(df: pd.DataFrame, n_cases: int) -> dict:
    """
    Rough estimate of total ALI observations across the full simulation
    period, using the night-skip rate observed in this benchmark sample
    and the actual number of simulation days from run_orbit_daily's h2
    file index -- rather than assuming a fixed daytime fraction.
    """
    daytime_frac = (df["status"] == "ok").sum() / max(len(df), 1)

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

    total_profiles = int(n_days * obs_per_day_theoretical * daytime_frac * n_cases)
    return {
        "n_simulation_days": n_days,
        "obs_per_day_theoretical": obs_per_day_theoretical,
        "observed_daytime_fraction": daytime_frac,
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
    samples = sample_observations(n_days=3, n_obs_per_day=8)
    rod.log.info("Sampled %d observations across %d days",
                 len(samples), len({s["date_str"] for s in samples}))

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
    count_est = estimate_total_profile_count(df, n_cases)
    print(count_est)

    extrap = extrapolate_to_full_run(
        summary,
        total_profile_count=count_est["estimated_total_profile_count"],
        n_parallel_workers=rod.N_WORKERS,
    )
    print(extrap)