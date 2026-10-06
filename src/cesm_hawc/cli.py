"""
cesm_hawc.cli
=============
``cesm-hawc`` console-script entry point.

    cesm-hawc save-inputs --config config.toml --mode {fixed,orbit}
    cesm-hawc run         --config config.toml --mode {fixed,orbit}

Every run processes one model case (``[case]`` in the config) and writes
under ``out_dir/<case name>/``.

- ``fixed``: one column at a fixed tangent point and solar geometry, once
  per history file matched by ``[case]``.
- ``orbit``: observations sampled along a real HAWC orbit ground track and
  moved onto the case's dates.

``save-inputs`` only needs the base install (numpy/xarray/scipy) and saves
simulator-ready columns via ``cesm_hawc.save_inputs.save_column_inputs()``.
``run`` needs the ``[sim]`` extra and runs the forward model, plus the L2
retrieval when ``run_l2`` is set.

Worker functions dispatched to ``ProcessPoolExecutor`` (via
``cesm_hawc.dispatch.run_jobs``) are all module-level (not closures/lambdas)
since the default multiprocessing start method on macOS/Windows (``spawn``)
cannot pickle a nested function.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cesm_hawc.config import CesmHawcConfig, ConfigError, load_config
from cesm_hawc.env import configure_environment

log = logging.getLogger("cesm_hawc")

MODES = ("fixed", "orbit")

_L2_DIAG_FIELDNAMES = [
    "case_label", "time", "lat", "lon", "elapsed_s",
    "converged", "termination_reason", "n_function_evaluations",
    "l2_num_iterations", "l2_final_cost", "status", "error",
]


def _require_sim_deps() -> None:
    try:
        import hawcsimulator  # noqa: F401
        import sasktran2  # noqa: F401
    except ImportError:
        sys.exit(
            "The 'run' command requires the [sim] extra:\n"
            "    pip install cesm-hawc[sim]\n"
        )


@dataclass(frozen=True)
class _RunSettings:
    """``[case]`` with command-line overrides applied."""
    name: str            # case name: also the output folder name
    waccm_dir: str
    out_dir: str
    n_workers: int
    run_l2: bool
    strip_ozone: bool

    @property
    def output_name(self) -> str:
        # An ozone-stripped run reads the same files as a normal run of this
        # case, so it writes to its own folder rather than overwriting or
        # resuming into the normal run's output.
        return f"{self.name}_no_ozone" if self.strip_ozone else self.name

    @property
    def case_out(self) -> str:
        return os.path.join(self.out_dir, self.output_name)


def _settings(cfg: CesmHawcConfig, args: argparse.Namespace) -> _RunSettings:
    c = cfg.case
    name = args.case_name or c.name
    return _RunSettings(
        name=name,
        waccm_dir=c.waccm_dir_for(name),
        out_dir=args.out_dir or c.out_dir,
        n_workers=args.n_workers or c.n_workers,
        run_l2=c.run_l2,
        strip_ozone=getattr(args, "strip_ozone", False) or c.strip_ozone,
    )


def _strip_ozone(profiles: dict) -> dict:
    return {**profiles, "vmr_o3": np.zeros_like(profiles["vmr_o3"])}


# ---------------------------------------------------------------------------
# fixed: job building
# ---------------------------------------------------------------------------

def _fixed_jobs(cfg: CesmHawcConfig, s: _RunSettings) -> list[tuple[str, str, pd.Timestamp]]:
    """One job per history file: ``(path, label, obs_time)``, where
    ``label`` is the file name without ``.nc``."""
    from cesm_hawc import file_index

    c, f = cfg.case, cfg.fixed
    jobs = []
    for path in file_index.list_files(s.waccm_dir, c.pattern):
        date = file_index.filename_date(path)
        if (c.start_date or c.end_date):
            if date is None:
                log.warning("Skipping %s: no date in the file name to compare with "
                            "start_date/end_date", path)
                continue
            if not file_index.date_in_range(date, c.start_date, c.end_date):
                continue
        if f.obs_time:
            obs_time = pd.Timestamp(f.obs_time)
        else:
            obs_time = file_index.filename_time(path)
            if obs_time is None:
                sys.exit(f"Cannot work out an observation time from the file name {path}; "
                         "set obs_time in [fixed].")
        label = os.path.splitext(os.path.basename(path))[0]
        jobs.append((path, label, obs_time))
    return jobs


# ---------------------------------------------------------------------------
# orbit: job building
# ---------------------------------------------------------------------------

def _orbit_day_index(o, out_dir):
    from cesm_hawc import orbit_files

    orbit_paths = orbit_files.load_orbit_files(o.orbit_dir, o.orbit_pattern)
    epoch = pd.Timestamp(o.orbit_epoch)
    cache_path = os.path.join(out_dir, ".orbit_day_index_cache.json")
    day_idx = orbit_files.build_orbit_day_index(orbit_paths, epoch, cache_path=cache_path)
    return day_idx, max(day_idx.keys()) + 1, epoch


def _build_orbit_daily_jobs(o, waccm_dir: str, pattern: str, start_date, end_date,
                            out_dir: str) -> list[tuple[str, list[dict], str]]:
    """One job per calendar date of the case's history files:
    ``(date, observations, history_file)``. The *n*-th date (sorted) uses
    orbit day *n* mod the number of orbit days."""
    from cesm_hawc import file_index, orbit_files

    day_idx, n_orbit_days, epoch = _orbit_day_index(o, out_dir)
    h2_index = file_index.index_by_date(waccm_dir, pattern)

    jobs = []
    for i, date_str in enumerate(sorted(h2_index.keys())):
        if not file_index.date_in_range(date_str, start_date, end_date):
            continue
        orbit_day = i % n_orbit_days
        if orbit_day not in day_idx:
            continue
        obs = orbit_files.extract_observations(
            day_idx[orbit_day], pd.Timestamp(date_str), o.obs_cadence_s, o.center_pixel, epoch
        )
        if obs:
            jobs.append((date_str, obs, h2_index[date_str]))
    return jobs


def _build_orbit_subdaily_jobs(o, waccm_dir: str, pattern: str, start_date, end_date,
                               out_dir: str) -> list[tuple[str, list[dict], str]]:
    """Like ``_build_orbit_daily_jobs``, but for history output written more
    than once per day: one job per *file*, labelled
    ``YYYY-MM-DD-SSSSS``.

    The full-day observation pool per calendar date is the same as the
    daily path. Each observation is then assigned to whichever snapshot is
    nearest to it in time (via bisection against all of the case's
    snapshot times, not a fixed clock-time split), which also handles an
    observation near midnight being closer to the next date's snapshot.
    """
    import bisect

    from cesm_hawc import file_index, orbit_files

    day_idx, n_orbit_days, epoch = _orbit_day_index(o, out_dir)
    h2_index = file_index.index_by_timestamp(waccm_dir, pattern)
    if not h2_index:
        return []
    snapshot_times = sorted(h2_index.keys())
    case_dates = sorted({ts.strftime("%Y-%m-%d") for ts in snapshot_times})

    buckets: dict[pd.Timestamp, list] = {ts: [] for ts in snapshot_times}
    for i, date_str in enumerate(case_dates):
        if not file_index.date_in_range(date_str, start_date, end_date):
            continue
        orbit_day = i % n_orbit_days
        if orbit_day not in day_idx:
            continue
        obs = orbit_files.extract_observations(
            day_idx[orbit_day], pd.Timestamp(date_str), o.obs_cadence_s, o.center_pixel, epoch
        )
        for ob in obs:
            t = pd.Timestamp(ob["time"])
            pos = bisect.bisect_left(snapshot_times, t)
            candidates = [c for c in (pos - 1, pos) if 0 <= c < len(snapshot_times)]
            nearest = min(candidates, key=lambda c: abs(snapshot_times[c] - t))
            buckets[snapshot_times[nearest]].append(ob)

    jobs = []
    for ts in snapshot_times:
        if not buckets[ts]:
            continue
        seconds_of_day = int((ts - ts.normalize()).total_seconds())
        label = f"{ts.strftime('%Y-%m-%d')}-{seconds_of_day:05d}"
        jobs.append((label, buckets[ts], h2_index[ts]))
    return jobs


def _orbit_jobs(cfg: CesmHawcConfig, s: _RunSettings) -> list[tuple[str, list[dict], str]]:
    c, o = cfg.case, cfg.orbit
    build = _build_orbit_subdaily_jobs if o.h2_cadence == "subdaily" else _build_orbit_daily_jobs
    return build(o, s.waccm_dir, c.pattern, c.start_date, c.end_date, s.out_dir)


# ---------------------------------------------------------------------------
# save-inputs workers
# ---------------------------------------------------------------------------

def _save_fixed_file(path: str, out_path: str, lat: float, lon: float, obs_time,
                     time_index: int, alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray,
                     profiles_only: bool) -> str:
    from cesm_hawc.save_inputs import save_column_inputs
    from cesm_hawc.waccm import WACCMAtmosphere

    label = os.path.basename(out_path)
    try:
        waccm = WACCMAtmosphere(path, alt_grid_km=alt_grid_m / 1e3)
        save_column_inputs(waccm, lat, lon, out_path, time_index, alt_grid_m,
                           wavelengths_nm, profiles_only, obs_time=obs_time)
        return f"OK   {label}"
    except Exception:
        log.error("[%s] FAILED:\n%s", label, traceback.format_exc())
        return f"FAIL {label}"


def _save_orbit_job(label: str, observations: list[dict], h2_path: str, job_out: str,
                    time_index: int, alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray,
                    profiles_only: bool) -> str:
    """Save every observation's column to ``job_out/column_<HHMMSS>.nc``,
    with its observation time and satellite position as attributes."""
    from cesm_hawc.save_inputs import save_column_inputs
    from cesm_hawc.waccm import WACCMAtmosphere

    try:
        waccm = WACCMAtmosphere(h2_path, alt_grid_km=alt_grid_m / 1e3)
        os.makedirs(job_out, exist_ok=True)
        for obs in observations:
            t_str = pd.Timestamp(obs["time"]).strftime("%H%M%S")
            save_column_inputs(
                waccm, obs["lat"], obs["lon"], os.path.join(job_out, f"column_{t_str}.nc"),
                time_index, alt_grid_m, wavelengths_nm, profiles_only,
                obs_time=obs["time"],
                extra_attrs={
                    "observer_latitude": float(obs["observer_lat"]),
                    "observer_longitude": float(obs["observer_lon"]),
                    "observer_altitude": float(obs["observer_alt"]),
                },
            )
        return f"OK   {label}  ({len(observations)} obs)"
    except Exception:
        log.error("[%s] FAILED:\n%s", label, traceback.format_exc())
        return f"FAIL {label}"


# ---------------------------------------------------------------------------
# run workers
# ---------------------------------------------------------------------------

def _save_cesm_extinction(waccm_obj, lat, lon, time_index, alt_grid_m, out_dir) -> None:
    import xarray as xr

    extracted = waccm_obj.extract_cesm_extinction(lat, lon, time_index, alt_grid_m)
    if not extracted:
        log.warning("No EXTINCT* variables in file; skipping cesm_extinction.nc")
        return
    label_map = {"EXTINCTdn": "ext_550nm", "EXTINCTUVdn": "ext_350nm", "EXTINCTNIRdn": "ext_1020nm"}
    data_vars = {label_map.get(k, k): ("altitude_m", v) for k, v in extracted.items()}
    ds = xr.Dataset(data_vars, coords={"altitude_m": alt_grid_m},
                    attrs={"description": "CESM aerosol extinction from EXTINCTdn/EXTINCTUVdn/EXTINCTNIRdn"})
    ds.to_netcdf(os.path.join(out_dir, "cesm_extinction.nc"))


def _run_fixed_file(path: str, file_out: str, lat: float, lon: float, sza_deg: float,
                    saa_deg: float, obs_time, time_index: int, alt_grid_m: np.ndarray,
                    wavelengths_nm: np.ndarray, run_l2: bool, strip_ozone: bool) -> str:
    """One history file, one column. Writes ``l1b.nc`` (with truth
    extinction), ``l2.nc`` when ``run_l2``, ``cesm_extinction.nc`` and
    ``summary.txt`` to ``file_out``."""
    from cesm_hawc.noise import default_noise_model
    from cesm_hawc.orbit_files import l1b_image_to_dataset
    from cesm_hawc.outputs import format_burden_summary, write_text_summary
    from cesm_hawc.simulation import products_for, run_ali_simulation_from_profiles
    from cesm_hawc.waccm import WACCMAtmosphere

    # Re-applies the calibration-database race patch inside this worker
    # process. configure_environment() in the main process (see main())
    # isn't guaranteed to reach worker processes depending on the
    # multiprocessing start method. Idempotent, cheap to call again.
    configure_environment()

    label = os.path.basename(file_out)
    try:
        waccm = WACCMAtmosphere(path, alt_grid_km=alt_grid_m / 1e3)
        profiles = waccm.get_column_profiles(lat, lon, time_index)
        if strip_ozone:
            profiles = _strip_ozone(profiles)
        sim_geometry = {
            "tangent_latitude": lat, "tangent_longitude": lon,
            "tangent_solar_zenith_angle": sza_deg, "tangent_solar_azimuth_angle": saa_deg,
            "altitude_grid": alt_grid_m, "polarization_states": ["I", "dolp"],
            "sample_wavelengths": wavelengths_nm, "time": pd.Timestamp(obs_time),
        }
        data, true_ext = run_ali_simulation_from_profiles(
            profiles, alt_grid_m, sim_geometry, products=products_for(run_l2),
            noise_model=default_noise_model(), return_extinction=True,
            truth_wavelengths_nm=wavelengths_nm,
        )

        os.makedirs(file_out, exist_ok=True)
        l1b_image_to_dataset(data["l1b"], wavelengths_nm, true_ext, alt_grid_m).to_netcdf(
            os.path.join(file_out, "l1b.nc"))
        if run_l2:
            data["l2"].to_netcdf(os.path.join(file_out, "l2.nc"))
        _save_cesm_extinction(waccm, lat, lon, time_index, alt_grid_m, file_out)

        lines = [f"File: {path}", f"Time index: {time_index}", f"Observation time: {obs_time}",
                 f"Tangent lat/lon: {lat}, {lon}", "",
                 "Stratospheric sulfate (15-35 km):"]
        lines += format_burden_summary(waccm.sulfate_column_burden(lat, lon, time_index))
        write_text_summary(lines, file_out)
        return f"OK   {label}"
    except Exception:
        log.error("[%s] FAILED:\n%s", label, traceback.format_exc())
        return f"FAIL {label}"


def _safe_time_str(t) -> str:
    return str(pd.Timestamp(t)).replace(" ", "T").replace(":", "")


def _run_orbit_job(label: str, observations: list[dict], case_name: str, h2_path: str,
                   out_root: str, time_index: int, alt_grid_m: np.ndarray,
                   wavelengths_nm: np.ndarray, run_l2: bool, strip_ozone: bool = False) -> str:
    """One job (a day, or one sub-daily history file) of an orbit run.
    Forward-only by default; full L2 retrieval per observation when
    ``run_l2`` is True (slow: minutes per profile). L2 mode is resumable
    within a job via an incrementally-written diagnostics CSV plus
    per-profile .nc saves; see ``cesm_hawc.resume``. All output lands under
    ``out_root/case_name/label/``.
    """
    import contextlib
    import io
    import time as time_mod

    import xarray as xr

    from cesm_hawc.constituents import build_waccm_constituents
    from cesm_hawc.convergence import extract_l2_native_diagnostics, parse_scipy_convergence
    from cesm_hawc.noise import default_noise_model
    from cesm_hawc.orbit_files import l1b_image_to_dataset
    from cesm_hawc.resume import append_csv_row, load_completed_keys
    from cesm_hawc.simulation import FORWARD_PRODUCTS, products_for
    from cesm_hawc.waccm import WACCMAtmosphere

    # See the matching comment in _run_fixed_file.
    configure_environment()

    try:
        from hawcsimulator.ali.configurations.ideal_dolp_imager import IdealALISimulator
        simulator = IdealALISimulator()
        waccm = WACCMAtmosphere(h2_path, alt_grid_km=alt_grid_m / 1e3)

        noise_model = default_noise_model()
        products = products_for(run_l2)

        case_out = os.path.join(out_root, case_name, label)
        l1b_results: list = []
        l2_results: list = []

        l2_diag_csv = os.path.join(case_out, "l2_diagnostics.csv")
        completed_keys: set[str] = set()
        if run_l2:
            completed_keys = {k[0] for k in
                              load_completed_keys(l2_diag_csv, ["time"], _L2_DIAG_FIELDNAMES)}

        successful_obs = []
        for obs in observations:
            t, lat, lon = obs["time"], obs["lat"], obs["lon"]
            time_key = str(pd.Timestamp(t))
            sim_input = {
                "tangent_latitude": float(lat), "tangent_longitude": float(lon),
                "observer_latitude": obs["observer_lat"], "observer_longitude": obs["observer_lon"],
                "observer_altitude": obs["observer_alt"], "altitude_grid": alt_grid_m,
                "polarization_states": ["I", "dolp"], "sample_wavelengths": wavelengths_nm, "time": t,
                "l1b_cfg": {"noise_model": noise_model},
            }

            profiles = waccm.get_column_profiles(lat, lon, time_index)
            if strip_ozone:
                profiles = _strip_ozone(profiles)
            constituents, true_ext = build_waccm_constituents(
                profiles, alt_grid_m, return_extinction=True, truth_wavelengths_nm=wavelengths_nm
            )
            already_done = run_l2 and time_key in completed_keys

            l2_stdout = io.StringIO()
            obs_t0 = time_mod.perf_counter()
            try:
                if already_done:
                    data = simulator.run(list(FORWARD_PRODUCTS), {**sim_input, "constituents": constituents})
                elif run_l2:
                    with contextlib.redirect_stdout(l2_stdout):
                        data = simulator.run(list(products), {**sim_input, "constituents": constituents})
                else:
                    data = simulator.run(list(products), {**sim_input, "constituents": constituents})
            except ValueError as e:
                if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                    continue  # night-side tangent point
                raise
            except Exception:
                tb = traceback.format_exc()
                log.error("[%s] %s at %s FAILED, skipping this profile only:\n%s",
                          label, case_name, time_key, tb)
                if run_l2:
                    append_csv_row(l2_diag_csv, {
                        "case_label": case_name, "time": time_key, "lat": lat, "lon": lon,
                        "elapsed_s": None, "converged": None, "termination_reason": None,
                        "n_function_evaluations": None, "l2_num_iterations": None,
                        "l2_final_cost": None, "status": "error", "error": str(tb)[-500:],
                    }, _L2_DIAG_FIELDNAMES)
                continue
            obs_elapsed = time_mod.perf_counter() - obs_t0

            ds_obs = l1b_image_to_dataset(data["l1b"], wavelengths_nm, true_ext, alt_grid_m)

            if run_l2:
                saved_path = os.path.join(case_out, "l2_profiles", f"{case_name}_{_safe_time_str(t)}.nc")
                if already_done:
                    if os.path.exists(saved_path):
                        l2_results.append(xr.open_dataset(saved_path).load())
                else:
                    l2_obj = data.get("l2")
                    diag = parse_scipy_convergence(l2_stdout.getvalue())
                    native_diag = extract_l2_native_diagnostics(l2_obj)
                    if l2_obj is not None:
                        l2_results.append(l2_obj)
                        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
                        l2_obj.to_netcdf(saved_path)
                    append_csv_row(l2_diag_csv, {
                        "case_label": case_name, "time": time_key, "lat": lat, "lon": lon,
                        "elapsed_s": obs_elapsed, "converged": diag["converged"],
                        "termination_reason": diag["termination_reason"],
                        "n_function_evaluations": diag["n_function_evaluations"],
                        "l2_num_iterations": native_diag["l2_num_iterations"],
                        "l2_final_cost": native_diag["l2_final_cost"],
                        "status": "ok", "error": None,
                    }, _L2_DIAG_FIELDNAMES)

            l1b_results.append(ds_obs)
            successful_obs.append(obs)

        lats = [o["lat"] for o in successful_obs]
        lons = [o["lon"] for o in successful_obs]
        times = [o["time"] for o in successful_obs]

        if l1b_results:
            os.makedirs(case_out, exist_ok=True)
            curtain = xr.concat(l1b_results, dim="along_track").assign_coords(
                lat=("along_track", lats), lon=("along_track", lons), time=("along_track", times)
            )
            curtain.to_netcdf(os.path.join(case_out, "curtain.nc"))

        if run_l2 and l2_results:
            try:
                l2_curtain = xr.concat(l2_results, dim="along_track").assign_coords(
                    lat=("along_track", lats), lon=("along_track", lons), time=("along_track", times)
                )
                l2_curtain.to_netcdf(os.path.join(case_out, "l2_retrieval.nc"))
            except Exception:
                log.error("[%s] failed to concat/save l2_retrieval.nc for case %s "
                          "(per-profile l2_profiles/*.nc are still on disk):\n%s",
                          label, case_name, traceback.format_exc())

        os.makedirs(case_out, exist_ok=True)
        pd.DataFrame({"time": [o["time"].isoformat() for o in successful_obs], "lat": lats, "lon": lons}
                     ).to_csv(os.path.join(case_out, "orbit_track.csv"), index=False)

        return f"OK   {label}  {case_name}  ({len(successful_obs)}/{len(observations)} obs)"
    except Exception:
        log.error("[%s] %s FAILED:\n%s", label, case_name, traceback.format_exc())
        return f"FAIL {label}  {case_name}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _require_mode_table(cfg: CesmHawcConfig, command: str, mode: str) -> None:
    if getattr(cfg, mode) is None:
        sys.exit(f"{command} --mode {mode} requires a [{mode}] table in config.toml")


def _save_inputs(cfg: CesmHawcConfig, args: argparse.Namespace) -> None:
    from cesm_hawc.dispatch import run_jobs

    mode = args.mode
    _require_mode_table(cfg, "save-inputs", mode)
    s = _settings(cfg, args)
    ins = cfg.instrument
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)
    time_index = cfg.case.time_index

    if mode == "fixed":
        f = cfg.fixed
        worker = _save_fixed_file
        jobs = [(path, os.path.join(s.case_out, f"{label}.nc"), f.tangent_lat, f.tangent_lon,
                 obs_time, time_index, alt_grid_m, wavelengths_nm, args.profiles_only)
                for path, label, obs_time in _fixed_jobs(cfg, s)]
        unit = "files"
    else:
        worker = _save_orbit_job
        jobs = [(label, obs, h2_path, os.path.join(s.case_out, label), time_index,
                 alt_grid_m, wavelengths_nm, args.profiles_only)
                for label, obs, h2_path in _orbit_jobs(cfg, s)]
        unit = "jobs"

    if args.dry_run:
        log.info("[dry-run] would save inputs for %d %s to %s", len(jobs), unit, s.case_out)
        return

    os.makedirs(s.case_out, exist_ok=True)
    if not args.profiles_only:
        _warm_mode_databases_if_available()
    results = run_jobs(worker, jobs, s.n_workers, on_result=lambda r: log.info(r))
    _report(results, unit)


def _run(cfg: CesmHawcConfig, args: argparse.Namespace) -> None:
    from cesm_hawc.calibration import warm_calibration_database, warm_retrieval_optical_database
    from cesm_hawc.constituents import warm_mode_databases
    from cesm_hawc.dispatch import run_jobs
    from cesm_hawc.resume import outputs_already_exist

    mode = args.mode
    _require_mode_table(cfg, "run", mode)
    s = _settings(cfg, args)
    ins = cfg.instrument
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)
    time_index = cfg.case.time_index

    if s.run_l2:
        log.warning("run_l2 is enabled: L2 retrieval takes minutes per profile. "
                    "Confirm your walltime/CPU-hour budget.")
    if s.strip_ozone:
        log.warning("strip_ozone is enabled: WACCM ozone VMR will be zeroed before building "
                    "the simulated atmosphere. Output folder: %s", s.case_out)

    jobs, n_skipped = [], 0
    if mode == "fixed":
        f = cfg.fixed
        worker = _run_fixed_file
        unit = "files"
        for path, label, obs_time in _fixed_jobs(cfg, s):
            file_out = os.path.join(s.case_out, label)
            expected = [os.path.join(file_out, "l1b.nc")]
            if s.run_l2:
                expected.append(os.path.join(file_out, "l2.nc"))
            if outputs_already_exist(expected):
                n_skipped += 1
                continue
            jobs.append((path, file_out, f.tangent_lat, f.tangent_lon, f.sza_deg, f.saa_deg,
                         obs_time, time_index, alt_grid_m, wavelengths_nm, s.run_l2, s.strip_ozone))
    else:
        worker = _run_orbit_job
        unit = "jobs"
        for label, obs, h2_path in _orbit_jobs(cfg, s):
            expected = [os.path.join(s.case_out, label, "curtain.nc")]
            if s.run_l2:
                expected.append(os.path.join(s.case_out, label, "l2_retrieval.nc"))
            if outputs_already_exist(expected):
                n_skipped += 1
                continue
            jobs.append((label, obs, s.output_name, h2_path, s.out_dir, time_index,
                         alt_grid_m, wavelengths_nm, s.run_l2, s.strip_ozone))

    if n_skipped:
        log.info("Skipping %d %s already completed by a previous run", n_skipped, unit)

    if args.dry_run:
        log.info("[dry-run] would run %d %s, output to %s", len(jobs), unit, s.case_out)
        return

    os.makedirs(s.case_out, exist_ok=True)
    log.info("Pre-warming calibration database and Mie databases...")
    warm_calibration_database()
    warm_mode_databases()
    warm_retrieval_optical_database()

    # Recycle workers after each job when running L2: retrieval state not
    # released between jobs was a real source of OOM kills in long runs.
    # See cesm_hawc.dispatch.
    max_tasks_per_child = 1 if s.run_l2 else None
    results = run_jobs(worker, jobs, s.n_workers, max_tasks_per_child=max_tasks_per_child,
                       on_result=lambda r: log.info(r))
    _report(results, unit)


# ---------------------------------------------------------------------------
# Shared helpers + entry point
# ---------------------------------------------------------------------------

def _warm_mode_databases_if_available() -> None:
    """Pre-warm the mode-specific Mie databases before ``save-inputs``
    dispatches to a worker pool, same race-condition concern as `run`'s
    pre-warm calls, but non-fatal here, since ``save-inputs`` must keep
    working when sasktran2 isn't installed at all."""
    try:
        from cesm_hawc.constituents import warm_mode_databases
    except ImportError:
        return
    log.info("Pre-warming mode-specific Mie databases...")
    try:
        warm_mode_databases()
    except Exception as e:
        log.warning("Could not pre-warm Mie databases: %s", e)


def _report(results: list[str], unit: str) -> None:
    ok = [r for r in results if r.startswith("OK")]
    fail = [r for r in results if r.startswith("FAIL")]
    log.info("-- Run complete --")
    log.info("  Succeeded: %d / %d %s", len(ok), len(results), unit)
    if fail:
        log.warning("  Failed:")
        for f in fail:
            log.warning("    %s", f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cesm-hawc",
        description="Simulate HAWC ALI observations and retrievals from CESM2/WACCM output.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default="config.toml", help="Path to config.toml")
        sp.add_argument("--mode", required=True, choices=MODES,
                        help="fixed: one column at a fixed point per history file; "
                             "orbit: observations along a real orbit ground track")
        sp.add_argument("--case-name", default=None,
                        help="Override [case] name (also fills {name} in waccm_dir), so one "
                             "config.toml can be reused across cases without editing it")
        sp.add_argument("--out-dir", default=None, help="Override [case] out_dir")
        sp.add_argument("--n-workers", type=int, default=None, help="Override [case] n_workers")
        sp.add_argument("--dry-run", action="store_true",
                        help="Report the job count without running anything")

    save_inputs_p = sub.add_parser(
        "save-inputs",
        help="Extract and save simulator-ready WACCM columns (no sasktran2/hawcsimulator needed)",
    )
    add_common(save_inputs_p)
    save_inputs_p.add_argument(
        "--profiles-only", action="store_true",
        help="Skip computing simulator constituents even if sasktran2 is available "
             "(save only the raw WACCM profile fields)",
    )

    run_p = sub.add_parser(
        "run",
        help="Run the forward model, and the L2 retrieval if run_l2 is set (requires the [sim] extra)",
    )
    add_common(run_p)
    run_p.add_argument(
        "--strip-ozone", action="store_true",
        help="Zero WACCM ozone VMR before building the simulated atmosphere, writing to "
             "<case name>_no_ozone/ instead of <case name>/. Turns strip_ozone on; it cannot "
             "turn off strip_ozone = true set in config.toml.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    configure_environment()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        sys.exit(str(e))

    if args.command == "save-inputs":
        _save_inputs(cfg, args)
    elif args.command == "run":
        _require_sim_deps()
        _run(cfg, args)


if __name__ == "__main__":
    main()
