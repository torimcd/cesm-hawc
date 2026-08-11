"""
cesm_hawc.cli
=============
``cesm-hawc`` console-script entry point.

    cesm-hawc save-inputs --config config.toml --mode {single,batch,orbit-track,orbit-file}
    cesm-hawc run         --config config.toml --mode {single,batch,orbit-track,orbit-file}

``save-inputs`` only needs the base install (numpy/xarray/scipy) — it saves
WACCM column profiles via ``WACCMAtmosphere.save_column_profiles()``.
``run`` needs the ``[sim]`` extra (hawcsimulator + a conda-forge sasktran2
install) — it runs the full forward model / L2 retrieval.

Worker functions dispatched to ``ProcessPoolExecutor`` (via
``cesm_hawc.dispatch.run_jobs``) are all module-level (not closures/lambdas)
since the default multiprocessing start method on macOS/Windows (``spawn``)
cannot pickle a nested function.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd

from cesm_hawc.config import CesmHawcConfig, ConfigError, load_config
from cesm_hawc.env import configure_environment

log = logging.getLogger("cesm_hawc")


def _require_sim_deps() -> None:
    try:
        import hawcsimulator  # noqa: F401
    except ImportError:
        sys.exit(
            "The 'run' command requires hawcsimulator and sasktran2.\n"
            "sasktran2 is conda-forge only (not pip-installable):\n"
            "    micromamba install -c conda-forge sasktran2\n"
            "then:\n"
            "    pip install cesm-hawc[sim]\n"
        )
    try:
        import sasktran2  # noqa: F401
    except ImportError:
        sys.exit(
            "The 'run' command requires sasktran2, which is conda-forge "
            "only (not pip-installable):\n"
            "    micromamba install -c conda-forge sasktran2\n"
        )


# ---------------------------------------------------------------------------
# save-inputs: single
# ---------------------------------------------------------------------------

def _save_inputs_single(cfg: CesmHawcConfig, out_dir_override, dry_run, profiles_only=False) -> None:
    from cesm_hawc.save_inputs import save_column_inputs
    from cesm_hawc.waccm import WACCMAtmosphere

    if cfg.single is None or cfg.geometry is None:
        sys.exit("save-inputs --mode single requires [single] and [geometry] in config.toml")
    s, geo, ins = cfg.single, cfg.geometry, cfg.instrument
    out_dir = out_dir_override or s.out_dir
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)

    n = 1 + (1 if s.waccm_injection else 0)
    if dry_run:
        log.info("[dry-run] would save %d column file(s) to %s", n, out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)
    waccm_bg = WACCMAtmosphere(s.waccm_background, alt_grid_km=alt_grid_m / 1e3)
    save_column_inputs(waccm_bg, geo.tangent_lat, geo.tangent_lon,
                        os.path.join(out_dir, "background_column.nc"), s.time_idx,
                        alt_grid_m, wavelengths_nm, profiles_only)
    if s.waccm_injection:
        waccm_inj = WACCMAtmosphere(s.waccm_injection, alt_grid_km=alt_grid_m / 1e3)
        save_column_inputs(waccm_inj, geo.tangent_lat, geo.tangent_lon,
                            os.path.join(out_dir, "injection_column.nc"), s.time_idx,
                            alt_grid_m, wavelengths_nm, profiles_only)


# ---------------------------------------------------------------------------
# save-inputs: batch (monthly h0 files)
# ---------------------------------------------------------------------------

def _save_inputs_month(date: str, bg_path: str, inj_path: str | None,
                        out_root: str, lat: float, lon: float,
                        alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray,
                        profiles_only: bool = False) -> str:
    from cesm_hawc.save_inputs import save_column_inputs
    from cesm_hawc.waccm import WACCMAtmosphere

    try:
        bg_out = os.path.join(out_root, "background")
        os.makedirs(bg_out, exist_ok=True)
        waccm_bg = WACCMAtmosphere(bg_path, alt_grid_km=alt_grid_m / 1e3)
        save_column_inputs(waccm_bg, lat, lon, os.path.join(bg_out, f"{date}.nc"), 0,
                            alt_grid_m, wavelengths_nm, profiles_only)
        if inj_path is not None:
            inj_out = os.path.join(out_root, "injection")
            os.makedirs(inj_out, exist_ok=True)
            waccm_inj = WACCMAtmosphere(inj_path, alt_grid_km=alt_grid_m / 1e3)
            save_column_inputs(waccm_inj, lat, lon, os.path.join(inj_out, f"{date}.nc"), 0,
                                alt_grid_m, wavelengths_nm, profiles_only)
        return f"OK   {date}"
    except Exception:
        log.error("[%s] FAILED:\n%s", date, traceback.format_exc())
        return f"FAIL {date}"


def _save_inputs_batch(cfg: CesmHawcConfig, out_dir_override, n_workers_override, dry_run,
                        profiles_only=False) -> None:
    from cesm_hawc import file_index
    from cesm_hawc.dispatch import run_jobs

    if cfg.batch is None or cfg.geometry is None:
        sys.exit("save-inputs --mode batch requires [batch] and [geometry] in config.toml")
    b, geo, ins = cfg.batch, cfg.geometry, cfg.instrument
    out_dir = out_dir_override or b.out_dir
    n_workers = n_workers_override or b.n_workers
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)

    bg_files = file_index.index_by_month(b.waccm_background_dir, b.h0_pattern, b.month_filter)
    inj_files = (file_index.index_by_month(b.waccm_injection_dir, b.h0_pattern, b.month_filter)
                 if b.waccm_injection_dir else {})

    dates = sorted(bg_files.keys())
    jobs = [(date, bg_files[date], inj_files.get(date), out_dir, geo.tangent_lat,
             geo.tangent_lon, alt_grid_m, wavelengths_nm, profiles_only) for date in dates]

    if dry_run:
        log.info("[dry-run] would save inputs for %d month(s) to %s", len(jobs), out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)
    if not profiles_only:
        _warm_mode_databases_if_available()
    results = run_jobs(_save_inputs_month, jobs, n_workers, on_result=lambda r: log.info(r))
    _report(results, "months")


# ---------------------------------------------------------------------------
# save-inputs: orbit-track
# ---------------------------------------------------------------------------

def _save_inputs_orbit_track_day(date_str: str, day_obs: list[dict],
                                  out_root: str, alt_grid_m: np.ndarray,
                                  wavelengths_nm: np.ndarray, profiles_only: bool = False) -> str:
    """``day_obs`` items: {time, lat, lon, bg_file, inj_file}."""
    from cesm_hawc.save_inputs import save_column_inputs
    from cesm_hawc.waccm import WACCMAtmosphere

    try:
        day_out = os.path.join(out_root, date_str)
        os.makedirs(day_out, exist_ok=True)
        waccm_cache: dict[str, WACCMAtmosphere] = {}

        def get_waccm(path: str) -> WACCMAtmosphere:
            if path not in waccm_cache:
                waccm_cache[path] = WACCMAtmosphere(path, alt_grid_km=alt_grid_m / 1e3)
            return waccm_cache[path]

        for i, obs in enumerate(day_obs):
            t_str = pd.Timestamp(obs["time"]).strftime("%H%M%S")
            save_column_inputs(
                get_waccm(obs["bg_file"]), obs["lat"], obs["lon"],
                os.path.join(day_out, f"background_{t_str}.nc"), 0,
                alt_grid_m, wavelengths_nm, profiles_only,
            )
            if obs.get("inj_file"):
                save_column_inputs(
                    get_waccm(obs["inj_file"]), obs["lat"], obs["lon"],
                    os.path.join(day_out, f"injection_{t_str}.nc"), 0,
                    alt_grid_m, wavelengths_nm, profiles_only,
                )
        return f"OK   {date_str}  ({len(day_obs)} obs)"
    except Exception:
        log.error("[%s] FAILED:\n%s", date_str, traceback.format_exc())
        return f"FAIL {date_str}"


def _save_inputs_orbit_track_real_files_day(date_str: str, observations: list[dict],
                                             h2_files: dict[str, str], out_root: str,
                                             alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray,
                                             profiles_only: bool = False) -> str:
    """``h2_files`` maps a case label (e.g. "background", "sai_1.0Tg") to
    that case's single h2 file for this date — shared across every
    observation in ``observations``, unlike the analytical mode's
    per-observation ``bg_file``/``inj_file``."""
    from cesm_hawc.save_inputs import save_column_inputs
    from cesm_hawc.waccm import WACCMAtmosphere

    try:
        waccm_cache: dict[str, WACCMAtmosphere] = {}

        def get_waccm(path: str) -> WACCMAtmosphere:
            if path not in waccm_cache:
                waccm_cache[path] = WACCMAtmosphere(path, alt_grid_km=alt_grid_m / 1e3)
            return waccm_cache[path]

        for obs in observations:
            t_str = pd.Timestamp(obs["time"]).strftime("%H%M%S")
            for label, h2_path in h2_files.items():
                case_out = os.path.join(out_root, label, date_str)
                os.makedirs(case_out, exist_ok=True)
                save_column_inputs(
                    get_waccm(h2_path), obs["lat"], obs["lon"],
                    os.path.join(case_out, f"column_{t_str}.nc"), 0,
                    alt_grid_m, wavelengths_nm, profiles_only,
                )
        return f"OK   {date_str}  ({len(observations)} obs, {len(h2_files)} case(s))"
    except Exception:
        log.error("[%s] FAILED:\n%s", date_str, traceback.format_exc())
        return f"FAIL {date_str}"


def _save_inputs_orbit_track(cfg: CesmHawcConfig, out_dir_override, n_workers_override, dry_run,
                              profiles_only=False) -> None:
    from cesm_hawc.dispatch import run_jobs

    if cfg.orbit is None:
        sys.exit("save-inputs --mode orbit-track requires [orbit] in config.toml")
    o, ins = cfg.orbit, cfg.instrument
    out_dir = out_dir_override or o.out_dir
    n_workers = n_workers_override or o.n_workers
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)

    if o.track_source == "analytical":
        raw_jobs = _build_orbit_track_analytical_jobs(o, out_dir, alt_grid_m)
        jobs = [(date, obs, out_dir, alt_grid_m, wavelengths_nm, profiles_only)
                for date, obs, _, _ in raw_jobs]
        worker_fn = _save_inputs_orbit_track_day
    else:
        raw_jobs = _build_orbit_track_real_files_jobs(o, out_dir, alt_grid_m, run_l2=False)
        jobs = [(date, obs, h2_files, out_dir, alt_grid_m, wavelengths_nm, profiles_only)
                for date, obs, h2_files, _, _, _ in raw_jobs]
        worker_fn = _save_inputs_orbit_track_real_files_day

    if dry_run:
        log.info("[dry-run] would save inputs for %d day(s) to %s", len(jobs), out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)
    if not profiles_only:
        _warm_mode_databases_if_available()
    results = run_jobs(worker_fn, jobs, n_workers, on_result=lambda r: log.info(r))
    _report(results, "days")


def _build_orbit_track_analytical_jobs(o, out_dir, alt_grid_m):
    from cesm_hawc import file_index, orbit as orbit_mod

    bg_index = file_index.index_by_timestamp(o.waccm_background_dir, o.file_pattern)
    inj_index = (file_index.index_by_timestamp(o.waccm_injection_dir, o.file_pattern)
                 if o.waccm_injection_dir else {})
    sorted_bg_ts = sorted(bg_index.keys())

    start_time = pd.Timestamp(o.start_time) if o.start_time else sorted_bg_ts[0]
    end_time = pd.Timestamp(o.end_time) if o.end_time else sorted_bg_ts[-1]

    track = orbit_mod.generate_sso_ground_track(
        start_time, end_time, o.obs_cadence_s, o.altitude_km, o.inclination_deg, o.start_lon_deg
    )
    track["bg_file"] = track["time"].apply(
        lambda t: file_index.find_nearest(t, bg_index, o.max_gap_s)
    )
    track["inj_file"] = (
        track["time"].apply(lambda t: file_index.find_nearest(t, inj_index, o.max_gap_s))
        if inj_index else None
    )
    track = track.dropna(subset=["bg_file"]).reset_index(drop=True)
    track["_date"] = track["time"].dt.strftime("%Y-%m-%d")

    jobs = []
    for date_str, day_df in track.groupby("_date"):
        day_obs = day_df.drop(columns="_date").to_dict(orient="records")
        jobs.append((date_str, day_obs, out_dir, alt_grid_m))
    return jobs


def _case_labels(background_case: str, injection_cases: list[str]) -> dict[str, str]:
    labels = {"background": background_case}
    for c in injection_cases:
        m = re.match(r"(sai_[\d.]+Tg)", c)
        labels[m.group(1) if m else c] = c
    return labels


def _build_orbit_track_real_files_jobs(o, out_dir, alt_grid_m, run_l2: bool):
    from cesm_hawc import file_index, orbit_files

    orbit_paths = orbit_files.load_orbit_files(o.orbit_dir, o.orbit_pattern)
    epoch = pd.Timestamp(o.orbit_epoch)
    cache_path = os.path.join(out_dir, ".orbit_day_index_cache.json")
    day_idx = orbit_files.build_orbit_day_index(orbit_paths, epoch, cache_path=cache_path)
    n_orbit_days = max(day_idx.keys()) + 1

    case_labels = _case_labels(o.background_case, o.injection_cases)
    h2_indices = {
        label: file_index.index_by_date(
            os.path.join(o.waccm_data_dir, case, "atm", "hist"), o.h2_pattern
        )
        for label, case in case_labels.items()
    }
    bg_dates = sorted(h2_indices["background"].keys())

    jobs = []
    for i, date_str in enumerate(bg_dates):
        if o.run_start_date and date_str < o.run_start_date:
            continue
        if o.run_end_date and date_str > o.run_end_date:
            continue
        orbit_day = i % n_orbit_days
        if orbit_day not in day_idx:
            continue

        sim_date = pd.Timestamp(date_str)
        obs = orbit_files.extract_observations(
            day_idx[orbit_day], sim_date, o.obs_cadence_s, o.center_pixel, epoch
        )
        if not obs:
            continue

        h2_for_day = {label: h2_indices[label][date_str]
                      for label in case_labels if date_str in h2_indices[label]}
        if not h2_for_day:
            continue
        jobs.append((date_str, obs, h2_for_day, out_dir, alt_grid_m, run_l2))
    return jobs


# ---------------------------------------------------------------------------
# save-inputs: orbit-file
# ---------------------------------------------------------------------------

def _save_inputs_orbit_file(orbit_path: str, h2_bg_path: str, h2_inj_path: str | None,
                             out_dir: str, across_indices: list[int], time_stride: int,
                             alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray,
                             profiles_only: bool = False) -> str:
    import xarray as xr
    from cesm_hawc.save_inputs import save_column_inputs
    from cesm_hawc.waccm import WACCMAtmosphere

    orbit_name = os.path.splitext(os.path.basename(orbit_path))[0]
    try:
        orbit = xr.open_dataset(orbit_path, decode_times=True)
        n_time, n_across = orbit.sizes["time"], orbit.sizes["across"]
        idxs = across_indices or list(range(n_across))
        times = list(range(0, n_time, time_stride))

        waccm_bg = WACCMAtmosphere(h2_bg_path, alt_grid_km=alt_grid_m / 1e3)
        waccm_inj = WACCMAtmosphere(h2_inj_path, alt_grid_km=alt_grid_m / 1e3) if h2_inj_path else None

        os.makedirs(out_dir, exist_ok=True)
        n_saved = 0
        for t_idx in times:
            for ac_idx in idxs:
                lat = float(orbit["latitude"].isel(time=t_idx, along=0, across=ac_idx))
                lon = float(orbit["longitude"].isel(time=t_idx, along=0, across=ac_idx))
                if not np.isfinite(lat) or not np.isfinite(lon):
                    continue
                stem = f"{orbit_name}_t{t_idx:05d}_a{ac_idx:04d}"
                save_column_inputs(waccm_bg, lat, lon, os.path.join(out_dir, f"{stem}_bg.nc"), 0,
                                    alt_grid_m, wavelengths_nm, profiles_only)
                if waccm_inj is not None:
                    save_column_inputs(waccm_inj, lat, lon, os.path.join(out_dir, f"{stem}_inj.nc"), 0,
                                        alt_grid_m, wavelengths_nm, profiles_only)
                n_saved += 1
        orbit.close()
        return f"OK   {orbit_name}  ({n_saved} columns)"
    except Exception:
        log.error("[%s] FAILED:\n%s", orbit_name, traceback.format_exc())
        return f"FAIL {orbit_name}"


def _build_orbit_file_jobs(orb, out_dir):
    from cesm_hawc import file_index, orbit_files

    bg_index = file_index.index_by_date(orb.waccm_background_dir, orb.h2_pattern)
    inj_index = file_index.index_by_date(orb.waccm_injection_dir, orb.h2_pattern) if orb.waccm_injection_dir else {}
    orbit_by_date = orbit_files.collect_orbit_files_by_date(orb.orbit_dir, orb.orbit_pattern)

    jobs = []
    for date, orbit_paths in sorted(orbit_by_date.items()):
        if date not in bg_index:
            continue
        day_out = os.path.join(out_dir, date)
        for op in orbit_paths:
            jobs.append((op, bg_index[date], inj_index.get(date), day_out))
    return jobs


def _save_inputs_orbit_file_cmd(cfg: CesmHawcConfig, out_dir_override, n_workers_override, dry_run,
                                 profiles_only=False) -> None:
    from cesm_hawc.dispatch import run_jobs

    if cfg.orbit_real is None:
        sys.exit("save-inputs --mode orbit-file requires [orbit_real] in config.toml")
    orb, ins = cfg.orbit_real, cfg.instrument
    out_dir = out_dir_override or orb.out_dir
    n_workers = n_workers_override or orb.n_workers
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)

    raw_jobs = _build_orbit_file_jobs(orb, out_dir)
    jobs = [(op, bg, inj, day_out, orb.across_indices, orb.time_stride, alt_grid_m,
             wavelengths_nm, profiles_only)
            for op, bg, inj, day_out in raw_jobs]

    if dry_run:
        log.info("[dry-run] would save inputs for %d orbit file(s) to %s", len(jobs), out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)
    if not profiles_only:
        _warm_mode_databases_if_available()
    results = run_jobs(_save_inputs_orbit_file, jobs, n_workers, on_result=lambda r: log.info(r))
    _report(results, "orbit files")


# ---------------------------------------------------------------------------
# run: single
# ---------------------------------------------------------------------------

def _run_single(cfg: CesmHawcConfig, out_dir_override, dry_run) -> None:
    if cfg.single is None or cfg.geometry is None:
        sys.exit("run --mode single requires [single] and [geometry] in config.toml")
    s, geo, ins = cfg.single, cfg.geometry, cfg.instrument
    out_dir = out_dir_override or s.out_dir

    if dry_run:
        log.info("[dry-run] would run single simulation, output to %s", out_dir)
        return

    from cesm_hawc.noise import default_noise_model
    from cesm_hawc.outputs import format_anomaly_summary, format_burden_summary, write_text_summary
    from cesm_hawc.simulation import run_ali_simulation
    from cesm_hawc.waccm import WACCMAtmosphere

    os.makedirs(out_dir, exist_ok=True)
    alt_grid_m = ins.altitude_grid_m()

    result = run_ali_simulation(
        background_file=s.waccm_background,
        injection_file=s.waccm_injection,
        lat=geo.tangent_lat, lon=geo.tangent_lon,
        time_index=s.time_idx,
        sza_deg=geo.sza_deg, saa_deg=geo.saa_deg,
        obs_time=s.obs_time,
        wavelengths_nm=np.array(ins.wavelengths_nm),
        alt_grid_m=alt_grid_m,
        noise_model=default_noise_model(),
    )

    result["data_bg"]["l2"].to_netcdf(os.path.join(out_dir, "l2_background.nc"))
    waccm_bg = WACCMAtmosphere(s.waccm_background, alt_grid_km=alt_grid_m / 1e3)
    _save_cesm_extinction(waccm_bg, geo.tangent_lat, geo.tangent_lon, s.time_idx,
                          alt_grid_m, out_dir, "background")

    lines = [f"TANGENT_LAT: {geo.tangent_lat}", f"TANGENT_LON: {geo.tangent_lon}", "",
             "Background stratospheric sulfate (15-35 km):"]
    lines += format_burden_summary(result["burden_bg"])

    if result["data_inj"] is not None:
        result["data_inj"]["l2"].to_netcdf(os.path.join(out_dir, "l2_injection.nc"))
        waccm_inj = WACCMAtmosphere(s.waccm_injection, alt_grid_km=alt_grid_m / 1e3)
        _save_cesm_extinction(waccm_inj, geo.tangent_lat, geo.tangent_lon, s.time_idx,
                              alt_grid_m, out_dir, "injection")
        lines += ["", "Injection stratospheric sulfate (15-35 km):"]
        lines += format_burden_summary(result["burden_inj"])
        lines += ["", *format_anomaly_summary(result["peak_extinction_anomaly_m"],
                                               result["peak_radius_anomaly_nm"],
                                               result["delta_burden_mg_m2"])]
    write_text_summary(lines, out_dir)


def _save_cesm_extinction(waccm_obj, lat, lon, time_index, alt_grid_m, out_dir, tag) -> None:
    import xarray as xr

    extracted = waccm_obj.extract_cesm_extinction(lat, lon, time_index, alt_grid_m)
    if not extracted:
        log.warning("[%s] No EXTINCT* variables found in file — skipping cesm_extinction_%s.nc",
                    tag, tag)
        return
    label_map = {"EXTINCTdn": "ext_550nm", "EXTINCTUVdn": "ext_350nm", "EXTINCTNIRdn": "ext_1020nm"}
    data_vars = {label_map.get(k, k): ("altitude_m", v) for k, v in extracted.items()}
    ds = xr.Dataset(data_vars, coords={"altitude_m": alt_grid_m},
                     attrs={"description": "CESM aerosol extinction from EXTINCTdn/EXTINCTUVdn/EXTINCTNIRdn"})
    ds.to_netcdf(os.path.join(out_dir, f"cesm_extinction_{tag}.nc"))


# ---------------------------------------------------------------------------
# run: batch (monthly h0 files)
# ---------------------------------------------------------------------------

def _run_month(date: str, bg_path: str, inj_path: str | None, out_root: str,
               lat: float, lon: float, sza_deg: float, saa_deg: float,
               alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray) -> str:
    from cesm_hawc.noise import default_noise_model
    from cesm_hawc.outputs import format_anomaly_summary, format_burden_summary, write_text_summary
    from cesm_hawc.simulation import DEFAULT_PRODUCTS, run_ali_simulation_from_profiles
    from cesm_hawc.waccm import WACCMAtmosphere

    try:
        sim_geometry = {
            "tangent_latitude": lat, "tangent_longitude": lon,
            "tangent_solar_zenith_angle": sza_deg, "tangent_solar_azimuth_angle": saa_deg,
            "altitude_grid": alt_grid_m, "polarization_states": ["I", "dolp"],
            "sample_wavelengths": wavelengths_nm,
            "time": pd.Timestamp(f"{date}-15T12:00:00Z"),
        }
        noise_model = default_noise_model()

        bg_out = os.path.join(out_root, "background", date)
        os.makedirs(bg_out, exist_ok=True)
        waccm_bg = WACCMAtmosphere(bg_path, alt_grid_km=alt_grid_m / 1e3)
        profiles_bg = waccm_bg.get_column_profiles(lat, lon, 0)
        data_bg = run_ali_simulation_from_profiles(
            profiles_bg, alt_grid_m, sim_geometry, products=DEFAULT_PRODUCTS, noise_model=noise_model
        )
        data_bg["l2"].to_netcdf(os.path.join(bg_out, "l2_background.nc"))
        _save_cesm_extinction(waccm_bg, lat, lon, 0, alt_grid_m, bg_out, "background")
        burden_bg = waccm_bg.sulfate_column_burden(lat, lon, 0)

        lines = [f"Month: {date}", f"TANGENT_LAT: {lat}", f"TANGENT_LON: {lon}", "",
                 "Background stratospheric sulfate (15-35 km):"]
        lines += format_burden_summary(burden_bg)

        if inj_path is not None:
            inj_out = os.path.join(out_root, "injection", date)
            os.makedirs(inj_out, exist_ok=True)
            waccm_inj = WACCMAtmosphere(inj_path, alt_grid_km=alt_grid_m / 1e3)
            profiles_inj = waccm_inj.get_column_profiles(lat, lon, 0)
            data_inj = run_ali_simulation_from_profiles(
                profiles_inj, alt_grid_m, sim_geometry, products=DEFAULT_PRODUCTS, noise_model=noise_model
            )
            data_inj["l2"].to_netcdf(os.path.join(inj_out, "l2_injection.nc"))
            _save_cesm_extinction(waccm_inj, lat, lon, 0, alt_grid_m, inj_out, "injection")
            burden_inj = waccm_inj.sulfate_column_burden(lat, lon, 0)

            ext_bg = data_bg["l2"]["stratospheric_aerosol_extinction_per_m"]
            ext_inj = data_inj["l2"]["stratospheric_aerosol_extinction_per_m"]
            r_bg = data_bg["l2"]["stratospheric_aerosol_median_radius"]
            r_inj = data_inj["l2"]["stratospheric_aerosol_median_radius"]
            strat = ext_bg.altitude.values > 15000
            peak_ext = float((ext_inj - ext_bg).values[strat].max())
            peak_r = float((r_inj - r_bg).values[strat].max())
            d_burden = burden_inj["burden_mg_m2"] - burden_bg["burden_mg_m2"]

            lines += ["", "Injection stratospheric sulfate (15-35 km):"]
            lines += format_burden_summary(burden_inj)
            lines += ["", *format_anomaly_summary(peak_ext, peak_r, d_burden)]

            diff_out = os.path.join(out_root, "diff", date)
            write_text_summary(lines, diff_out, "summary_diff.txt")
        else:
            write_text_summary(lines, bg_out, "summary.txt")

        return f"OK   {date}"
    except Exception:
        log.error("[%s] FAILED:\n%s", date, traceback.format_exc())
        return f"FAIL {date}"


def _run_batch(cfg: CesmHawcConfig, out_dir_override, n_workers_override, dry_run) -> None:
    from cesm_hawc import file_index
    from cesm_hawc.calibration import warm_calibration_database
    from cesm_hawc.constituents import warm_mode_databases
    from cesm_hawc.dispatch import run_jobs

    if cfg.batch is None or cfg.geometry is None:
        sys.exit("run --mode batch requires [batch] and [geometry] in config.toml")
    b, geo, ins = cfg.batch, cfg.geometry, cfg.instrument
    out_dir = out_dir_override or b.out_dir
    n_workers = n_workers_override or b.n_workers
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)

    bg_files = file_index.index_by_month(b.waccm_background_dir, b.h0_pattern, b.month_filter)
    inj_files = (file_index.index_by_month(b.waccm_injection_dir, b.h0_pattern, b.month_filter)
                 if b.waccm_injection_dir else {})
    dates = sorted(bg_files.keys())
    jobs = [(date, bg_files[date], inj_files.get(date), out_dir, geo.tangent_lat, geo.tangent_lon,
             geo.sza_deg, geo.saa_deg, alt_grid_m, wavelengths_nm) for date in dates]

    if dry_run:
        log.info("[dry-run] would run %d month(s), output to %s", len(jobs), out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)
    log.info("Pre-warming calibration database and Mie databases...")
    warm_calibration_database()
    warm_mode_databases()

    results = run_jobs(_run_month, jobs, n_workers, on_result=lambda r: log.info(r))
    _report(results, "months")


# ---------------------------------------------------------------------------
# run: orbit-track (analytical + real_files)
# ---------------------------------------------------------------------------

def _run_orbit_track_analytical_day(date_str: str, day_obs: list[dict], out_root: str,
                                     alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray) -> str:
    import xarray as xr
    from cesm_hawc.noise import default_noise_model
    from cesm_hawc.outputs import write_text_summary
    from cesm_hawc.simulation import DEFAULT_PRODUCTS, run_ali_simulation_from_profiles
    from cesm_hawc.waccm import WACCMAtmosphere

    try:
        day_out = os.path.join(out_root, date_str)
        os.makedirs(day_out, exist_ok=True)
        waccm_cache: dict[str, WACCMAtmosphere] = {}

        def get_waccm(path: str) -> WACCMAtmosphere:
            if path not in waccm_cache:
                waccm_cache[path] = WACCMAtmosphere(path, alt_grid_km=alt_grid_m / 1e3)
            return waccm_cache[path]

        noise_model = default_noise_model()
        l2_bg_list, l2_inj_list, summary_rows = [], [], []

        for obs in day_obs:
            t, lat, lon = obs["time"], obs["lat"], obs["lon"]
            sim_geometry = {
                "tangent_latitude": float(lat), "tangent_longitude": float(lon),
                "altitude_grid": alt_grid_m, "polarization_states": ["I", "dolp"],
                "sample_wavelengths": wavelengths_nm, "time": t,
            }
            profiles_bg = get_waccm(obs["bg_file"]).get_column_profiles(lat, lon, 0)
            data_bg = run_ali_simulation_from_profiles(
                profiles_bg, alt_grid_m, sim_geometry, products=DEFAULT_PRODUCTS, noise_model=noise_model
            )
            l2_bg_list.append(data_bg["l2"])

            peak_ext_anom = peak_r_anom = None
            if obs.get("inj_file"):
                profiles_inj = get_waccm(obs["inj_file"]).get_column_profiles(lat, lon, 0)
                data_inj = run_ali_simulation_from_profiles(
                    profiles_inj, alt_grid_m, sim_geometry, products=DEFAULT_PRODUCTS, noise_model=noise_model
                )
                l2_inj_list.append(data_inj["l2"])
                ext_bg = data_bg["l2"]["stratospheric_aerosol_extinction_per_m"]
                ext_inj = data_inj["l2"]["stratospheric_aerosol_extinction_per_m"]
                r_bg = data_bg["l2"]["stratospheric_aerosol_median_radius"]
                r_inj = data_inj["l2"]["stratospheric_aerosol_median_radius"]
                strat = ext_bg.altitude.values > 15000
                peak_ext_anom = float((ext_inj - ext_bg).values[strat].max())
                peak_r_anom = float((r_inj - r_bg).values[strat].max())

            summary_rows.append({"time": t.isoformat(), "lat": lat, "lon": lon,
                                  "peak_ext_anom_per_m": peak_ext_anom, "peak_r_anom_nm": peak_r_anom})

        lats = [o["lat"] for o in day_obs]
        lons = [o["lon"] for o in day_obs]
        times = [o["time"] for o in day_obs]

        def make_curtain(l2_list):
            curtain = xr.concat(l2_list, dim="along_track")
            return curtain.assign_coords(lat=("along_track", lats), lon=("along_track", lons),
                                          time=("along_track", times))

        make_curtain(l2_bg_list).to_netcdf(os.path.join(day_out, "curtain_background.nc"))
        if l2_inj_list:
            make_curtain(l2_inj_list).to_netcdf(os.path.join(day_out, "curtain_injection.nc"))

        pd.DataFrame([{"time": o["time"], "lat": o["lat"], "lon": o["lon"], "bg_file": o["bg_file"]}
                      for o in day_obs]).to_csv(os.path.join(day_out, "orbit_track.csv"), index=False)

        lines = [f"Date: {date_str}", f"Observations: {len(day_obs)}", ""]
        for row in summary_rows:
            line = f"  {row['time']}  lat={row['lat']:+7.2f}  lon={row['lon']:+8.2f}"
            if row["peak_ext_anom_per_m"] is not None:
                line += f"  Δext={row['peak_ext_anom_per_m']:.2e} m⁻¹  Δr={row['peak_r_anom_nm']:.1f} nm"
            lines.append(line)
        write_text_summary(lines, day_out)

        return f"OK   {date_str}  ({len(day_obs)} obs)"
    except Exception:
        log.error("[%s] FAILED:\n%s", date_str, traceback.format_exc())
        return f"FAIL {date_str}"


_L2_PRODUCTS = ("l2", "sk2_atmosphere", "front_end_radiance", "l1b")
_FORWARD_ONLY_PRODUCTS = ("front_end_radiance", "l1b")
_L2_DIAG_FIELDNAMES = [
    "case_label", "time", "lat", "lon", "elapsed_s",
    "converged", "termination_reason", "n_function_evaluations",
    "l2_num_iterations", "l2_final_cost", "status", "error",
]


def _safe_time_str(t) -> str:
    return str(pd.Timestamp(t)).replace(" ", "T").replace(":", "")


def _run_orbit_daily_case_day(sim_date_str: str, observations: list[dict],
                               h2_files: dict[str, str], out_root: str,
                               alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray,
                               run_l2: bool) -> str:
    """One day, all cases. Forward-only by default; full L2 retrieval per
    observation when ``run_l2`` is True (slow: 100-600s/profile measured in
    the original benchmarking). L2 mode is resumable within a day via an
    incrementally-written diagnostics CSV plus per-profile .nc saves — see
    ``cesm_hawc.resume``.
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
    from cesm_hawc.waccm import WACCMAtmosphere

    try:
        from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator
        simulator = IdealALISimulator()
        waccm_cache: dict[str, WACCMAtmosphere] = {}

        def get_waccm(path: str) -> WACCMAtmosphere:
            if path not in waccm_cache:
                waccm_cache[path] = WACCMAtmosphere(path, alt_grid_km=alt_grid_m / 1e3)
            return waccm_cache[path]

        noise_model = default_noise_model()
        products = _L2_PRODUCTS if run_l2 else _FORWARD_ONLY_PRODUCTS

        results: dict[str, list] = {label: [] for label in h2_files}
        l2_results: dict[str, list] = {label: [] for label in h2_files}

        l2_diag_csv = os.path.join(out_root, "background", sim_date_str, "l2_diagnostics.csv")
        completed_keys: set[tuple] = set()
        if run_l2:
            completed_keys = load_completed_keys(l2_diag_csv, ["case_label", "time"], _L2_DIAG_FIELDNAMES)

        successful_obs = []
        for obs in observations:
            t, lat, lon = obs["time"], obs["lat"], obs["lon"]
            time_key = str(pd.Timestamp(t))
            sim_geometry = {
                "tangent_latitude": float(lat), "tangent_longitude": float(lon),
                "observer_latitude": obs["observer_lat"], "observer_longitude": obs["observer_lon"],
                "observer_altitude": obs["observer_alt"], "altitude_grid": alt_grid_m,
                "polarization_states": ["I", "dolp"], "sample_wavelengths": wavelengths_nm, "time": t,
            }
            sim_input = dict(sim_geometry)
            sim_input["l1b_cfg"] = {"noise_model": noise_model}

            obs_l1b: dict[str, "xr.Dataset"] = {}
            skip_obs = False
            for label, h2_path in h2_files.items():
                profiles = get_waccm(h2_path).get_column_profiles(lat, lon, 0)
                constituents, true_ext = build_waccm_constituents(
                    profiles, alt_grid_m, return_extinction=True, truth_wavelengths_nm=wavelengths_nm
                )
                already_done = run_l2 and (label, time_key) in completed_keys

                l2_stdout = io.StringIO()
                obs_t0 = time_mod.perf_counter()
                try:
                    if already_done:
                        data = simulator.run(list(_FORWARD_ONLY_PRODUCTS), {**sim_input, "constituents": constituents})
                    elif run_l2:
                        with contextlib.redirect_stdout(l2_stdout):
                            data = simulator.run(list(products), {**sim_input, "constituents": constituents})
                    else:
                        data = simulator.run(list(products), {**sim_input, "constituents": constituents})
                except ValueError as e:
                    if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                        skip_obs = True
                        break
                    raise
                except Exception:
                    tb = traceback.format_exc()
                    log.error("[%s] %s at %s FAILED, skipping this profile only:\n%s",
                              sim_date_str, label, time_key, tb)
                    if run_l2:
                        append_csv_row(l2_diag_csv, {
                            "case_label": label, "time": time_key, "lat": lat, "lon": lon,
                            "elapsed_s": None, "converged": None, "termination_reason": None,
                            "n_function_evaluations": None, "l2_num_iterations": None,
                            "l2_final_cost": None, "status": "error", "error": str(tb)[-500:],
                        }, _L2_DIAG_FIELDNAMES)
                    continue
                obs_elapsed = time_mod.perf_counter() - obs_t0

                ds_obs = l1b_image_to_dataset(data["l1b"], wavelengths_nm, true_ext, alt_grid_m)
                obs_l1b[label] = ds_obs

                if run_l2:
                    if already_done:
                        saved_path = os.path.join(out_root, label, sim_date_str, "l2_profiles",
                                                   f"{label}_{_safe_time_str(t)}.nc")
                        if os.path.exists(saved_path):
                            l2_results[label].append(xr.open_dataset(saved_path).load())
                    else:
                        l2_obj = data.get("l2")
                        diag = parse_scipy_convergence(l2_stdout.getvalue())
                        native_diag = extract_l2_native_diagnostics(l2_obj)
                        if l2_obj is not None:
                            l2_results[label].append(l2_obj)
                            saved_path = os.path.join(out_root, label, sim_date_str, "l2_profiles",
                                                       f"{label}_{_safe_time_str(t)}.nc")
                            os.makedirs(os.path.dirname(saved_path), exist_ok=True)
                            l2_obj.to_netcdf(saved_path)
                        append_csv_row(l2_diag_csv, {
                            "case_label": label, "time": time_key, "lat": lat, "lon": lon,
                            "elapsed_s": obs_elapsed, "converged": diag["converged"],
                            "termination_reason": diag["termination_reason"],
                            "n_function_evaluations": diag["n_function_evaluations"],
                            "l2_num_iterations": native_diag["l2_num_iterations"],
                            "l2_final_cost": native_diag["l2_final_cost"],
                            "status": "ok", "error": None,
                        }, _L2_DIAG_FIELDNAMES)

            if skip_obs:
                continue
            for label, l1b in obs_l1b.items():
                results[label].append(l1b)
            successful_obs.append(obs)

        lats = [o["lat"] for o in successful_obs]
        lons = [o["lon"] for o in successful_obs]
        times = [o["time"] for o in successful_obs]

        for label, l1b_list in results.items():
            if not l1b_list:
                continue
            case_out = os.path.join(out_root, label, sim_date_str)
            os.makedirs(case_out, exist_ok=True)
            curtain = xr.concat(l1b_list, dim="along_track").assign_coords(
                lat=("along_track", lats), lon=("along_track", lons), time=("along_track", times)
            )
            curtain.to_netcdf(os.path.join(case_out, "curtain.nc"))

        if run_l2:
            for label, l2_list in l2_results.items():
                if not l2_list:
                    continue
                case_out = os.path.join(out_root, label, sim_date_str)
                try:
                    l2_curtain = xr.concat(l2_list, dim="along_track").assign_coords(
                        lat=("along_track", lats), lon=("along_track", lons), time=("along_track", times)
                    )
                    l2_curtain.to_netcdf(os.path.join(case_out, "l2_retrieval.nc"))
                except Exception:
                    log.error("[%s] failed to concat/save l2_retrieval.nc for case %s "
                              "(per-profile l2_profiles/*.nc are still on disk):\n%s",
                              sim_date_str, label, traceback.format_exc())

        bg_out = os.path.join(out_root, "background", sim_date_str)
        os.makedirs(bg_out, exist_ok=True)
        pd.DataFrame({"time": [o["time"].isoformat() for o in successful_obs], "lat": lats, "lon": lons}
                     ).to_csv(os.path.join(bg_out, "orbit_track.csv"), index=False)

        return f"OK   {sim_date_str}  ({len(successful_obs)}/{len(observations)} obs)"
    except Exception:
        log.error("[%s] FAILED:\n%s", sim_date_str, traceback.format_exc())
        return f"FAIL {sim_date_str}"


def _run_orbit_track(cfg: CesmHawcConfig, out_dir_override, n_workers_override, dry_run) -> None:
    from cesm_hawc.calibration import warm_calibration_database
    from cesm_hawc.constituents import warm_mode_databases
    from cesm_hawc.dispatch import run_jobs
    from cesm_hawc.resume import outputs_already_exist

    if cfg.orbit is None:
        sys.exit("run --mode orbit-track requires [orbit] in config.toml")
    o, ins = cfg.orbit, cfg.instrument
    out_dir = out_dir_override or o.out_dir
    n_workers = n_workers_override or o.n_workers
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)

    if o.track_source == "analytical":
        raw_jobs = _build_orbit_track_analytical_jobs(o, out_dir, alt_grid_m)
        jobs = [(date, obs, out_dir, alt_grid_m, wavelengths_nm) for date, obs, _, _ in raw_jobs]
        worker_fn = _run_orbit_track_analytical_day
        max_tasks_per_child = None
    else:
        if o.run_l2:
            log.warning("run_l2 is enabled: L2 retrieval is slow (100-600s/profile measured "
                        "in the original benchmarking). Confirm your walltime/CPU-hour budget.")
        raw_jobs = _build_orbit_track_real_files_jobs(o, out_dir, alt_grid_m, o.run_l2)
        jobs = []
        n_skipped = 0
        for date_str, obs, h2_for_day, _, _, run_l2 in raw_jobs:
            expected = [os.path.join(out_dir, label, date_str, "curtain.nc") for label in h2_for_day]
            if o.run_l2:
                expected += [os.path.join(out_dir, label, date_str, "l2_retrieval.nc") for label in h2_for_day]
            if outputs_already_exist(expected):
                n_skipped += 1
                continue
            jobs.append((date_str, obs, h2_for_day, out_dir, alt_grid_m, wavelengths_nm, o.run_l2))
        if n_skipped:
            log.info("Skipping %d day(s) already fully completed from a previous run", n_skipped)
        worker_fn = _run_orbit_daily_case_day
        # Recycle workers after each day: an un-closed WACCMAtmosphere per
        # observation across a long-lived worker was a real source of OOM
        # kills in production L2 runs. See cesm_hawc.dispatch docstring.
        max_tasks_per_child = 1 if o.run_l2 else None

    if dry_run:
        log.info("[dry-run] would run %d day(s), output to %s", len(jobs), out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)
    log.info("Pre-warming calibration database and Mie databases...")
    warm_calibration_database()
    warm_mode_databases()

    results = run_jobs(worker_fn, jobs, n_workers, max_tasks_per_child=max_tasks_per_child,
                        on_result=lambda r: log.info(r))
    _report(results, "days")


# ---------------------------------------------------------------------------
# run: orbit-file
# ---------------------------------------------------------------------------

def _run_orbit_file(orbit_path: str, h2_bg_path: str, h2_inj_path: str | None,
                     out_dir: str, across_indices: list[int], time_stride: int,
                     alt_grid_m: np.ndarray, wavelengths_nm: np.ndarray) -> str:
    import xarray as xr

    from cesm_hawc.constituents import build_waccm_constituents
    from cesm_hawc.noise import default_noise_model
    from cesm_hawc.waccm import WACCMAtmosphere

    orbit_name = os.path.splitext(os.path.basename(orbit_path))[0]
    try:
        from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator

        orbit = xr.open_dataset(orbit_path, decode_times=True)
        n_time, n_across = orbit.sizes["time"], orbit.sizes["across"]
        idxs = across_indices or list(range(n_across))
        times = list(range(0, n_time, time_stride))

        waccm_bg = WACCMAtmosphere(h2_bg_path, alt_grid_km=alt_grid_m / 1e3)
        waccm_inj = WACCMAtmosphere(h2_inj_path, alt_grid_km=alt_grid_m / 1e3) if h2_inj_path else None
        simulator = IdealALISimulator()
        noise_model = default_noise_model()

        l2_bg_list, l2_inj_list, meta = [], [], []
        for t_idx in times:
            obs_lat = float(orbit["observer_latitude"].isel(time=t_idx))
            obs_lon = float(orbit["observer_longitude"].isel(time=t_idx))
            obs_alt = float(orbit["observer_altitude"].isel(time=t_idx))
            obs_time = pd.Timestamp(orbit["time"].isel(time=t_idx).values)

            for ac_idx in idxs:
                lat = float(orbit["latitude"].isel(time=t_idx, along=0, across=ac_idx))
                lon = float(orbit["longitude"].isel(time=t_idx, along=0, across=ac_idx))
                if not np.isfinite(lat) or not np.isfinite(lon):
                    continue

                profiles_bg = waccm_bg.get_column_profiles(lat, lon, 0)
                sim_input = {
                    "tangent_latitude": lat, "tangent_longitude": lon,
                    "observer_latitude": obs_lat, "observer_longitude": obs_lon,
                    "observer_altitude": obs_alt, "altitude_grid": alt_grid_m,
                    "polarization_states": ["I", "dolp"], "sample_wavelengths": wavelengths_nm,
                    "time": obs_time, "l1b_cfg": {"noise_model": noise_model},
                    "constituents": build_waccm_constituents(profiles_bg, alt_grid_m),
                }
                data_bg = simulator.run(["l2", "l1b"], sim_input)
                l2_bg_list.append(data_bg["l2"])

                if waccm_inj is not None:
                    profiles_inj = waccm_inj.get_column_profiles(lat, lon, 0)
                    data_inj = simulator.run(
                        ["l2", "l1b"],
                        {**sim_input, "constituents": build_waccm_constituents(profiles_inj, alt_grid_m)},
                    )
                    l2_inj_list.append(data_inj["l2"])

                meta.append({"time": obs_time, "across": ac_idx, "lat": lat, "lon": lon})

        os.makedirs(out_dir, exist_ok=True)

        def _save(l2_list, tag):
            curtain = xr.concat(l2_list, dim="obs").assign_coords(
                obs_time=("obs", [r["time"] for r in meta]),
                across_idx=("obs", [r["across"] for r in meta]),
                lat=("obs", [r["lat"] for r in meta]), lon=("obs", [r["lon"] for r in meta]),
            )
            curtain.to_netcdf(os.path.join(out_dir, f"{orbit_name}_{tag}.nc"))

        _save(l2_bg_list, "l2_bg")
        if l2_inj_list:
            _save(l2_inj_list, "l2_inj")
        orbit.close()
        return f"OK   {orbit_name}  ({len(meta)} obs)"
    except Exception:
        log.error("[%s] FAILED:\n%s", orbit_name, traceback.format_exc())
        return f"FAIL {orbit_name}"


def _run_orbit_file_cmd(cfg: CesmHawcConfig, out_dir_override, n_workers_override, dry_run) -> None:
    from cesm_hawc.calibration import warm_calibration_database
    from cesm_hawc.dispatch import run_jobs

    if cfg.orbit_real is None:
        sys.exit("run --mode orbit-file requires [orbit_real] in config.toml")
    orb, ins = cfg.orbit_real, cfg.instrument
    out_dir = out_dir_override or orb.out_dir
    n_workers = n_workers_override or orb.n_workers
    alt_grid_m = ins.altitude_grid_m()
    wavelengths_nm = np.array(ins.wavelengths_nm)

    raw_jobs = _build_orbit_file_jobs(orb, out_dir)
    jobs = [(op, bg, inj, day_out, orb.across_indices, orb.time_stride, alt_grid_m, wavelengths_nm)
            for op, bg, inj, day_out in raw_jobs]

    if dry_run:
        log.info("[dry-run] would run %d orbit file(s), output to %s", len(jobs), out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)
    log.info("Pre-warming calibration database...")
    warm_calibration_database()

    results = run_jobs(_run_orbit_file, jobs, n_workers, on_result=lambda r: log.info(r))
    _report(results, "orbit files")


# ---------------------------------------------------------------------------
# Shared helpers + entry point
# ---------------------------------------------------------------------------

def _warm_mode_databases_if_available() -> None:
    """Pre-warm the mode-specific Mie databases before ``save-inputs``
    dispatches to a worker pool, same race-condition concern as `run`'s
    pre-warm calls -- but non-fatal here, since ``save-inputs`` must keep
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


_SAVE_INPUTS_DISPATCH = {
    "single": lambda cfg, out, workers, dry, profiles_only: _save_inputs_single(
        cfg, out, dry, profiles_only),
    "batch": _save_inputs_batch,
    "orbit-track": _save_inputs_orbit_track,
    "orbit-file": _save_inputs_orbit_file_cmd,
}
_RUN_DISPATCH = {
    "single": lambda cfg, out, workers, dry: _run_single(cfg, out, dry),
    "batch": _run_batch,
    "orbit-track": _run_orbit_track,
    "orbit-file": _run_orbit_file_cmd,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cesm-hawc",
        description="Feed CESM2/WACCM SAI output into the HAWC ALI simulator.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default="config.toml", help="Path to config.toml")
        sp.add_argument("--mode", required=True,
                         choices=["single", "batch", "orbit-track", "orbit-file"])
        sp.add_argument("--out-dir", default=None, help="Override config's out_dir")
        sp.add_argument("--n-workers", type=int, default=None, help="Override config's n_workers")
        sp.add_argument("--dry-run", action="store_true",
                         help="Print the job count without running anything")

    save_inputs_p = sub.add_parser(
        "save-inputs",
        help="Extract and save WACCM column inputs (no sasktran2/hawcsimulator needed)",
    )
    add_common(save_inputs_p)
    save_inputs_p.add_argument(
        "--profiles-only", action="store_true",
        help="Skip computing simulator constituents even if sasktran2 is available "
             "(save only the raw WACCM profile fields)",
    )

    add_common(sub.add_parser(
        "run",
        help="Run the full forward model / L2 retrieval end to end (requires the [sim] extra)",
    ))
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
        _SAVE_INPUTS_DISPATCH[args.mode](cfg, args.out_dir, args.n_workers, args.dry_run,
                                          args.profiles_only)
    elif args.command == "run":
        _require_sim_deps()
        _RUN_DISPATCH[args.mode](cfg, args.out_dir, args.n_workers, args.dry_run)


if __name__ == "__main__":
    main()
