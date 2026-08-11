#!/usr/bin/env python
"""
examples/quickstart.py
=======================
End-to-end walkthrough of the cesm-hawc pipeline on one CESM/WACCM h2 file
+ one HAWC orbit file: extract a column, save it two ways (raw profile vs.
simulator-ready constituents input), run the forward model + full L2
retrieval, and compare the retrieval against the known truth.

Requires the [sim] extra:
    pip install cesm-hawc[sim]

Usage
-----
    python examples/quickstart.py case.cam.h2.YYYY-MM-DD-SSSSS.nc orbit_file.nc \\
        [--out-dir ./quickstart_output] [--obs-index N] [--date YYYY-MM-DD]

The h2 file's date is parsed from its filename and used as the simulation
date; the orbit file supplies the real time-of-day and tangent-point/
observer geometry for that date (same day-of-year-offset convention as
cesm_hawc.orbit_files). By default this scans the orbit file for the first
daytime observation at CENTER_PIXEL; pass --obs-index to pick a specific one.

Writes to --out-dir:
    profiles_<date>.nc            raw WACCM column (WACCMAtmosphere.save_column_profiles)
    constituents_input_<date>.nc  simulator-ready input (cesm_hawc.save_inputs)
    forward_<date>.nc             forward model output (front_end_radiance/l1b)
    l2_retrieval_<date>.nc        full L2 retrieval output (cesm-hawc's own code path)
    l2_retrieval_direct_<date>.nc L2 retrieval rebuilt from the saved constituents-input
                                   file using only native sasktran2 calls (no
                                   cesm_hawc.constituents), same geometry -- a round-trip
                                   check that the saved file reproduces the same result
    comparison_<date>.txt         truth vs. both retrievals + convergence
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

import cesm_hawc
from cesm_hawc.calibration import warm_calibration_database
from cesm_hawc.constituents import warm_mode_databases
from cesm_hawc.convergence import extract_l2_native_diagnostics, parse_scipy_convergence
from cesm_hawc.orbit_files import extract_observations, l1b_image_to_dataset
from cesm_hawc.outputs import write_text_summary
from cesm_hawc.save_inputs import save_column_inputs
from cesm_hawc.simulation import DEFAULT_PRODUCTS, run_ali_simulation_from_profiles
from cesm_hawc.waccm import WACCMAtmosphere

ORBIT_EPOCH = pd.Timestamp("2019-08-01")   # orbit file's own time origin
CENTER_PIXEL = 256                         # cross-track index used as tangent point
WAVELENGTHS_NM = np.array([470.0, 745.0, 1020.0])
ALT_GRID_M = np.arange(0.0, 65001.0, 1000.0)
REFERENCE_WAVELENGTH_NM = 745.0            # matches ExtinctionScatterer's reference


def h2_filename_date(h2_path: str) -> pd.Timestamp:
    m = re.search(r"(\d{4}-\d{2}-\d{2})-\d+\.nc$", os.path.basename(h2_path))
    if not m:
        sys.exit(
            f"Could not parse a YYYY-MM-DD date from h2 filename "
            f"'{os.path.basename(h2_path)}' -- expected pattern "
            f"'*.cam.h2.YYYY-MM-DD-SSSSS.nc'. Pass --date explicitly instead."
        )
    return pd.Timestamp(m.group(1))


def find_daytime_observation(waccm, simulator, orbit_path, sim_date, obs_index):
    """Scan the orbit file's along-track observations for the first
    daytime one (or use --obs-index directly), running the simulator on
    each candidate until one doesn't raise the night-side SZA error."""
    observations = extract_observations(
        [orbit_path], sim_date, cadence_s=60.0, center_pixel=CENTER_PIXEL, epoch=ORBIT_EPOCH,
    )
    print(f"  {len(observations)} along-track observations, simulation date {sim_date.date()}")
    if not observations:
        sys.exit(f"No observations found in {orbit_path} for {sim_date.date()}")

    candidates = [observations[obs_index]] if obs_index is not None else observations

    for cand in candidates:
        profiles = waccm.get_column_profiles(cand["lat"], cand["lon"], time_index=0)
        sim_geometry = {
            "tangent_latitude": cand["lat"],
            "tangent_longitude": cand["lon"],
            "observer_latitude": cand["observer_lat"],
            "observer_longitude": cand["observer_lon"],
            "observer_altitude": cand["observer_alt"],
            "altitude_grid": ALT_GRID_M,
            "polarization_states": ["I", "dolp"],
            "sample_wavelengths": WAVELENGTHS_NM,
            "time": cand["time"],
            # SZA/SAA omitted -> computed automatically from time + observer position
        }
        print(f"Trying obs at time={cand['time']}, lat={cand['lat']:.2f}, "
              f"lon={cand['lon']:.2f} ...")
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                data, true_ext = run_ali_simulation_from_profiles(
                    profiles, ALT_GRID_M, sim_geometry, simulator=simulator,
                    products=DEFAULT_PRODUCTS, return_extinction=True,
                    truth_wavelengths_nm=WAVELENGTHS_NM,
                )
            return cand, data, true_ext, captured.getvalue()
        except ValueError as e:
            if "SZA" in str(e) and "greater than the allowed maximum" in str(e):
                if obs_index is not None:
                    sys.exit(
                        f"Observation at index {obs_index} is night-side (SZA too "
                        f"large). Try a different --obs-index, or omit it to "
                        f"auto-scan for a daytime observation."
                    )
                continue
            raise

    sys.exit(
        f"No daytime observation found in {orbit_path} for simulation date "
        f"{sim_date.date()} (all {len(candidates)} candidates were night-side). "
        f"Try a different orbit file or --date."
    )


def run_simulator_direct(constituents_path: str, sim_geometry: dict):
    """
    Re-read the just-saved constituents-input file and rebuild the
    simulator's aerosol/gas constituents using only native sasktran2 calls. Then run
    the simulator with the exact same ``sim_geometry`` the cesm-hawc-native
    run used. Mirrors the README's "Consuming saved inputs externally"
    example.
    
    """
    import sasktran2 as sk
    from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator

    ds = xr.open_dataset(constituents_path)
    assert ds.attrs["includes_constituents"], f"{constituents_path} was saved with --profiles-only"
    alt_m = ds["altitude_m"].values

    def mode_constituent(mode: str):
        mode_width = ds.attrs[f"mode_width_{'accum' if mode == 'aerosol_accum' else 'coarse'}"]
        mode_db = sk.database.MieDatabase(
            sk.mie.distribution.LogNormalDistribution().freeze(mode_width=mode_width),
            sk.mie.refractive.H2SO4(),  # ds.attrs["mie_refractive_index"]
            ds.attrs["mie_wavelength_grid_nm"],
            median_radius=ds.attrs["mie_median_radius_grid_nm"],
        )
        return sk.constituent.ExtinctionScatterer(
            mode_db, altitudes_m=alt_m,
            extinction_per_m=ds[f"{mode}_reference_extinction_per_m"].values,
            extinction_wavelength_nm=ds.attrs["extinction_reference_wavelength_nm"],
            median_radius=ds[f"{mode}_median_radius_nm"].values,
        )

    constituents = {
        "o3": sk.constituent.VMRAltitudeAbsorber(sk.optical.O3DBM(), altitudes_m=alt_m, vmr=ds["vmr_o3"].values),
        "no2": sk.constituent.VMRAltitudeAbsorber(sk.optical.NO2Vandaele(), altitudes_m=alt_m, vmr=ds["vmr_no2"].values),
        "aerosol_accum": mode_constituent("aerosol_accum"),
        "aerosol_coarse": mode_constituent("aerosol_coarse"),
    }

    data = IdealALISimulator().run(
        ["l2", "front_end_radiance", "l1b"], {**sim_geometry, "constituents": constituents}
    )
    return data["l2"]


def compare_retrieved_vs_truth(l2, true_ext, l2_direct=None) -> list[str]:
    """Interpolate the WACCM-derived truth extinction (summed across both
    MAM4 modes at the 745 nm reference wavelength) onto L2's own altitude
    grid and diff it against the retrieved extinction. If l2_direct is
    given (the independently-reconstructed retrieval from
    run_simulator_direct()), it's added as a third column so you
    can see both retrievals against truth and against each other."""
    ref_idx = int(np.argmin(np.abs(WAVELENGTHS_NM - REFERENCE_WAVELENGTH_NM)))
    truth_ext = (true_ext["aerosol_accum_extinction_per_m"][ref_idx]
                 + true_ext["aerosol_coarse_extinction_per_m"][ref_idx])  # [atm altitude]

    retrieved_ext = l2["stratospheric_aerosol_extinction_per_m"]
    l2_alt_m = retrieved_ext.altitude.values
    truth_on_l2_grid = np.interp(l2_alt_m, ALT_GRID_M, truth_ext)
    residual = retrieved_ext.values - truth_on_l2_grid

    peak_truth_idx = int(np.argmax(truth_ext))
    peak_retrieved_idx = int(np.argmax(retrieved_ext.values))

    lines = [
        "Retrieved vs. truth extinction (745 nm reference)",
        "---------------------------------------------------",
        f"  Peak truth extinction:                 {truth_ext[peak_truth_idx]:.4e} m^-1 "
        f"at {ALT_GRID_M[peak_truth_idx] / 1e3:.1f} km",
        f"  Peak retrieved extinction (cesm-hawc): {retrieved_ext.values[peak_retrieved_idx]:.4e} m^-1 "
        f"at {l2_alt_m[peak_retrieved_idx] / 1e3:.1f} km",
        f"  Mean |residual| (cesm-hawc - truth):   {np.mean(np.abs(residual)):.4e} m^-1",
        f"  Max  |residual| (cesm-hawc - truth):   {np.max(np.abs(residual)):.4e} m^-1",
    ]

    direct_on_l2_grid = None
    if l2_direct is not None:
        direct_ext = l2_direct["stratospheric_aerosol_extinction_per_m"]
        direct_on_l2_grid = np.interp(l2_alt_m, direct_ext.altitude.values, direct_ext.values)
        direct_residual = direct_on_l2_grid - truth_on_l2_grid
        cesm_hawc_vs_direct = retrieved_ext.values - direct_on_l2_grid
        peak_direct_idx = int(np.argmax(direct_on_l2_grid))
        lines += [
            f"  Peak retrieved extinction (direct):    {direct_on_l2_grid[peak_direct_idx]:.4e} m^-1 "
            f"at {l2_alt_m[peak_direct_idx] / 1e3:.1f} km",
            f"  Mean |residual| (direct - truth):      {np.mean(np.abs(direct_residual)):.4e} m^-1",
            f"  Mean |cesm-hawc - direct|:              {np.mean(np.abs(cesm_hawc_vs_direct)):.4e} m^-1 "
            f"(should be ~0 if the saved file round-trips correctly)",
        ]

    lines.append("")
    if direct_on_l2_grid is not None:
        lines.append("  altitude_km  truth_per_m  cesm_hawc_per_m  direct_per_m  "
                      "cesm_hawc_residual  direct_residual")
        for alt, truth, ret, direct in zip(l2_alt_m, truth_on_l2_grid, retrieved_ext.values, direct_on_l2_grid):
            lines.append(f"  {alt / 1e3:10.1f}  {truth:11.4e}  {ret:15.4e}  {direct:12.4e}  "
                         f"{ret - truth:18.4e}  {direct - truth:15.4e}")
    else:
        lines.append("  altitude_km  retrieved_per_m  truth_per_m  residual_per_m")
        for alt, ret, truth, res in zip(l2_alt_m, retrieved_ext.values, truth_on_l2_grid, residual):
            lines.append(f"  {alt / 1e3:10.1f}  {ret:14.4e}  {truth:11.4e}  {res:14.4e}")
    return lines


def main(h2_path: str, orbit_path: str, out_dir: str,
         obs_index: int | None, date_override: str | None) -> None:
    cesm_hawc.configure_environment()

    from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator

    sim_date = pd.Timestamp(date_override) if date_override else h2_filename_date(h2_path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {h2_path} ...")
    waccm = WACCMAtmosphere(h2_path, alt_grid_km=ALT_GRID_M / 1e3)

    print("Pre-warming calibration database and Mie databases...")
    warm_calibration_database()
    warm_mode_databases()
    simulator = IdealALISimulator()

    print(f"Reading orbit geometry from {orbit_path} ...")
    obs, data, true_ext, captured_stdout = find_daytime_observation(
        waccm, simulator, orbit_path, sim_date, obs_index
    )
    date_str = obs["time"].strftime("%Y-%m-%d")
    print(f"\nUsed observation: time={obs['time']}, lat={obs['lat']:.2f}, "
          f"lon={obs['lon']:.2f}, observer_alt={obs['observer_alt'] / 1e3:.0f} km")

    # 1. Model profiles (base tier -- no simulator constituents)
    profiles_path = os.path.join(out_dir, f"profiles_{date_str}.nc")
    waccm.save_column_profiles(obs["lat"], obs["lon"], profiles_path, time_index=0)

    # 2. Simulator constituents input (adds per-mode extinction/radius +
    #    Mie build params on top of the same column)
    constituents_path = os.path.join(out_dir, f"constituents_input_{date_str}.nc")
    save_column_inputs(waccm, obs["lat"], obs["lon"], constituents_path, 0,
                        ALT_GRID_M, WAVELENGTHS_NM, obs_time=obs["time"])

    # 3. Forward model output (front_end_radiance/l1b), with truth
    #    extinction attached on both the native and instrument grids
    forward_path = os.path.join(out_dir, f"forward_{date_str}.nc")
    l1b_ds = l1b_image_to_dataset(data["l1b"], WAVELENGTHS_NM, true_ext, ALT_GRID_M)
    l1b_ds = l1b_ds.assign_coords(lat=obs["lat"], lon=obs["lon"], time=str(obs["time"]))
    l1b_ds.to_netcdf(forward_path)
    print(f"Saved forward output       -> {forward_path}")

    # 4. Full L2 retrieval output
    l2 = data["l2"]
    l2_path = os.path.join(out_dir, f"l2_retrieval_{date_str}.nc")
    l2_ds = l2.assign_coords(lat=obs["lat"], lon=obs["lon"], time=str(obs["time"]))
    l2_ds.to_netcdf(l2_path)
    print(f"Saved L2 retrieval         -> {l2_path}")

    diag = parse_scipy_convergence(captured_stdout)
    native_diag = extract_l2_native_diagnostics(l2)
    print(f"  converged: {diag['converged']}  (reason: {diag['termination_reason']}, "
          f"nfev: {diag['n_function_evaluations']})")
    print(f"  num_iterations (native): {native_diag['l2_num_iterations']}  "
          f"final cost: {native_diag['l2_final_cost']}")

    # 4.5. Independent retrieval: rebuild constituents from the saved file
    #      using only native sasktran2 calls (no cesm_hawc.constituents),
    #      re-run with the SAME geometry as step 4 -- a round-trip check
    #      that the saved file reproduces the same simulator inputs.
    print("\nRunning independently of cesm_hawc (native sasktran2, from the saved file) ...")
    sim_geometry = {
        "tangent_latitude": obs["lat"],
        "tangent_longitude": obs["lon"],
        "observer_latitude": obs["observer_lat"],
        "observer_longitude": obs["observer_lon"],
        "observer_altitude": obs["observer_alt"],
        "altitude_grid": ALT_GRID_M,
        "polarization_states": ["I", "dolp"],
        "sample_wavelengths": WAVELENGTHS_NM,
        "time": obs["time"],
    }
    l2_direct = run_simulator_direct(constituents_path, sim_geometry)
    l2_direct_path = os.path.join(out_dir, f"l2_retrieval_direct_{date_str}.nc")
    l2_direct_ds = l2_direct.assign_coords(lat=obs["lat"], lon=obs["lon"], time=str(obs["time"]))
    l2_direct_ds.to_netcdf(l2_direct_path)
    print(f"Saved independent L2 retrieval -> {l2_direct_path}")

    # 5. Compare both retrievals vs. truth (and vs. each other)
    lines = [
        f"Observation: time={obs['time']}, lat={obs['lat']:.2f}, lon={obs['lon']:.2f}",
        f"Convergence (cesm-hawc): {diag['converged']} ({diag['termination_reason']}, "
        f"{diag['n_function_evaluations']} function evaluations)",
        f"L2 native diagnostics (cesm-hawc): num_iterations={native_diag['l2_num_iterations']}, "
        f"cost={native_diag['l2_final_cost']}",
        "",
    ]
    lines += compare_retrieved_vs_truth(l2, true_ext, l2_direct=l2_direct)
    comparison_path = write_text_summary(lines, out_dir, f"comparison_{date_str}.txt")

    print(f"\nDone. Outputs in {out_dir}:")
    for p in (profiles_path, constituents_path, forward_path, l2_path, l2_direct_path, comparison_path):
        print(f"  {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cesm-hawc quickstart: h2 file + orbit file -> profiles -> "
                     "constituents input -> forward model -> L2 retrieval -> compare."
    )
    parser.add_argument("h2_path", help="CESM/WACCM h2 NetCDF file, e.g. "
                                         "*.cam.h2.YYYY-MM-DD-SSSSS.nc")
    parser.add_argument("orbit_path", help="Orbit NetCDF file (real ground track)")
    parser.add_argument("--out-dir", default="./quickstart_output",
                         help="Directory to write all outputs into. "
                              "Default: ./quickstart_output")
    parser.add_argument("--obs-index", type=int, default=None,
                         help="Use a specific along-track observation index "
                              "instead of auto-scanning for the first daytime one.")
    parser.add_argument("--date", type=str, default=None,
                         help="Override the simulation date instead of parsing "
                              "it from the h2 filename (YYYY-MM-DD).")
    args = parser.parse_args()
    main(args.h2_path, args.orbit_path, args.out_dir, args.obs_index, args.date)
