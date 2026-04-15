#!/usr/bin/env python
"""
run_simulation.py
=================
Run the HAWC ALI simulator on a WACCM background and injection scenario.

Edit the CONFIGURATION section below, then run:

    python run_simulation.py

or submit via SLURM:

    sbatch slurm/submit.sh
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

from cesm_hawc import WACCMAtmosphere, build_waccm_constituents
from cesm_hawc.waccm import R_DRY, R_H2O, hybrid_to_pressure, pressure_to_altitude
from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator

# ── CONFIGURATION ──────────────────────────────────────────────────────────

WACCM_BACKGROUND = "/path/to/background.cam.h0.YYYY-MM.nc"
WACCM_INJECTION  = "/path/to/injection.cam.h0.YYYY-MM.nc"   # set None to skip
TIME_IDX         = 0   # time slice index within the file

# Observation geometry — match your SO₂ injection latitude
TANGENT_LAT = 30.6    # degrees
TANGENT_LON = 180.0   # degrees
SZA_DEG     = 60.0
SAA_DEG     = 0.0
OBS_TIME    = "2035-02-01T12:00:00Z"

# ALI sample wavelengths [nm]
# Use the 3-channel quickstart set for development.
# Extend to the full suite [470,525,745,1020,1230,1450,1500] for production.
ALI_WAVELENGTHS = np.array([470.0, 745.0, 1020.0])

# Altitude grid [m]
ALT_GRID_M = np.arange(0.0, 65001.0, 1000.0)

# Output directory
OUT_DIR = os.path.expanduser("~/results/hawc_ali/")

# ── END CONFIGURATION ──────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    simulator = IdealALISimulator()
    sim_input = {
        "tangent_latitude":           TANGENT_LAT,
        "tangent_longitude":          TANGENT_LON,
        "tangent_solar_zenith_angle":  SZA_DEG,
        "tangent_solar_azimuth_angle": SAA_DEG,
        "altitude_grid":               ALT_GRID_M,
        "polarization_states":         ["I", "dolp"],
        "sample_wavelengths":          ALI_WAVELENGTHS,
        "time":                        pd.Timestamp(OBS_TIME),
    }

    # ── Background ──────────────────────────────────────────────────────
    print("Loading background WACCM file...")
    waccm_bg    = WACCMAtmosphere(WACCM_BACKGROUND, alt_grid_km=ALT_GRID_M / 1e3)
    profiles_bg = waccm_bg.get_column_profiles(TANGENT_LAT, TANGENT_LON, TIME_IDX)

    print("Running background simulation...")
    data_bg = simulator.run(
        ["l2", "sk2_atmosphere"],
        {**sim_input,
         "constituents": build_waccm_constituents(profiles_bg, ALT_GRID_M)},
    )
    print(f"  Converged: {data_bg['l2']['num_iterations'].values} iterations, "
          f"final cost {data_bg['l2']['cost'].values:.4f}")

    data_bg["l2"].to_netcdf(os.path.join(OUT_DIR, "l2_background.nc"))
    _save_cesm_extinction(waccm_bg, TANGENT_LAT, TANGENT_LON, TIME_IDX,
                          ALT_GRID_M, OUT_DIR, "background")

    # ── Injection ────────────────────────────────────────────────────────
    data_inj, waccm_inj = None, None
    if WACCM_INJECTION is not None:
        print("Loading injection WACCM file...")
        waccm_inj    = WACCMAtmosphere(WACCM_INJECTION, alt_grid_km=ALT_GRID_M / 1e3)
        profiles_inj = waccm_inj.get_column_profiles(TANGENT_LAT, TANGENT_LON, TIME_IDX)

        print("Running injection simulation...")
        data_inj = simulator.run(
            ["l2", "sk2_atmosphere"],
            {**sim_input,
             "constituents": build_waccm_constituents(profiles_inj, ALT_GRID_M)},
        )
        print(f"  Converged: {data_inj['l2']['num_iterations'].values} iterations, "
              f"final cost {data_inj['l2']['cost'].values:.4f}")

        data_inj["l2"].to_netcdf(os.path.join(OUT_DIR, "l2_injection.nc"))
        _save_cesm_extinction(waccm_inj, TANGENT_LAT, TANGENT_LON, TIME_IDX,
                              ALT_GRID_M, OUT_DIR, "injection")

    # ── Summary ──────────────────────────────────────────────────────────
    # Figures are generated separately — no need to rerun the simulation
    # to adjust a plot.  Run:  python scripts/plot_results.py
    burden_bg = waccm_bg.sulfate_column_burden(TANGENT_LAT, TANGENT_LON, TIME_IDX)
    print_summary(burden_bg, data_bg, data_inj, waccm_inj, OUT_DIR)


def _save_cesm_extinction(waccm_obj: "WACCMAtmosphere", lat: float, lon: float,
                          time_index: int, alt_m: np.ndarray,
                          out_dir: str, tag: str) -> None:
    """
    Extract CESM extinction profiles directly from the h0 file and save
    to NetCDF alongside the L2 output.

    Uses EXTINCTdn (550 nm), EXTINCTUVdn (350 nm), and EXTINCTNIRdn (1020 nm)
    — CESM's own internally computed aerosol extinction on model levels,
    including all aerosol species. This is strictly preferable to computing
    extinction analytically from N and r because it reflects what the model
    actually computed.

    EXTINCTNIRdn at 1020 nm is a direct ALI wavelength match and requires
    no wavelength correction for comparison with retrieved extinction.
    """
    col = waccm_obj.ds.isel(time=time_index).sel(
        lat=lat, lon=lon, method="nearest"
    )

    # Reconstruct altitude grid from hybrid coordinates
    p0   = float(waccm_obj.ds["P0"].values) if "P0" in waccm_obj.ds else 100_000.0
    ps   = float(col["PS"].values)
    T    = col["T"].values
    Q    = col["Q"].values
    hyam = col["hyam"].values
    hybm = col["hybm"].values

    pressure = hybrid_to_pressure(hyam, hybm, p0, ps)
    T_v      = T * (1.0 + (R_H2O / R_DRY - 1.0) * Q)
    altitude = pressure_to_altitude(pressure, T_v, ps, waccm_obj.z_surface)

    def interp_ext(varname):
        if varname not in col:
            return None
        vals = np.maximum(col[varname].values, 0.0)
        idx  = np.argsort(altitude)
        f    = interp1d(altitude[idx], vals[idx], kind="linear",
                        bounds_error=False, fill_value=0.0)
        return f(alt_m)

    data_vars = {}
    for varname, label in [
        ("EXTINCTdn",    "ext_550nm"),
        ("EXTINCTUVdn",  "ext_350nm"),
        ("EXTINCTNIRdn", "ext_1020nm"),
    ]:
        interped = interp_ext(varname)
        if interped is not None:
            data_vars[label] = ("altitude_m", interped)

    if not data_vars:
        print(f"  Warning: no EXTINCT* variables found in file — skipping cesm_extinction_{tag}.nc")
        return

    ds = xr.Dataset(
        data_vars,
        coords={"altitude_m": alt_m},
        attrs={
            "description": "CESM aerosol extinction profiles from EXTINCTdn, "
                           "EXTINCTUVdn, EXTINCTNIRdn — all aerosol species",
            "wavelengths_nm": "350, 550, 1020",
            "source_variable": "EXTINCTdn / EXTINCTUVdn / EXTINCTNIRdn",
        }
    )
    path = os.path.join(out_dir, f"cesm_extinction_{tag}.nc")
    ds.to_netcdf(path)
    print(f"  CESM extinction saved to {path}")


def print_summary(burden_bg, data_bg, data_inj, waccm_inj, out_dir):
    """Print and save a text summary of key results."""
    lines = [
        f"TANGENT_LAT:       {TANGENT_LAT}",
        f"TANGENT_LON:       {TANGENT_LON}",
        f"TIME_IDX:          {TIME_IDX}",
        "",
        "Background stratospheric sulfate (15–35 km):",
    ]
    for k, v in burden_bg.items():
        lines.append(f"  {k:25s}: {v}" if isinstance(v, str)
                     else f"  {k:25s}: {v:.4g}")

    if data_inj is not None and waccm_inj is not None:
        burden_inj = waccm_inj.sulfate_column_burden(
            TANGENT_LAT, TANGENT_LON, TIME_IDX)

        ext_bg  = data_bg["l2"]["stratospheric_aerosol_extinction_per_m"]
        ext_inj = data_inj["l2"]["stratospheric_aerosol_extinction_per_m"]
        r_bg    = data_bg["l2"]["stratospheric_aerosol_median_radius"]
        r_inj   = data_inj["l2"]["stratospheric_aerosol_median_radius"]
        strat   = ext_bg.altitude.values > 15000

        peak_ext  = float((ext_inj - ext_bg).values[strat].max())
        peak_r    = float((r_inj - r_bg).values[strat].max())
        d_burden  = burden_inj["burden_mg_m2"] - burden_bg["burden_mg_m2"]

        lines += [
            "",
            "Injection stratospheric sulfate (15–35 km):",
        ]
        for k, v in burden_inj.items():
            lines.append(f"  {k:25s}: {v}" if isinstance(v, str)
                         else f"  {k:25s}: {v:.4g}")
        lines += [
            "",
            f"Peak extinction anomaly (>15 km):  {peak_ext:.3e} m⁻¹",
            f"Peak radius anomaly (>15 km):      {peak_r:.1f} nm",
            f"Δ SO₄ burden:                      {d_burden:.3f} mg m⁻²",
        ]

    summary = "\n".join(lines)
    print(summary)
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {path}")


if __name__ == "__main__":
    main()
