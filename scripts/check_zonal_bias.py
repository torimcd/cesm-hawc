#!/usr/bin/env python
"""
check_zonal_bias.py

Tests whether sampling only the daylit hemisphere at 00:00 UTC introduces
bias in stratospheric aerosol extinction statistics relative to the full
zonal mean.

Uses the ALI Mie database (via cesm_hawc.constituents.aerosol_median_radius_db)
for wavelength-dependent extinction — identical to what the simulator uses.

For each input h2 file:
  1. Loads a spatial subsample of grid points (every Nth lat/lon)
  2. Computes extinction profiles at ALI wavelengths using the Mie database
  3. Identifies daylit grid points at 00:00 UTC (SZA < 90 deg)
  4. Compares zonal mean extinction from full vs daylit-only sampling
  5. Computes fractional bias: (daylit - full) / full

Produces:
  fig_zonal_bias_470nm.png   — fractional bias vs altitude per lat band
  fig_zonal_bias_745nm.png
  fig_zonal_bias_1020nm.png
  fig_zonal_bias_map_745nm.png  — 2D lat x altitude bias map
  zonal_bias_summary.txt        — max bias per altitude range

Usage:
    python check_zonal_bias.py \\
        --files /path/to/sai_background_2030_001/atm/hist/*.cam.h2.2030-0[1-3]-*.nc \\
        --out_dir ./figures/zonal_bias \\
        --lat 30.6 \\
        --every_nth_file 10 \\
        --spatial_stride 4
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import RegularGridInterpolator

from cesm_hawc.waccm import WACCMAtmosphere
from cesm_hawc.constituents import aerosol_median_radius_db

# ---------------------------------------------------------------------------
# Mie database interpolator — loaded once at module level
# ---------------------------------------------------------------------------

print("Loading Mie database...", flush=True)
_mie_db   = aerosol_median_radius_db()
_xs_total = _mie_db._database["xs_total"]          # (wavelength_nm, median_radius)
_wl_nm    = _xs_total.coords["wavelength_nm"].values.astype(float)
_r_nm     = _xs_total.coords["median_radius"].values
_xs       = _xs_total.values                        # m² per particle

_mie_interp = RegularGridInterpolator(
    (_wl_nm, _r_nm), _xs,
    method="linear", bounds_error=False, fill_value=0.0
)
print(f"  Mie database loaded: {len(_wl_nm)} wavelengths x {len(_r_nm)} radii",
      flush=True)
print(f"  Wavelengths: {_wl_nm.tolist()} nm", flush=True)
print(f"  Radius range: {_r_nm.min():.0f} – {_r_nm.max():.0f} nm", flush=True)

ALT_GRID_M = np.arange(500, 65001, 1000)   # 65 levels, 0.5–64.5 km
ALT_GRID_KM = ALT_GRID_M / 1e3


# ---------------------------------------------------------------------------
# Mie extinction from profiles
# ---------------------------------------------------------------------------

def extinction_profile(profiles, wavelength_nm):
    """
    Compute total aerosol extinction [m^-1] at one wavelength from
    WACCMAtmosphere column profiles, using the ALI Mie database.

    Sums accumulation (a1) and coarse (a3) modes.
    """
    ext = np.zeros(len(ALT_GRID_M))

    for N_key, r_key in [
        ("sulfate_a1_N_cm3", "sulfate_a1_r_um"),
        ("sulfate_a3_N_cm3", "sulfate_a3_r_um"),
    ]:
        N_cm3 = profiles[N_key]   # cm^-3, shape (65,)
        r_um  = profiles[r_key]   # um,    shape (65,)
        r_nm  = r_um * 1e3        # um -> nm

        # clamp to Mie database range
        r_nm_clamped = np.clip(r_nm, _r_nm.min(), _r_nm.max())

        # query points: (wavelength, radius) for each altitude level
        pts = np.column_stack([
            np.full(len(r_nm_clamped), wavelength_nm),
            r_nm_clamped
        ])
        xs = _mie_interp(pts)   # m² per particle

        # extinction [m^-1] = N [m^-3] * xs [m²]
        N_m3 = N_cm3 * 1e6
        valid = (N_cm3 > 0) & (r_nm > _r_nm.min())
        ext += np.where(valid, N_m3 * xs, 0.0)

    return ext


# ---------------------------------------------------------------------------
# Solar zenith angle
# ---------------------------------------------------------------------------

def solar_zenith_angle(lat_deg, lon_deg, doy, utc_hour=0.0):
    """
    Compute solar zenith angle [degrees] using simple analytical formula.
    """
    lat  = np.deg2rad(lat_deg)
    lon  = np.deg2rad(lon_deg)
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360 / 365 * (doy - 81))))
    lst  = utc_hour + np.rad2deg(lon) / 15.0
    ha   = np.deg2rad(15.0 * (lst - 12.0))
    cos_sza = (np.sin(lat) * np.sin(decl) +
               np.cos(lat) * np.cos(decl) * np.cos(ha))
    return np.rad2deg(np.arccos(np.clip(cos_sza, -1, 1)))


# ---------------------------------------------------------------------------
# Pressure to altitude
# ---------------------------------------------------------------------------

def pressure_to_alt_km(p_hpa):
    """z = -7 * ln(p/1013) km (standard atmosphere approximation)."""
    return -7.0 * np.log(np.asarray(p_hpa, dtype=float) / 1013.0)


# ---------------------------------------------------------------------------
# Process one file
# ---------------------------------------------------------------------------

def process_file(fpath, wavelengths_nm, strat_alt_min_km,
                 strat_alt_max_km, spatial_stride):
    """
    For one h2 file, compute extinction profiles at a spatial subsample
    of grid points, then compute full vs daylit zonal mean per latitude.

    Returns:
        results : dict {wl: {full_sum, daylit_sum, n_full, n_daylit}} (lat, alt)
        lats    : array of latitude values at stride
        alt_km  : altitude grid (km) restricted to stratosphere
        doy     : day of year
    """
    ds  = xr.open_dataset(fpath, engine="netcdf4")
    all_lats = ds["lat"].values
    all_lons = ds["lon"].values

    # extract day of year from cftime
    t0  = ds["time"].values[0]
    doy = t0.timetuple().tm_yday if hasattr(t0, "timetuple") else 1
    ds.close()

    # stratospheric altitude mask
    strat_mask = (ALT_GRID_KM >= strat_alt_min_km) & \
                 (ALT_GRID_KM <= strat_alt_max_km)
    alt_km = ALT_GRID_KM[strat_mask]
    n_alt  = strat_mask.sum()

    # subsampled lat/lon indices
    lat_idx = np.arange(0, len(all_lats), spatial_stride)
    lon_idx = np.arange(0, len(all_lons), spatial_stride)
    lats = all_lats[lat_idx]
    lons = all_lons[lon_idx]

    # SZA at subsampled grid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    sza = solar_zenith_angle(lat_grid, lon_grid, doy, utc_hour=0.0)
    daylit = sza < 90.0   # (n_lat, n_lon)

    # accumulate per-latitude zonal sums
    results = {
        wl: {
            "full_sum":   np.zeros((len(lats), n_alt)),
            "daylit_sum": np.zeros((len(lats), n_alt)),
            "n_full":     np.zeros(len(lats)),
            "n_daylit":   np.zeros(len(lats)),
        }
        for wl in wavelengths_nm
    }

    waccm = WACCMAtmosphere(fpath, alt_grid_km=ALT_GRID_KM)

    for i_lat, lat in enumerate(lats):
        for i_lon, lon in enumerate(lons):
            try:
                profiles = waccm.get_column_profiles(lat, lon, time_index=0)
            except Exception:
                continue

            for wl in wavelengths_nm:
                ext_full = extinction_profile(profiles, wl)[strat_mask]
                results[wl]["full_sum"][i_lat]   += ext_full
                results[wl]["n_full"][i_lat]     += 1
                if daylit[i_lat, i_lon]:
                    results[wl]["daylit_sum"][i_lat] += ext_full
                    results[wl]["n_daylit"][i_lat]   += 1

    # compute means
    for wl in wavelengths_nm:
        nf = results[wl]["n_full"][:, np.newaxis]
        nd = results[wl]["n_daylit"][:, np.newaxis]
        results[wl]["full_mean"]   = np.where(nf > 0,
                                              results[wl]["full_sum"] / np.maximum(nf, 1),
                                              np.nan)
        results[wl]["daylit_mean"] = np.where(nd > 0,
                                              results[wl]["daylit_sum"] / np.maximum(nd, 1),
                                              np.nan)

    return results, lats, alt_km, doy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bias_profiles(all_bias, alt_km, lats, wavelengths_nm, out_dir):
    lat_bands = [
        (-90, -60, "SH high"),
        (-30,   0, "SH subtropics"),
        (  0,  30, "NH tropics"),
        ( 25,  35, "Injection band"),
        ( 30,  60, "NH midlat"),
        ( 60,  90, "NH high"),
    ]

    for wl in wavelengths_nm:
        fig, axes = plt.subplots(1, len(lat_bands), figsize=(16, 7), sharey=True)
        for ax, (lat_min, lat_max, label) in zip(axes, lat_bands):
            lat_mask = (lats >= lat_min) & (lats < lat_max)
            if not np.any(lat_mask):
                ax.set_visible(False)
                continue

            bias = all_bias[wl][lat_mask, :]     # (n_lat, n_alt)
            mean_bias = np.nanmean(bias, axis=0)
            std_bias  = np.nanstd(bias, axis=0)

            ax.fill_betweenx(alt_km,
                             (mean_bias - std_bias) * 100,
                             (mean_bias + std_bias) * 100,
                             alpha=0.3, color="steelblue")
            ax.plot(mean_bias * 100, alt_km, color="steelblue", linewidth=1.8)
            ax.axvline(0,    color="black",  linewidth=0.8, linestyle="--")
            ax.axvline( 10,  color="gray",   linewidth=0.6, linestyle=":")
            ax.axvline(-10,  color="gray",   linewidth=0.6, linestyle=":")
            ax.axhline(22.0, color="orange", linewidth=0.8, linestyle="--",
                       alpha=0.7, label="Injection alt")

            ax.set_xlabel("Bias (%)", fontsize=9)
            ax.set_title(f"{label}\n({lat_min} to {lat_max} N)", fontsize=9)
            ax.set_ylim(alt_km.min(), alt_km.max())
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-50, 50)

        axes[0].set_ylabel("Altitude (km)", fontsize=10)
        fig.suptitle(
            f"Zonal sampling bias: (daylit - full) / full\n"
            f"Wavelength {wl:.0f} nm — mean +/- std across files\n"
            f"Orange = injection altitude (22 km), dotted = 10% threshold",
            fontsize=11
        )
        fig.tight_layout()
        fname = os.path.join(out_dir, f"fig_zonal_bias_{wl:.0f}nm.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {fname}", flush=True)


def plot_bias_map(all_bias, alt_km, lats, wavelength_nm, out_dir):
    bias_2d = all_bias[wavelength_nm] * 100   # (lat, alt) percent

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.contourf(lats, alt_km, bias_2d.T,
                     levels=np.linspace(-30, 30, 25),
                     cmap="RdBu_r", extend="both")
    fig.colorbar(im, ax=ax, label="Fractional bias (%)")
    ax.axvline(30.6, color="black",  linewidth=0.8, linestyle="--",
               label="Injection lat (30.6 N)")
    ax.axhline(22.0, color="orange", linewidth=0.8, linestyle="--",
               label="Injection alt (22 km)")
    ax.set_xlabel("Latitude (deg N)", fontsize=11)
    ax.set_ylabel("Altitude (km)", fontsize=11)
    ax.set_ylim(alt_km.min(), alt_km.max())
    ax.set_title(
        f"Zonal sampling bias: (daylit - full) / full\n"
        f"Wavelength {wavelength_nm:.0f} nm",
        fontsize=11
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = os.path.join(out_dir, f"fig_zonal_bias_map_{wavelength_nm:.0f}nm.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}", flush=True)


def write_summary(all_bias, alt_km, lats, wavelengths_nm, inj_lat, out_dir):
    inj_lat_mask = (lats >= inj_lat - 5) & (lats <= inj_lat + 5)
    lines = [
        "Zonal sampling bias summary",
        "===========================",
        "(daylit hemisphere at 00:00 UTC vs full zonal mean)",
        f"Injection latitude band: {inj_lat-5:.1f} to {inj_lat+5:.1f} N",
        "",
    ]

    alt_ranges = [
        (16, 20, "Lower stratosphere  (16-20 km)"),
        (20, 26, "Middle stratosphere (20-26 km) — main aerosol layer"),
        (26, 32, "Upper stratosphere  (26-32 km)"),
        (32, 36, "High stratosphere   (32-36 km)"),
    ]

    for wl in wavelengths_nm:
        lines.append(f"Wavelength: {wl:.0f} nm")
        lines.append("-" * 40)
        bias = all_bias[wl]   # (lat, alt)

        for alt_lo, alt_hi, label in alt_ranges:
            alt_mask = (alt_km >= alt_lo) & (alt_km < alt_hi)
            if not np.any(alt_mask):
                continue
            b_global = np.nanmean(np.abs(bias[:, alt_mask])) * 100
            b_inj    = np.nanmean(np.abs(bias[inj_lat_mask, :][:, alt_mask])) * 100
            lines.append(f"  {label}")
            lines.append(f"    Global mean |bias|:        {b_global:.1f}%")
            lines.append(f"    Injection lat |bias|:      {b_inj:.1f}%")
        lines.append("")

    text = "\n".join(lines)
    print(text, flush=True)
    fname = os.path.join(out_dir, "zonal_bias_summary.txt")
    with open(fname, "w") as f:
        f.write(text + "\n")
    print(f"  saved: {fname}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="+", required=True,
                   help="h2 NetCDF files (glob patterns ok)")
    p.add_argument("--out_dir", default="./figures/zonal_bias")
    p.add_argument("--lat", type=float, default=30.6,
                   help="Injection latitude for annotation (default 30.6)")
    p.add_argument("--every_nth_file", type=int, default=10,
                   help="Process every Nth file (default 10)")
    p.add_argument("--spatial_stride", type=int, default=4,
                   help="Process every Nth lat/lon grid point (default 4 "
                        "= every 2.5 deg for 192x288 grid)")
    p.add_argument("--wavelengths_nm", nargs="+", type=float,
                   default=[470.0, 745.0, 1020.0],
                   help="ALI wavelengths nm (default 470 745 1020)")
    p.add_argument("--strat_alt_min_km", type=float, default=15.0)
    p.add_argument("--strat_alt_max_km", type=float, default=40.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # expand globs
    files = []
    for pat in args.files:
        files.extend(sorted(glob.glob(pat)))
    files = sorted(set(files))
    if not files:
        sys.exit("ERROR: no files found")

    files_to_process = files[::args.every_nth_file]
    print(f"Processing {len(files_to_process)} of {len(files)} files "
          f"(every {args.every_nth_file}th, spatial stride {args.spatial_stride})",
          flush=True)

    # accumulate bias across files
    all_bias_accum = {wl: [] for wl in args.wavelengths_nm}
    alt_km = None
    lats   = None

    for i, fpath in enumerate(files_to_process):
        print(f"\n[{i+1}/{len(files_to_process)}] {os.path.basename(fpath)}",
              flush=True)
        try:
            results, lats, alt_km, doy = process_file(
                fpath, args.wavelengths_nm,
                args.strat_alt_min_km, args.strat_alt_max_km,
                args.spatial_stride
            )
            for wl in args.wavelengths_nm:
                full   = results[wl]["full_mean"]    # (lat, alt)
                daylit = results[wl]["daylit_mean"]
                bias   = np.where(
                    (full > 0) & ~np.isnan(full) & ~np.isnan(daylit),
                    (daylit - full) / full,
                    np.nan
                )
                all_bias_accum[wl].append(bias)
            print(f"  lats sampled: {len(lats)}, alt levels: {len(alt_km)}",
                  flush=True)
        except Exception as e:
            print(f"  WARNING: failed — {e}", flush=True)
            continue

    if alt_km is None:
        sys.exit("ERROR: no files processed successfully")

    # average bias across files
    all_bias = {}
    for wl in args.wavelengths_nm:
        if all_bias_accum[wl]:
            stack = np.stack(all_bias_accum[wl], axis=0)   # (n_files, lat, alt)
            all_bias[wl] = np.nanmean(stack, axis=0)         # (lat, alt)
        else:
            all_bias[wl] = np.full((len(lats), len(alt_km)), np.nan)

    print("\nPlotting bias profiles...", flush=True)
    plot_bias_profiles(all_bias, alt_km, lats, args.wavelengths_nm, args.out_dir)

    print("Plotting bias maps...", flush=True)
    for wl in args.wavelengths_nm:
        plot_bias_map(all_bias, alt_km, lats, wl, args.out_dir)

    print("Writing summary...", flush=True)
    write_summary(all_bias, alt_km, lats, args.wavelengths_nm,
                  args.lat, args.out_dir)

    print(f"\nDone. Figures saved to {args.out_dir}", flush=True)