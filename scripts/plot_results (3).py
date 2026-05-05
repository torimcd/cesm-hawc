#!/usr/bin/env python
"""
plot_results.py
===============
Generate all figures from saved ALI simulation output and CESM hourly files.

Run after run_simulation.py has completed:

    python scripts/plot_results.py

Or point at a specific results directory:

    python scripts/plot_results.py ~/results/hawc_ali/

Figures produced
----------------
fig1_extinction_profiles.png   — ALI retrieved extinction + CESM EXTINCTdn
fig2_anomaly.png               — Injection anomaly in extinction and radius
fig3_retrieval_diagnostics.png — Prior vs retrieved, 1σ uncertainty
figA_profile_timeseries.png    — Analog to Sellitto et al. Fig. 7:
                                 CESM PR vs ALI PO at selected days
figB_aod_timeseries.png        — Analog to Sellitto et al. Fig. 1d:
                                 Stratospheric AOD time series
figC_hovmoller.png             — Analog to Sellitto et al. Fig. 6:
                                 Time–altitude Hovmöller of extinction anomaly
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "font.family":     "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "legend.frameon":  False,
})

# ── CONFIGURATION ──────────────────────────────────────────────────────────

# L2 results from run_simulation.py
OUT_DIR = os.path.expanduser("~/projects/rrg-czg/vmcd/hawc/results/")

# Hourly CESM files for Sellitto-analog figures
BG_DIR       = "/home/vmcd/scratch/cesm/output/archive/ssp45_2035_001/atm/hist/"
INJ_DIR      = "/home/vmcd/scratch/cesm/output/archive/sai_2035_001/atm/hist/"
FILE_PATTERN = "*.cam.h0.2035-01-*.nc"

# Observation column
TANGENT_LAT = 30.6
TANGENT_LON = 180.0

# Altitude range for all plots [km]
ALT_MIN_KM = 5.0
ALT_MAX_KM = 40.0
REF_ALT_KM = 20.0   # reference line (injection altitude)

# Days within the month to show in Figure A (0-indexed)
PROFILE_DAYS = [0, 7, 14, 21, 30]   # Jan 1, 8, 15, 22, Feb 1

# ── END CONFIGURATION ──────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_l2_results(out_dir: str) -> tuple:
    """Load L2 and CESM extinction NetCDF files saved by run_simulation.py."""
    bg_path       = os.path.join(out_dir, "l2_background.nc")
    inj_path      = os.path.join(out_dir, "l2_injection.nc")
    cesm_bg_path  = os.path.join(out_dir, "cesm_extinction_background.nc")
    cesm_inj_path = os.path.join(out_dir, "cesm_extinction_injection.nc")

    if not os.path.exists(bg_path):
        raise FileNotFoundError(
            f"Background L2 file not found: {bg_path}\n"
            "Run run_simulation.py first."
        )

    l2_bg    = xr.open_dataset(bg_path)
    l2_inj   = xr.open_dataset(inj_path)   if os.path.exists(inj_path)      else None
    cesm_bg  = xr.open_dataset(cesm_bg_path)  if os.path.exists(cesm_bg_path)  else None
    cesm_inj = xr.open_dataset(cesm_inj_path) if os.path.exists(cesm_inj_path) else None

    if l2_inj  is None: print("No injection L2 file found — plotting background only.")
    if cesm_bg is None: print("No CESM extinction file — rerun run_simulation.py.")

    return l2_bg, l2_inj, cesm_bg, cesm_inj


def load_cesm_hourly(directory: str, pattern: str) -> xr.Dataset:
    """Load all daily hourly files for a scenario via dask."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    print(f"  {len(files)} files from {os.path.basename(directory.rstrip('/'))}")
    return xr.open_mfdataset(files, engine="netcdf4",
                              combine="by_coords", chunks={"time": 24})


def extract_column_profile(ds: xr.Dataset, lat: float, lon: float,
                            time_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (altitude_km, EXTINCTdn) for one column and timestep, bottom→top."""
    col = ds.isel(time=time_idx).sel(lat=lat, lon=lon, method="nearest")
    ext = col["EXTINCTdn"].values
    z   = col["Z3"].values / 1e3
    idx = np.argsort(z)
    return z[idx], ext[idx]


def strat_aod(ds: xr.Dataset, lat: float, lon: float) -> xr.DataArray:
    """
    Stratospheric AOD (15–35 km) at every timestep via extinction integral.
    Returns a DataArray with dimension 'time'.
    """
    col  = ds.sel(lat=lat, lon=lon, method="nearest")
    z    = col["Z3"]
    ext  = col["EXTINCTdn"]
    dz   = np.abs(z.differentiate("lev"))
    mask = (z >= 15e3) & (z <= 35e3)
    return (ext * dz * mask).sum("lev")


def _ref_hline(ax, label=True):
    ax.axhline(REF_ALT_KM, color="grey", lw=0.6, ls=":",
               label=f"{REF_ALT_KM} km" if label else None)


# ─────────────────────────────────────────────────────────────────────────────
# Figures 1–3: ALI L2 results
# ─────────────────────────────────────────────────────────────────────────────

def plot_extinction_profiles(l2_bg, l2_inj, cesm_bg, cesm_inj, out_dir):
    """
    Figure 1: ALI retrieved extinction and median radius profiles.

    Left panel shows three lines per scenario:
      solid  — ALI L2 retrieved (745 nm)
      dotted — CESM EXTINCTdn   (550 nm)
    Right panel shows retrieved median radius.
    """
    ext_bg  = l2_bg["stratospheric_aerosol_extinction_per_m"]
    r_bg    = l2_bg["stratospheric_aerosol_median_radius"]
    alts_km = ext_bg.altitude.values / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    axes[0].plot(ext_bg.values, alts_km,
                 lw=2, color="steelblue", label="ALI retrieved 745 nm (bg)")
    axes[1].plot(r_bg.values,  alts_km,
                 lw=2, color="steelblue", label="Retrieved (bg)")

    if cesm_bg is not None:
        z_cesm = cesm_bg["altitude_m"].values / 1e3
        axes[0].plot(cesm_bg["ext_550nm"].values, z_cesm,
                     lw=1.5, ls=":", color="steelblue",
                     label="CESM EXTINCTdn 550 nm (bg)")

    if l2_inj is not None:
        ext_inj = l2_inj["stratospheric_aerosol_extinction_per_m"]
        r_inj   = l2_inj["stratospheric_aerosol_median_radius"]
        alts_inj_km = ext_inj.altitude.values / 1e3

        axes[0].plot(ext_inj.values, alts_inj_km,
                     lw=2, ls="--", color="firebrick",
                     label="ALI retrieved 745 nm (inj)")
        axes[1].plot(r_inj.values,   alts_inj_km,
                     lw=2, ls="--", color="firebrick", label="Retrieved (inj)")

        if cesm_inj is not None:
            z_cesm = cesm_inj["altitude_m"].values / 1e3
            axes[0].plot(cesm_inj["ext_550nm"].values, z_cesm,
                         lw=1.5, ls=":", color="firebrick",
                         label="CESM EXTINCTdn 550 nm (inj)")

    axes[0].set_xlabel("Extinction @ 745 nm [m⁻¹]")
    axes[0].set_xscale("log")
    axes[1].set_xlabel("Retrieved median radius [nm]")

    for ax in axes:
        ax.set_ylabel("Altitude [km]")
        ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
        _ref_hline(ax)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("ALI retrieved aerosol profiles")
    plt.tight_layout()
    _save(fig, out_dir, "fig1_extinction_profiles.png")


def plot_anomaly(l2_bg, l2_inj, out_dir):
    """
    Figure 2: Injection anomaly in retrieved extinction and median radius.
    """
    ext_bg  = l2_bg["stratospheric_aerosol_extinction_per_m"]
    ext_inj = l2_inj["stratospheric_aerosol_extinction_per_m"]
    r_bg    = l2_bg["stratospheric_aerosol_median_radius"]
    r_inj   = l2_inj["stratospheric_aerosol_median_radius"]

    anom_ext = (ext_inj.values - ext_bg.values) * 1e5
    anom_r   = r_inj.values - r_bg.values
    alts     = ext_bg.altitude.values / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    for ax, anom, xlabel in [
        (axes[0], anom_ext, "Δ extinction [×10⁻⁵ m⁻¹]"),
        (axes[1], anom_r,   "Δ median radius [nm]"),
    ]:
        ax.barh(alts, np.where(anom >= 0, anom, 0),
                height=0.4, color="firebrick", label="+anomaly")
        ax.barh(alts, np.where(anom <  0, anom, 0),
                height=0.4, color="steelblue", label="−anomaly")
        ax.axvline(0, color="k", lw=0.8)
        _ref_hline(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Altitude [km]")
        ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
        ax.legend(fontsize=9)

    fig.suptitle("Injection anomaly (injection − background)")
    plt.tight_layout()
    _save(fig, out_dir, "fig2_anomaly.png")


def plot_retrieval_diagnostics(l2_bg, l2_inj, out_dir):
    """
    Figure 3: Prior vs retrieved extinction and 1σ retrieval uncertainty.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    for ds, label, color in [
        (l2_bg,  "Background", "steelblue"),
        (l2_inj, "Injection",  "firebrick"),
    ]:
        if ds is None:
            continue
        ext       = ds["stratospheric_aerosol_extinction_per_m"]
        ext_prior = ds["stratospheric_aerosol_extinction_per_m_prior"]
        ext_sigma = ds["stratospheric_aerosol_extinction_per_m_1sigma_error"]
        alts_km   = ext.altitude.values / 1e3

        axes[0].plot(ext.values,       alts_km, lw=2,   color=color,
                     label=f"{label} retrieved")
        axes[0].plot(ext_prior.values, alts_km, lw=1.5, color=color,
                     ls=":", label=f"{label} prior")
        axes[1].plot(ext_sigma.values / np.maximum(ext.values, 1e-12),
                     alts_km, lw=2, color=color, label=label)

    axes[0].set_xlabel("Extinction @ 745 nm [m⁻¹]")
    axes[0].set_xscale("log")
    axes[0].set_title("Retrieved vs prior")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Relative 1σ uncertainty")
    axes[1].set_title("Retrieval uncertainty")
    axes[1].axvline(1.0, color="grey", lw=0.7, ls=":")
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.set_ylabel("Altitude [km]")
        ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
        _ref_hline(ax)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Retrieval diagnostics")
    plt.tight_layout()
    _save(fig, out_dir, "fig3_retrieval_diagnostics.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figures A–C: Sellitto et al. analogs using hourly CESM output
# ─────────────────────────────────────────────────────────────────────────────

def plot_figA_profile_timeseries(ds_bg, ds_inj, l2_bg, l2_inj, out_dir):
    """
    Figure A — analog to Sellitto et al. Fig. 7.

    Rows: background | injection | anomaly (inj − bg)
    Columns: selected days through January 2035.
    Solid lines: CESM EXTINCTdn (pseudo-reality, PR).
    Dashed line: ALI L2 retrieved (pseudo-observation, PO) — shown on day 1
                 only since the L2 is from a single-timestep simulation.
    """
    n_per_day = 24
    time_idxs = [d * n_per_day for d in PROFILE_DAYS
                 if d * n_per_day < ds_inj.sizes["time"]]
    day_labels = [str(ds_inj.time.values[i])[:10] for i in time_idxs]
    n_cols = len(time_idxs)

    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 10),
                              sharey=True, sharex=False)

    # ALI L2 profiles for PO overlay on first column
    ext_l2_bg  = l2_bg["stratospheric_aerosol_extinction_per_m"].values
    ext_l2_inj = l2_inj["stratospheric_aerosol_extinction_per_m"].values
    alt_l2_km  = l2_bg["stratospheric_aerosol_extinction_per_m"].altitude.values / 1e3

    z_grid = np.linspace(ALT_MIN_KM, ALT_MAX_KM, 200)

    for ci, (t_idx, label) in enumerate(zip(time_idxs, day_labels)):
        ax_bg, ax_inj, ax_diff = axes[0, ci], axes[1, ci], axes[2, ci]

        z_bg,  ext_bg_c  = extract_column_profile(ds_bg,  TANGENT_LAT, TANGENT_LON, t_idx)
        z_inj, ext_inj_c = extract_column_profile(ds_inj, TANGENT_LAT, TANGENT_LON, t_idx)

        m_bg  = (z_bg  >= ALT_MIN_KM) & (z_bg  <= ALT_MAX_KM)
        m_inj = (z_inj >= ALT_MIN_KM) & (z_inj <= ALT_MAX_KM)

        # Background
        ax_bg.plot(ext_bg_c[m_bg], z_bg[m_bg],
                   color="steelblue", lw=2, label="CESM PR")
        if ci == 0:
            ax_bg.plot(ext_l2_bg, alt_l2_km,
                       color="steelblue", lw=1.5, ls="--", label="ALI PO")
        ax_bg.set_xscale("log")
        ax_bg.set_title(label, fontsize=9)

        # Injection
        ax_inj.plot(ext_inj_c[m_inj], z_inj[m_inj],
                    color="firebrick", lw=2, label="CESM PR")
        if ci == 0:
            ax_inj.plot(ext_l2_inj, alt_l2_km,
                        color="firebrick", lw=1.5, ls="--", label="ALI PO")
        ax_inj.set_xscale("log")

        # Anomaly
        ext_bg_i  = np.interp(z_grid, z_bg[m_bg],   ext_bg_c[m_bg],   left=0, right=0)
        ext_inj_i = np.interp(z_grid, z_inj[m_inj], ext_inj_c[m_inj], left=0, right=0)
        diff = (ext_inj_i - ext_bg_i) * 1e5
        ax_diff.fill_betweenx(z_grid, 0, diff, where=diff >= 0,
                               color="firebrick", alpha=0.6)
        ax_diff.fill_betweenx(z_grid, 0, diff, where=diff <  0,
                               color="steelblue", alpha=0.6)
        ax_diff.axvline(0, color="k", lw=0.8)
        ax_diff.set_xlabel("Δ ext\n[×10⁻⁵ m⁻¹]", fontsize=8)

        for ax in [ax_bg, ax_inj, ax_diff]:
            ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
            _ref_hline(ax, label=False)
            if ci == 0:
                ax.set_ylabel("Altitude [km]")
            if ci == 0 or ax is ax_diff:
                ax.legend(fontsize=8)

    for ax, row_label in zip(axes[:, 0],
                              ["Background", "Injection", "Anomaly (inj−bg)"]):
        ax.annotate(row_label, xy=(-0.38, 0.5), xycoords="axes fraction",
                    rotation=90, va="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Figure A — CESM pseudo-reality vs ALI pseudo-observation\n"
        f"Extinction profiles at {TANGENT_LAT}°N, {TANGENT_LON}°E  "
        "(solid: CESM EXTINCTdn 550 nm | dashed: ALI retrieved 745 nm)",
        y=1.01, fontsize=11
    )
    plt.tight_layout()
    _save(fig, out_dir, "figA_profile_timeseries.png")


def plot_figB_aod_timeseries(ds_bg, ds_inj, out_dir):
    """
    Figure B — analog to Sellitto et al. Fig. 1d.
    Stratospheric AOD time series over January 2035.
    """
    print("  Computing AOD time series (may take a moment)...")
    aod_bg  = strat_aod(ds_bg,  TANGENT_LAT, TANGENT_LON).compute()
    aod_inj = strat_aod(ds_inj, TANGENT_LAT, TANGENT_LON).compute()

    times = ds_inj.time.values
    days  = (times - times[0]).astype("timedelta64[h]").astype(float) / 24.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(days, aod_bg.values,  color="steelblue", lw=1.5,
                 label="Background (SSP2-4.5)")
    axes[0].plot(days, aod_inj.values, color="firebrick", lw=1.5,
                 label="SAI injection")
    axes[0].fill_between(days, aod_bg.values, aod_inj.values,
                          where=aod_inj.values >= aod_bg.values,
                          color="firebrick", alpha=0.15,
                          label="Injection excess")
    axes[0].set_ylabel("Stratospheric AOD (15–35 km, 550 nm)")
    axes[0].legend()
    axes[0].set_title(
        f"Figure B — Stratospheric AOD time series\n"
        f"{TANGENT_LAT}°N, {TANGENT_LON}°E  |  January 2035"
    )

    anom = aod_inj.values - aod_bg.values
    axes[1].fill_between(days, 0, anom, where=anom >= 0,
                          color="firebrick", alpha=0.7, label="+anomaly")
    axes[1].fill_between(days, 0, anom, where=anom <  0,
                          color="steelblue", alpha=0.7, label="−anomaly")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("ΔAOD")
    axes[1].set_xlabel("Day of January 2035")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, out_dir, "figB_aod_timeseries.png")


def plot_figC_hovmoller(ds_bg, ds_inj, out_dir):
    """
    Figure C — analog to Sellitto et al. Fig. 6.
    Time–altitude Hovmöller of extinction anomaly at the injection column.
    Their figure uses longitude on x-axis; ours uses time since we have
    a single column but hourly resolution — more temporally detailed than
    their monthly-mean approach.
    """
    print("  Building time-altitude arrays (may take a moment)...")

    col_bg  = ds_bg.sel( lat=TANGENT_LAT, lon=TANGENT_LON, method="nearest")
    col_inj = ds_inj.sel(lat=TANGENT_LAT, lon=TANGENT_LON, method="nearest")

    ext_bg_t  = col_bg["EXTINCTdn"].compute().values
    ext_inj_t = col_inj["EXTINCTdn"].compute().values
    z_bg_t    = col_bg["Z3"].compute().values  / 1e3
    z_inj_t   = col_inj["Z3"].compute().values / 1e3

    z_grid = np.arange(ALT_MIN_KM, ALT_MAX_KM + 0.5, 0.5)
    n_t    = ext_bg_t.shape[0]
    ext_bg_i  = np.zeros((n_t, len(z_grid)))
    ext_inj_i = np.zeros((n_t, len(z_grid)))

    for i in range(n_t):
        idx_bg  = np.argsort(z_bg_t[i])
        idx_inj = np.argsort(z_inj_t[i])
        ext_bg_i[i]  = np.interp(z_grid, z_bg_t[i,  idx_bg],
                                  ext_bg_t[i,  idx_bg],  left=0, right=0)
        ext_inj_i[i] = np.interp(z_grid, z_inj_t[i, idx_inj],
                                  ext_inj_t[i, idx_inj], left=0, right=0)

    anom_t = (ext_inj_i - ext_bg_i) * 1e5

    times = ds_inj.time.values
    days  = (times - times[0]).astype("timedelta64[h]").astype(float) / 24.0

    vmax = max(np.nanpercentile(np.abs(anom_t), 98), 0.1)

    fig, ax = plt.subplots(figsize=(12, 6))
    cf = ax.contourf(days, z_grid, anom_t.T,
                     levels=np.linspace(-vmax, vmax, 21),
                     cmap="RdBu_r", extend="both")
    ax.contour(days, z_grid, anom_t.T, levels=[0],
               colors="k", linewidths=0.8)

    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("Δ extinction [×10⁻⁵ m⁻¹]  (injection − background)")

    ax.set_xlabel("Day of January 2035")
    ax.set_ylabel("Altitude [km]")
    ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
    _ref_hline(ax, label=True)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Figure C — Time–altitude Hovmöller of extinction anomaly\n"
        f"{TANGENT_LAT}°N, {TANGENT_LON}°E  |  CESM EXTINCTdn 550 nm  |  January 2035"
    )

    plt.tight_layout()
    _save(fig, out_dir, "figC_hovmoller.png")


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: str, filename: str) -> None:
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"Results directory: {out_dir}")

    # ── Figures 1–3: ALI L2 results ─────────────────────────────────────
    print("\n── Figures 1–3: ALI L2 results ──")
    l2_bg, l2_inj, cesm_bg, cesm_inj = load_l2_results(out_dir)

    plot_extinction_profiles(l2_bg, l2_inj, cesm_bg, cesm_inj, out_dir)
    plot_retrieval_diagnostics(l2_bg, l2_inj, out_dir)
    if l2_inj is not None:
        plot_anomaly(l2_bg, l2_inj, out_dir)

    # ── Figures A–C: Sellitto analog (requires hourly CESM files) ────────
    print("\n── Figures A–C: Sellitto et al. analogs ──")
    try:
        print("Loading hourly CESM files...")
        ds_bg_h  = load_cesm_hourly(BG_DIR,  FILE_PATTERN)
        ds_inj_h = load_cesm_hourly(INJ_DIR, FILE_PATTERN)

        # Align to the shorter injection time axis
        t_inj   = ds_inj_h.time.values
        ds_bg_h = ds_bg_h.sel(time=t_inj, method="nearest")
        print(f"  Aligned: {len(t_inj)} timesteps  "
              f"{str(t_inj[0])[:10]} → {str(t_inj[-1])[:10]}")

        if l2_inj is not None:
            plot_figA_profile_timeseries(ds_bg_h, ds_inj_h,
                                         l2_bg, l2_inj, out_dir)
        plot_figB_aod_timeseries(ds_bg_h, ds_inj_h, out_dir)
        plot_figC_hovmoller(ds_bg_h, ds_inj_h, out_dir)

    except FileNotFoundError as e:
        print(f"  Skipping Figures A–C: {e}")
        print("  Set BG_DIR and INJ_DIR in the CONFIGURATION section.")

    print("\nDone.")


if __name__ == "__main__":
    main()
