#!/usr/bin/env python
"""
plot_results.py
===============
Generate all figures from saved ALI simulation output and CESM monthly files,
across ALL months produced by run_simulation_all.py.

Output layout expected:
    OUT_DIR/
      background/YYYY-MM/l2_background.nc
      background/YYYY-MM/cesm_extinction_background.nc
      injection/YYYY-MM/l2_injection.nc          (optional)
      injection/YYYY-MM/cesm_extinction_injection.nc

Figures produced
----------------
fig1_extinction_profiles.png   — Multi-panel: ALI retrieved extinction + CESM
                                 EXTINCTdn, one subplot per month
fig2_anomaly.png               — Multi-panel: injection − background anomaly
                                 in extinction and radius, one subplot per month
fig3_retrieval_diagnostics.png — Multi-panel: prior vs retrieved, 1σ uncertainty
figA_profile_timeseries.png    — Analog to Sellitto et al. Fig. 7:
                                 CESM PR vs ALI PO at selected months
figB_aod_timeseries.png        — Analog to Sellitto et al. Fig. 1d:
                                 Monthly stratospheric AOD time series
figC_hovmoller.png             — Analog to Sellitto et al. Fig. 6:
                                 Month–altitude Hovmöller of extinction anomaly
ts_summary.png                 — Stratospheric burden, peak radius, peak anomaly

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py ~/results/hawc_ali/
    python scripts/plot_results.py ~/results/hawc_ali/ --months 2035-01 2035-06
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         10,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    10,
    "legend.fontsize":   8,
    "legend.frameon":    False,
})

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

OUT_DIR = os.path.expanduser("~/projects/rrg-czg/vmcd/hawc/results/")

# Root archive directory containing <casename>/atm/hist/*.cam.h0.*.nc
# Used only for Figures A–C (Sellitto analogs).
CESM_ARCHIVE_DIR = os.path.expanduser("~/scratch/cesm/output/archive/")
BG_CASENAME      = "sai_background_2035_001"
INJ_CASENAME     = "sai_1.0Tg_2035_001"

# Observation column
TANGENT_LAT = 30.6
TANGENT_LON = 180.0

# Altitude range for all plots [km]
ALT_MIN_KM   = 5.0
ALT_MAX_KM   = 40.0
REF_ALT_KM   = 20.0    # reference line (injection altitude)
STRAT_MIN_KM = 15.0    # lower bound for stratospheric integrals

# ── END CONFIGURATION ──────────────────────────────────────────────────────────


# ── data discovery ─────────────────────────────────────────────────────────────

def discover_months(out_dir: str) -> list[str]:
    """Return sorted YYYY-MM list for all months with a background L2 file."""
    bg_root = Path(out_dir) / "background"
    if not bg_root.exists():
        raise FileNotFoundError(
            f"No 'background' subdirectory in {out_dir}.\n"
            "Run run_simulation_all.py first."
        )
    months = sorted(
        d.name for d in bg_root.iterdir()
        if d.is_dir() and (d / "l2_background.nc").exists()
    )
    if not months:
        raise FileNotFoundError(f"No l2_background.nc files under {bg_root}")
    return months


def load_month(out_dir: str, date: str) -> tuple[
    xr.Dataset,
    Optional[xr.Dataset],
    Optional[xr.Dataset],
    Optional[xr.Dataset],
]:
    """Load (l2_bg, l2_inj, cesm_bg, cesm_inj) for one month; inj/cesm may be None."""
    root = Path(out_dir)

    def _open(p):
        return xr.open_dataset(p) if p.exists() else None

    l2_bg    = xr.open_dataset(root / "background" / date / "l2_background.nc")
    l2_inj   = _open(root / "injection"  / date / "l2_injection.nc")
    cesm_bg  = _open(root / "background" / date / "cesm_extinction_background.nc")
    cesm_inj = _open(root / "injection"  / date / "cesm_extinction_injection.nc")
    return l2_bg, l2_inj, cesm_bg, cesm_inj


def load_cesm_h0(archive_dir: str, casename: str) -> xr.Dataset:
    """Load all monthly h0 files for a case."""
    import glob
    pattern = os.path.join(archive_dir, casename, "atm", "hist",
                           "*.cam.h0.*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No h0 files found at: {pattern}")
    print(f"  {len(files)} h0 file(s) for {casename}")
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    return xr.open_mfdataset(
        files, combine="by_coords", decode_times=time_coder,
        data_vars="minimal", coords="minimal", compat="override",
    )


# ── layout helpers ─────────────────────────────────────────────────────────────

def _panel_grid(n: int) -> tuple[int, int]:
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _make_grid(n: int, figsize_per=(4.5, 6),
               sharey=True) -> tuple[plt.Figure, np.ndarray]:
    nrows, ncols = _panel_grid(n)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        sharey=sharey, constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat


def _ref_hline(ax, label=True):
    ax.axhline(REF_ALT_KM, color="grey", lw=0.6, ls=":",
               label=f"{REF_ALT_KM} km" if label else None)


def _save(fig, out_dir: str, filename: str) -> None:
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _strat_burden(ext: np.ndarray, alts_km: np.ndarray) -> float:
    mask = alts_km >= STRAT_MIN_KM
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(ext[mask], alts_km[mask]))


# ── Figures 1–3: ALI L2 results (multi-panel, one subplot per month) ──────────

def fig1_extinction_profiles(months: list[str], out_dir: str) -> None:
    """
    Multi-panel figure: ALI retrieved extinction & median radius profiles.
    Each month gets two sub-columns (extinction | radius); shared y-axis.
    CESM direct extinction overlaid as dotted line where available.
    """
    n = len(months)
    nrows, ncols = _panel_grid(n)
    fig, all_axes = plt.subplots(
        nrows, ncols * 2,
        figsize=(3.8 * ncols * 2, 6 * nrows),
        sharey=True, constrained_layout=True,
    )
    all_axes_flat = np.array(all_axes).flatten()
    for i in range(n * 2, len(all_axes_flat)):
        all_axes_flat[i].set_visible(False)

    for idx, date in enumerate(months):
        row = idx // ncols
        col = idx %  ncols
        ax_ext = all_axes[row, col * 2]
        ax_r   = all_axes[row, col * 2 + 1]

        try:
            l2_bg, l2_inj, cesm_bg, cesm_inj = load_month(out_dir, date)
        except Exception as e:
            ax_ext.set_title(f"{date}\n(error)", fontsize=8, color="red")
            continue

        ext_bg  = l2_bg["stratospheric_aerosol_extinction_per_m"]
        r_bg    = l2_bg["stratospheric_aerosol_median_radius"]
        alts_km = ext_bg.altitude.values / 1e3

        ax_ext.plot(ext_bg.values, alts_km, lw=1.5, color="steelblue",
                    label="ALI retrieved (bg)")
        ax_r.plot(r_bg.values, alts_km, lw=1.5, color="steelblue",
                  label="Retrieved (bg)")

        if cesm_bg is not None and "extinction_total" in cesm_bg:
            cesm_alt = cesm_bg["altitude_m"].values / 1e3
            ax_ext.plot(cesm_bg["extinction_total"].values, cesm_alt,
                        lw=1, ls=":", color="steelblue", label="CESM (bg)")

        if l2_inj is not None:
            ext_inj = l2_inj["stratospheric_aerosol_extinction_per_m"]
            r_inj   = l2_inj["stratospheric_aerosol_median_radius"]
            ax_ext.plot(ext_inj.values, alts_km, lw=1.5, ls="--",
                        color="firebrick", label="ALI retrieved (inj)")
            ax_r.plot(r_inj.values, alts_km, lw=1.5, ls="--",
                      color="firebrick", label="Retrieved (inj)")
            if cesm_inj is not None and "extinction_total" in cesm_inj:
                cesm_alt = cesm_inj["altitude_m"].values / 1e3
                ax_ext.plot(cesm_inj["extinction_total"].values, cesm_alt,
                            lw=1, ls=":", color="firebrick", label="CESM (inj)")

        ax_ext.set_xscale("log")
        ax_ext.set_xlabel("Extinction [m⁻¹]", fontsize=8)
        ax_r.set_xlabel("Radius [nm]",         fontsize=8)
        ax_ext.set_title(date, fontsize=9)
        ax_ext.set_ylabel("Altitude [km]",     fontsize=8)
        for ax in (ax_ext, ax_r):
            ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
            _ref_hline(ax, label=False)
            ax.grid(axis="x", alpha=0.25)
        if idx == 0:
            ax_ext.legend(fontsize=7)

    fig.suptitle("Fig 1 — ALI retrieved aerosol extinction & median radius",
                 fontsize=13)
    _save(fig, out_dir, "fig1_extinction_profiles.png")


def fig2_anomaly(months: list[str], out_dir: str) -> None:
    """
    Multi-panel figure: injection − background anomaly in extinction.
    Only months with injection output are shown.
    """
    inj_months = [d for d in months
                  if (Path(out_dir) / "injection" / d / "l2_injection.nc").exists()]
    if not inj_months:
        print("No injection months found — skipping fig2.")
        return

    n = len(inj_months)
    fig, axes = _make_grid(n, figsize_per=(4.0, 6))

    for idx, date in enumerate(inj_months):
        ax = axes[idx]
        try:
            l2_bg, l2_inj, _, _ = load_month(out_dir, date)
        except Exception:
            ax.set_title(f"{date}\n(error)", fontsize=8, color="red")
            continue

        ext_bg  = l2_bg["stratospheric_aerosol_extinction_per_m"].values
        ext_inj = l2_inj["stratospheric_aerosol_extinction_per_m"].values
        alts_km = l2_bg["stratospheric_aerosol_extinction_per_m"].altitude.values / 1e3
        anom    = (ext_inj - ext_bg) * 1e5

        ax.barh(alts_km, np.where(anom >= 0, anom, 0),
                height=0.4, color="firebrick", label="+")
        ax.barh(alts_km, np.where(anom <  0, anom, 0),
                height=0.4, color="steelblue", label="−")
        ax.axvline(0, color="k", lw=0.6)
        _ref_hline(ax, label=False)
        ax.set_xlim(-np.abs(anom).max() * 1.3, np.abs(anom).max() * 1.3)
        ax.set_xlabel("Δ ext [×10⁻⁵ m⁻¹]", fontsize=8)
        ax.set_ylabel("Altitude [km]",        fontsize=8)
        ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
        ax.set_title(date, fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Fig 2 — Extinction anomaly: Injection − Background", fontsize=13)
    _save(fig, out_dir, "fig2_anomaly.png")


def fig3_retrieval_diagnostics(months: list[str], out_dir: str) -> None:
    """
    Multi-panel: prior vs retrieved extinction and 1σ relative uncertainty.
    """
    n = len(months)
    nrows, ncols = _panel_grid(n)
    fig, all_axes = plt.subplots(
        nrows, ncols * 2,
        figsize=(3.8 * ncols * 2, 6 * nrows),
        sharey=True, constrained_layout=True,
    )
    all_axes_flat = np.array(all_axes).flatten()
    for i in range(n * 2, len(all_axes_flat)):
        all_axes_flat[i].set_visible(False)

    for idx, date in enumerate(months):
        row = idx // ncols
        col = idx %  ncols
        ax_prof  = all_axes[row, col * 2]
        ax_sigma = all_axes[row, col * 2 + 1]

        try:
            l2_bg, l2_inj, _, _ = load_month(out_dir, date)
        except Exception:
            ax_prof.set_title(f"{date}\n(error)", fontsize=8, color="red")
            continue

        for ds, label, color in [(l2_bg, "bg", "steelblue"),
                                  (l2_inj, "inj", "firebrick")]:
            if ds is None:
                continue
            ext   = ds["stratospheric_aerosol_extinction_per_m"]
            alts  = ext.altitude.values / 1e3
            prior = ds.get("stratospheric_aerosol_extinction_per_m_prior")
            sigma = ds.get("stratospheric_aerosol_extinction_per_m_1sigma_error")

            ax_prof.plot(ext.values, alts, lw=1.5, color=color,
                         label=f"{label} ret")
            if prior is not None:
                ax_prof.plot(prior.values, alts, lw=1, ls=":",
                             color=color, label=f"{label} prior")
            if sigma is not None:
                ax_sigma.plot(sigma.values / np.maximum(ext.values, 1e-12),
                              alts, lw=1.5, color=color, label=label)

        ax_prof.set_xscale("log")
        ax_prof.set_xlabel("Extinction [m⁻¹]", fontsize=8)
        ax_sigma.set_xlabel("Rel. 1σ",          fontsize=8)
        ax_prof.set_title(date, fontsize=9)
        ax_prof.set_ylabel("Altitude [km]",     fontsize=8)
        ax_sigma.axvline(1.0, color="grey", lw=0.6, ls=":")
        for ax in (ax_prof, ax_sigma):
            ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
            _ref_hline(ax, label=False)
            ax.grid(axis="x", alpha=0.25)
        if idx == 0:
            ax_prof.legend(fontsize=7)
            ax_sigma.legend(fontsize=7)

    fig.suptitle("Fig 3 — Retrieval diagnostics: prior vs retrieved & uncertainty",
                 fontsize=13)
    _save(fig, out_dir, "fig3_retrieval_diagnostics.png")


# ── Figures A–C: Sellitto et al. analogs using monthly CESM h0 output ─────────

def figA_profile_timeseries(ds_bg: xr.Dataset, ds_inj: xr.Dataset,
                             months: list[str], out_dir: str,
                             l2_data: dict) -> None:
    """
    Analog to Sellitto et al. Fig. 7.
    Rows: background | injection | anomaly.
    Columns: selected months (up to 6 shown to keep the figure readable).
    Solid: CESM EXTINCTdn (PR).  Dashed: ALI L2 retrieved (PO) where available.
    """
    # Show at most 6 evenly-spaced months
    step = max(1, len(months) // 6)
    show = months[::step][:6]

    alts_grid = np.linspace(ALT_MIN_KM, ALT_MAX_KM, 300)
    n_cols = len(show)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 10),
                              sharey=True, sharex=False,
                              constrained_layout=True)

    for ci, date in enumerate(show):
        ax_bg, ax_inj, ax_diff = axes[0, ci], axes[1, ci], axes[2, ci]

        # CESM monthly column profiles
        def _col_profile(ds):
            t_str = date + "-15"   # mid-month representative
            try:
                t_sel = ds.sel(time=t_str, method="nearest")
            except Exception:
                t_sel = ds.isel(time=0)
            col = t_sel.sel(lat=TANGENT_LAT, lon=TANGENT_LON, method="nearest")
            ext = col["EXTINCTdn"].values
            z   = col["Z3"].values / 1e3
            idx = np.argsort(z)
            return z[idx], ext[idx]

        z_bg,  ext_bg_c  = _col_profile(ds_bg)
        z_inj, ext_inj_c = _col_profile(ds_inj)

        m_bg  = (z_bg  >= ALT_MIN_KM) & (z_bg  <= ALT_MAX_KM)
        m_inj = (z_inj >= ALT_MIN_KM) & (z_inj <= ALT_MAX_KM)

        ax_bg.plot(ext_bg_c[m_bg],   z_bg[m_bg],
                   color="steelblue", lw=2, label="CESM PR")
        ax_inj.plot(ext_inj_c[m_inj], z_inj[m_inj],
                    color="firebrick", lw=2, label="CESM PR")

        # ALI L2 PO overlay where available
        if date in l2_data:
            l2_bg_ds, l2_inj_ds = l2_data[date]
            alt_l2 = l2_bg_ds["stratospheric_aerosol_extinction_per_m"].altitude.values / 1e3
            ax_bg.plot(
                l2_bg_ds["stratospheric_aerosol_extinction_per_m"].values, alt_l2,
                color="steelblue", lw=1.5, ls="--", label="ALI PO")
            if l2_inj_ds is not None:
                ax_inj.plot(
                    l2_inj_ds["stratospheric_aerosol_extinction_per_m"].values, alt_l2,
                    color="firebrick", lw=1.5, ls="--", label="ALI PO")

        # Anomaly panel
        ext_bg_i  = np.interp(alts_grid, z_bg[m_bg],   ext_bg_c[m_bg],   left=0, right=0)
        ext_inj_i = np.interp(alts_grid, z_inj[m_inj], ext_inj_c[m_inj], left=0, right=0)
        diff = (ext_inj_i - ext_bg_i) * 1e5
        ax_diff.fill_betweenx(alts_grid, 0, diff, where=diff >= 0,
                               color="firebrick", alpha=0.6)
        ax_diff.fill_betweenx(alts_grid, 0, diff, where=diff <  0,
                               color="steelblue", alpha=0.6)
        ax_diff.axvline(0, color="k", lw=0.8)

        for ax in (ax_bg, ax_inj):
            ax.set_xscale("log")
            ax.set_xlabel("Extinction [m⁻¹]", fontsize=8)
        ax_diff.set_xlabel("Δ ext\n[×10⁻⁵ m⁻¹]", fontsize=8)

        for ax in (ax_bg, ax_inj, ax_diff):
            ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
            _ref_hline(ax, label=False)
            if ci == 0:
                ax.set_ylabel("Altitude [km]")

        axes[0, ci].set_title(date, fontsize=9)
        if ci == 0:
            ax_bg.legend(fontsize=7)
            ax_inj.legend(fontsize=7)

    for ax, row_label in zip(axes[:, 0],
                              ["Background", "Injection", "Anomaly (inj−bg)"]):
        ax.annotate(row_label, xy=(-0.38, 0.5), xycoords="axes fraction",
                    rotation=90, va="center", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Fig A — CESM PR vs ALI PO extinction profiles\n"
        f"{TANGENT_LAT}°N, {TANGENT_LON}°E  "
        "(solid: CESM EXTINCTdn | dashed: ALI retrieved)",
        fontsize=11,
    )
    _save(fig, out_dir, "figA_profile_timeseries.png")


def figB_aod_timeseries(ds_bg: xr.Dataset, ds_inj: xr.Dataset,
                         out_dir: str) -> None:
    """
    Analog to Sellitto et al. Fig. 1d.
    Monthly stratospheric AOD time series across the full run.
    """
    print("  Computing monthly AOD time series...")
    col_bg  = ds_bg.sel( lat=TANGENT_LAT, lon=TANGENT_LON, method="nearest")
    col_inj = ds_inj.sel(lat=TANGENT_LAT, lon=TANGENT_LON, method="nearest")

    # Integrate EXTINCTdn × ΔZ between STRAT_MIN_KM and 35 km
    def _aod(col):
        z   = col["Z3"]             # (time, lev) m
        ext = col["EXTINCTdn"]      # (time, lev)
        dz  = z.differentiate("lev").pipe(np.abs)
        mask = (z >= STRAT_MIN_KM * 1e3) & (z <= 35e3)
        return (ext * dz * mask).sum("lev")

    aod_bg  = _aod(col_bg).compute()
    aod_inj = _aod(col_inj).compute()

    # Build a numeric x-axis from time coordinate
    times_bg  = aod_bg.time.values
    times_inj = aod_inj.time.values
    x_bg  = np.arange(len(times_bg))
    x_inj = np.arange(len(times_inj))
    xlabels = [str(t)[:7] for t in times_inj]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(x_inj) * 0.5), 8),
                              sharex=True, constrained_layout=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(x_bg,  aod_bg.values,  "o-", color="steelblue", lw=1.5,
                 label="Background")
    axes[0].plot(x_inj, aod_inj.values, "s-", color="firebrick", lw=1.5,
                 label="Injection")
    axes[0].fill_between(x_inj, aod_bg.values[:len(x_inj)],
                          aod_inj.values,
                          where=aod_inj.values >= aod_bg.values[:len(x_inj)],
                          color="firebrick", alpha=0.15, label="Injection excess")
    axes[0].set_ylabel(f"Stratospheric AOD\n({STRAT_MIN_KM:.0f}–35 km, 550 nm)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    n_common = min(len(x_inj), len(aod_bg))
    anom = aod_inj.values[:n_common] - aod_bg.values[:n_common]
    axes[1].bar(x_inj[:n_common], anom,
                color=["firebrick" if v >= 0 else "steelblue" for v in anom])
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set_ylabel("ΔAOD")
    axes[1].grid(alpha=0.3)

    axes[1].set_xticks(x_inj)
    axes[1].set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)

    fig.suptitle(
        f"Fig B — Monthly stratospheric AOD time series\n"
        f"{TANGENT_LAT}°N, {TANGENT_LON}°E  |  CESM EXTINCTdn 550 nm",
        fontsize=12,
    )
    _save(fig, out_dir, "figB_aod_timeseries.png")


def figC_hovmoller(ds_bg: xr.Dataset, ds_inj: xr.Dataset,
                   out_dir: str) -> None:
    """
    Analog to Sellitto et al. Fig. 6.
    Month–altitude Hovmöller of extinction anomaly at the injection column.
    """
    print("  Building month-altitude Hovmöller arrays...")
    col_bg  = ds_bg.sel( lat=TANGENT_LAT, lon=TANGENT_LON, method="nearest")
    col_inj = ds_inj.sel(lat=TANGENT_LAT, lon=TANGENT_LON, method="nearest")

    z_grid = np.arange(ALT_MIN_KM, ALT_MAX_KM + 0.5, 0.5)
    n_t    = min(len(ds_inj.time), len(ds_bg.time))

    ext_bg_arr  = col_bg["EXTINCTdn"].values[:n_t]   # (time, lev)
    ext_inj_arr = col_inj["EXTINCTdn"].values[:n_t]
    z_bg_arr    = col_bg["Z3"].values[:n_t]  / 1e3
    z_inj_arr   = col_inj["Z3"].values[:n_t] / 1e3

    ext_bg_i  = np.zeros((n_t, len(z_grid)))
    ext_inj_i = np.zeros((n_t, len(z_grid)))

    for i in range(n_t):
        idx_bg  = np.argsort(z_bg_arr[i])
        idx_inj = np.argsort(z_inj_arr[i])
        ext_bg_i[i]  = np.interp(z_grid, z_bg_arr[i, idx_bg],
                                  ext_bg_arr[i, idx_bg],   left=0, right=0)
        ext_inj_i[i] = np.interp(z_grid, z_inj_arr[i, idx_inj],
                                  ext_inj_arr[i, idx_inj], left=0, right=0)

    anom = (ext_inj_i - ext_bg_i) * 1e5   # ×10⁻⁵ m⁻¹
    vmax = max(np.nanpercentile(np.abs(anom), 98), 0.1)

    times   = ds_inj.time.values[:n_t]
    x       = np.arange(n_t)
    xlabels = [str(t)[:7] for t in times]

    fig, ax = plt.subplots(figsize=(max(10, n_t * 0.45), 6),
                            constrained_layout=True)
    cf = ax.contourf(x, z_grid, anom.T,
                     levels=np.linspace(-vmax, vmax, 21),
                     cmap="RdBu_r", extend="both")
    ax.contour(x, z_grid, anom.T, levels=[0],
               colors="k", linewidths=0.6)

    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("Δ extinction [×10⁻⁵ m⁻¹]  (injection − background)")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Altitude [km]")
    ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
    _ref_hline(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Fig C — Month–altitude Hovmöller of extinction anomaly\n"
        f"{TANGENT_LAT}°N, {TANGENT_LON}°E  |  CESM EXTINCTdn 550 nm",
        fontsize=12,
    )
    _save(fig, out_dir, "figC_hovmoller.png")


# ── time-series summary ────────────────────────────────────────────────────────

def fig_ts_summary(months: list[str], out_dir: str) -> None:
    """
    Three-panel time series from L2 output across all months:
      1. Stratospheric extinction burden (bg and inj)
      2. Peak retrieved median radius (bg and inj)
      3. Peak injection anomaly
    """
    dates, burden_bg, burden_inj, radius_bg, radius_inj, anom_peak = \
        [], [], [], [], [], []

    for date in months:
        try:
            l2_bg, l2_inj, _, _ = load_month(out_dir, date)
        except Exception:
            continue

        ext_bg  = l2_bg["stratospheric_aerosol_extinction_per_m"]
        r_bg    = l2_bg["stratospheric_aerosol_median_radius"]
        alts_km = ext_bg.altitude.values / 1e3

        dates.append(date)
        burden_bg.append(_strat_burden(ext_bg.values, alts_km))
        strat = alts_km >= STRAT_MIN_KM
        radius_bg.append(float(r_bg.values[strat].max()) if strat.any() else np.nan)

        if l2_inj is not None:
            ext_inj = l2_inj["stratospheric_aerosol_extinction_per_m"].values
            r_inj   = l2_inj["stratospheric_aerosol_median_radius"].values
            burden_inj.append(_strat_burden(ext_inj, alts_km))
            radius_inj.append(float(r_inj[strat].max()) if strat.any() else np.nan)
            anom_peak.append(float((ext_inj - ext_bg.values)[strat].max())
                             if strat.any() else np.nan)
        else:
            burden_inj.append(np.nan)
            radius_inj.append(np.nan)
            anom_peak.append(np.nan)

    if not dates:
        print("No months loaded — skipping time-series summary.")
        return

    x = np.arange(len(dates))
    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(x) * 0.5), 10),
                              sharex=True, constrained_layout=True)

    axes[0].plot(x, burden_bg,  "o-", color="steelblue", label="Background")
    axes[0].plot(x, burden_inj, "s--", color="firebrick", label="Injection")
    axes[0].set_ylabel("Strat. extinction burden\n[m⁻¹·km]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, radius_bg,  "o-",  color="steelblue", label="Background")
    axes[1].plot(x, radius_inj, "s--", color="firebrick",  label="Injection")
    axes[1].set_ylabel("Peak median radius [nm]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].bar(x, anom_peak,
                color=["firebrick" if (v is not None and not np.isnan(v) and v >= 0)
                       else "steelblue" for v in anom_peak])
    axes[2].axhline(0, color="k", lw=0.7)
    axes[2].set_ylabel("Peak Δ extinction [m⁻¹]")
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(dates, rotation=45, ha="right", fontsize=8)

    fig.suptitle("ALI retrieval time series — all months", fontsize=13)
    _save(fig, out_dir, "ts_summary.png")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ALI retrieval results across all months"
    )
    parser.add_argument("out_dir", nargs="?", default=OUT_DIR,
                        help="Root results directory (default: %(default)s)")
    parser.add_argument("--months", nargs="+", default=None, metavar="YYYY-MM",
                        help="Process only these months (default: all)")
    parser.add_argument("--skip-cesm", action="store_true",
                        help="Skip Figures A–C (no CESM h0 files needed)")
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    months  = discover_months(out_dir)

    if args.months:
        months = [m for m in months if m in args.months]
        if not months:
            sys.exit(f"None of the requested months found in {out_dir}")

    print(f"Found {len(months)} month(s): {months[0]} … {months[-1]}")

    # ── Figures 1–3: ALI L2 panel figures ────────────────────────────────
    print("\n── Figures 1–3: ALI L2 results ──")
    fig1_extinction_profiles(months, out_dir)
    fig2_anomaly(months, out_dir)
    fig3_retrieval_diagnostics(months, out_dir)

    # ── Time-series summary from L2 output ───────────────────────────────
    print("\n── Time-series summary ──")
    fig_ts_summary(months, out_dir)

    # ── Figures A–C: Sellitto analogs (requires CESM h0 files) ───────────
    if not args.skip_cesm:
        print("\n── Figures A–C: Sellitto et al. analogs ──")
        try:
            print("Loading CESM h0 files...")
            ds_bg  = load_cesm_h0(CESM_ARCHIVE_DIR, BG_CASENAME)
            ds_inj = load_cesm_h0(CESM_ARCHIVE_DIR, INJ_CASENAME)

            # Pre-load L2 data for the PO overlay in Fig A
            print("Pre-loading L2 data for PO overlay...")
            l2_data = {}
            for date in months:
                try:
                    l2_bg, l2_inj, _, _ = load_month(out_dir, date)
                    l2_data[date] = (l2_bg, l2_inj)
                except Exception:
                    pass

            figA_profile_timeseries(ds_bg, ds_inj, months, out_dir, l2_data)
            figB_aod_timeseries(ds_bg, ds_inj, out_dir)
            figC_hovmoller(ds_bg, ds_inj, out_dir)

        except FileNotFoundError as e:
            print(f"  Skipping Figures A–C: {e}")
            print("  Check CESM_ARCHIVE_DIR, BG_CASENAME, INJ_CASENAME in config.")
        except Exception as e:
            print(f"  Figures A–C failed: {e}")
            import traceback; traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
