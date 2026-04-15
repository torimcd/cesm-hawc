#!/usr/bin/env python
"""
plot_results.py
===============
Generate figures from saved ALI simulation NetCDF output.

Run after run_simulation.py has completed:

    python scripts/plot_results.py

Or point at a specific results directory:

    python scripts/plot_results.py ~/results/hawc_ali/
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── CONFIGURATION ──────────────────────────────────────────────────────────

OUT_DIR = os.path.expanduser("~/results/hawc_ali/")

# Altitude range for plots [km]
ALT_MIN_KM = 5.0
ALT_MAX_KM = 40.0

# Reference altitude line [km] (e.g. peak injection altitude)
REF_ALT_KM = 20.0

# ── END CONFIGURATION ──────────────────────────────────────────────────────


def load_results(out_dir: str) -> tuple:
    """Load L2 and CESM extinction NetCDF files from out_dir."""
    bg_path      = os.path.join(out_dir, "l2_background.nc")
    inj_path     = os.path.join(out_dir, "l2_injection.nc")
    cesm_bg_path = os.path.join(out_dir, "cesm_extinction_background.nc")
    cesm_inj_path = os.path.join(out_dir, "cesm_extinction_injection.nc")

    if not os.path.exists(bg_path):
        raise FileNotFoundError(
            f"Background L2 file not found: {bg_path}\n"
            "Run run_simulation.py first."
        )

    l2_bg      = xr.open_dataset(bg_path)
    l2_inj     = xr.open_dataset(inj_path) if os.path.exists(inj_path) else None
    cesm_bg    = xr.open_dataset(cesm_bg_path) if os.path.exists(cesm_bg_path) else None
    cesm_inj   = xr.open_dataset(cesm_inj_path) if os.path.exists(cesm_inj_path) else None

    if l2_inj is None:
        print("No injection L2 file found — plotting background only.")
    if cesm_bg is None:
        print("No CESM extinction file found — rerun run_simulation.py to generate it.")

    return l2_bg, l2_inj, cesm_bg, cesm_inj


def plot_extinction_profiles(l2_bg: xr.Dataset, l2_inj: xr.Dataset | None,
                              cesm_bg: xr.Dataset | None,
                              cesm_inj: xr.Dataset | None,
                              out_dir: str) -> None:
    """
    Figure 1: Retrieved aerosol extinction and median radius profiles.

    Left panel shows three lines for each scenario:
      - L2 retrieved    (solid)   — what the ALI retrieval produces
      - Forward model   (dashed)  — sum of ExtinctionScatterer constituents
      - CESM direct     (dotted)  — extinction computed directly from MAM4
                                    number density and radius profiles

    Comparing all three shows how faithfully the retrieval recovers the
    CESM atmospheric state and whether the forward model is consistent
    with the input profiles.
    """
    ext_bg  = l2_bg["stratospheric_aerosol_extinction_per_m"]
    r_bg    = l2_bg["stratospheric_aerosol_median_radius"]
    alts_km = ext_bg.altitude.values / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    # ── Background ────────────────────────────────────────────────────────
    axes[0].plot(ext_bg.values, alts_km,
                 lw=2, color="steelblue", label="Retrieved 745 nm (bg)")

    if cesm_bg is not None:
        cesm_alt_km = cesm_bg["altitude_m"].values / 1e3
        # EXTINCTdn is at 550 nm — closest available CESM wavelength to 745 nm.
        # EXTINCTNIRdn at 1020 nm is an exact ALI channel but typically weaker.
        axes[0].plot(cesm_bg["ext_550nm"].values, cesm_alt_km,
                     lw=1.5, ls=":", color="steelblue",
                     label="CESM EXTINCTdn 550 nm (bg)")

    axes[1].plot(r_bg.values, alts_km,
                 lw=2, color="steelblue", label="Retrieved (bg)")

    # ── Injection ─────────────────────────────────────────────────────────
    if l2_inj is not None:
        ext_inj = l2_inj["stratospheric_aerosol_extinction_per_m"]
        r_inj   = l2_inj["stratospheric_aerosol_median_radius"]
        alts_inj_km = ext_inj.altitude.values / 1e3

        axes[0].plot(ext_inj.values, alts_inj_km,
                     lw=2, ls="--", color="firebrick", label="Retrieved 745 nm (inj)")

        if cesm_inj is not None:
            cesm_alt_km = cesm_inj["altitude_m"].values / 1e3
            axes[0].plot(cesm_inj["ext_550nm"].values, cesm_alt_km,
                         lw=1.5, ls=":", color="firebrick",
                         label="CESM EXTINCTdn 550 nm (inj)")

        axes[1].plot(r_inj.values, alts_inj_km,
                     lw=2, ls="--", color="firebrick", label="Retrieved (inj)")

    axes[0].set_xlabel("Extinction @ 745 nm [m⁻¹]")
    axes[0].set_xscale("log")
    axes[1].set_xlabel("Retrieved median radius [nm]")

    for ax in axes:
        ax.set_ylabel("Altitude [km]")
        ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
        ax.axhline(REF_ALT_KM, color="grey", lw=0.7, ls=":")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("ALI retrieved aerosol profiles")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_extinction_profiles.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_anomaly(l2_bg: xr.Dataset, l2_inj: xr.Dataset,
                 out_dir: str) -> None:
    """
    Figure 2: Injection anomaly in retrieved extinction and median radius.

    Shows the difference (injection − background) as horizontal bar charts.
    """
    ext_bg  = l2_bg["stratospheric_aerosol_extinction_per_m"]
    ext_inj = l2_inj["stratospheric_aerosol_extinction_per_m"]
    r_bg    = l2_bg["stratospheric_aerosol_median_radius"]
    r_inj   = l2_inj["stratospheric_aerosol_median_radius"]

    anom_ext = (ext_inj.values - ext_bg.values) * 1e5   # ×10⁻⁵ m⁻¹
    anom_r   = r_inj.values - r_bg.values               # nm (already in nm)
    alts     = ext_bg.altitude.values / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    for ax, anom, xlabel in [
        (axes[0], anom_ext, "Δ extinction [×10⁻⁵ m⁻¹]"),
        (axes[1], anom_r,   "Δ median radius [nm]"),
    ]:
        ax.barh(alts, np.where(anom >= 0, anom, 0),
                height=0.4, color="firebrick", label="+anomaly")
        ax.barh(alts, np.where(anom < 0, anom, 0),
                height=0.4, color="steelblue", label="−anomaly")
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(REF_ALT_KM, color="grey", lw=0.7, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Altitude [km]")
        ax.set_ylim(ALT_MIN_KM, ALT_MAX_KM)
        ax.legend(fontsize=9)

    fig.suptitle("Injection anomaly (injection − background)")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_anomaly.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_retrieval_diagnostics(l2_bg: xr.Dataset, l2_inj: xr.Dataset | None,
                                out_dir: str) -> None:
    """
    Figure 3: Retrieval quality — prior vs retrieved extinction, and
    1-sigma retrieval uncertainty.

    Useful for assessing whether the retrieval is well-constrained.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    for ds, label, color in [
        (l2_bg,  "Background", "steelblue"),
        (l2_inj, "Injection",  "firebrick"),
    ]:
        if ds is None:
            continue
        ext      = ds["stratospheric_aerosol_extinction_per_m"]
        ext_prior = ds["stratospheric_aerosol_extinction_per_m_prior"]
        ext_sigma = ds["stratospheric_aerosol_extinction_per_m_1sigma_error"]
        alts_km  = ext.altitude.values / 1e3

        axes[0].plot(ext.values,       alts_km, lw=2,     color=color, label=f"{label} retrieved")
        axes[0].plot(ext_prior.values, alts_km, lw=1.5, ls=":", color=color, label=f"{label} prior")
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
        ax.axhline(REF_ALT_KM, color="grey", lw=0.7, ls=":")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Retrieval diagnostics")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_retrieval_diagnostics.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR

    print(f"Reading results from {out_dir}")
    l2_bg, l2_inj, cesm_bg, cesm_inj = load_results(out_dir)

    plot_extinction_profiles(l2_bg, l2_inj, cesm_bg, cesm_inj, out_dir)
    plot_retrieval_diagnostics(l2_bg, l2_inj, out_dir)
    if l2_inj is not None:
        plot_anomaly(l2_bg, l2_inj, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
