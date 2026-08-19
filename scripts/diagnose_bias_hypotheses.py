#!/usr/bin/env python
"""
diagnose_bias_hypotheses.py
============================
Lightweight, single-machine diagnostic (NOT a SLURM production run) for the
magnitude-dependent positive bias seen in ALI L2 retrieved extinction
(retrieval overestimates at high true extinction, even pre-injection).

Isolates two competing hypotheses by re-running one BACKGROUND WACCM case
three ways -- baseline, no-ozone, single-mode-matched -- and comparing
retrieved-minus-true extinction vs. true extinction across the three:

1. Ozone mismatch
   -----------------
   Confirmed directly against the retrieval source
   (aliprocessing.processing.l1b_to_l2.process_l1b_to_l2_image): the L2
   state vector DOES include an "o3" absorber, but it's pinned close to a
   MIPAS climatological prior via heavy regularization
   (prior_influence=1e6, tikh_factor=2.5e4) -- it can't adapt far from
   MIPAS regardless of the true simulated ozone. If WACCM ozone departs
   from MIPAS, that mismatch can leak into the aerosol fit (mainly via the
   470/525 nm channels, near the Chappuis band; 745 nm is largely
   ozone-insensitive). The "no_ozone" variant zeroes WACCM ozone VMR
   before building the simulated atmosphere, removing the mismatch
   entirely (retrieval prior AND truth both ~ozone-free) -- if the bias
   shrinks, ozone mismatch is implicated.

2. Single-mode retrieval assumption
   -----------------------------------
   Confirmed directly against the retrieval source
   (aliprocessing.l2.optical.aerosol_median_radius_db): the retrieval's
   aerosol optical property is a SINGLE Mie database built with
   mode_width=1.6 (matching the accumulation mode only). The true
   simulated atmosphere is bimodal MAM4 sulfate (accumulation sigma_g=1.6
   + coarse sigma_g=1.2, per Mills et al. 2016; see cesm_hawc.waccm). The
   "single_mode" variant collapses both modes into ONE lognormal mode with
   mode_width=1.6 -- conserving total particle number and total mass, so
   only the DISTRIBUTION SHAPE assumption changes -- via
   combine_to_single_mode() below. If the bias disappears, the retrieval's
   fixed single-mode assumption is implicated.

Both hypotheses change the atmosphere fed into ``sk2_atmosphere``, which
changes ``front_end_radiance`` and everything downstream -- so all three
variants re-run the FULL chain (sk2_atmosphere -> front_end_radiance ->
l1b -> l2), not just l2 alone.

Usage
-----
Requires config.toml with [single] (for waccm_background) and [geometry]
(for tangent_lat/lon and fixed sza_deg/saa_deg -- explicit, not derived
from real orbit geometry, so there's no day/night filtering to worry
about). Run from the repo root:

    python scripts/diagnose_bias_hypotheses.py --config config.toml

Samples ``--n-profiles`` columns from the SAME background file, spread
across latitude around [geometry] tangent_lat (background aerosol
magnitude varies with latitude/altitude even with no injection -- plus
each column's own vertical profile already spans several orders of
magnitude in extinction, from near-zero to the layer peak). Each column is
run three ways, giving one (true, retrieved) extinction pair per
altitude level per variant, pooled across columns for the final plot.

Output: a CSV of all (variant, lat, altitude, true, retrieved) rows, and a
bias-vs-true-extinction plot (binned by magnitude) comparing the three
variants, both under --out-dir.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REFERENCE_WAVELENGTH_NM = 745.0  # matches the retrieval's aerosol nominal_wavelength
SINGLE_MODE_WIDTH = 1.6          # matches the retrieval's fixed Mie mode_width


def _make_no_ozone(profiles: dict) -> dict:
    """Variant for hypothesis 1: zero out WACCM ozone VMR before building
    the simulated atmosphere, so the forward-simulated radiance has no
    ozone mismatch against the retrieval's MIPAS-pinned o3 prior."""
    out = dict(profiles)
    out["vmr_o3"] = np.zeros_like(profiles["vmr_o3"])
    return out


def combine_to_single_mode(profiles: dict, mode_width: float = SINGLE_MODE_WIDTH) -> dict:
    """Variant for hypothesis 2: collapse the bimodal MAM4 sulfate
    (accumulation sigma_g=1.6 + coarse sigma_g=1.2) into ONE lognormal
    mode with ``mode_width`` -- matching the retrieval's fixed single-mode
    assumption exactly.

    Conserves total particle number and total mass (both physically
    meaningful) by inverting the same lognormal mass-moment relation
    cesm_hawc.waccm.mam4_lognormal() uses in the forward direction, so
    only the DISTRIBUTION SHAPE (mode_width) differs from the true
    bimodal atmosphere -- not the total burden.

    The result is stashed entirely under the "sulfate_a1_*" keys (which
    cesm_hawc.constituents.build_waccm_constituents always builds with a
    1.6-width Mie database, regardless of what's in
    profiles["sulfate_a1_sigma"] -- see _MODE_WIDTHS there), with
    "sulfate_a3_*" zeroed out so aerosol_coarse contributes ~nothing.
    """
    from cesm_hawc.waccm import RHO_SULFATE

    N1_m3 = profiles["sulfate_a1_N_cm3"] * 1e6
    N3_m3 = profiles["sulfate_a3_N_cm3"] * 1e6
    r1_m = profiles["sulfate_a1_r_um"] * 1e-6
    r3_m = profiles["sulfate_a3_r_um"] * 1e-6
    sg1 = profiles["sulfate_a1_sigma"]
    sg3 = profiles["sulfate_a3_sigma"]

    def mass_conc(N_m3, r_m, sigma_g):
        return N_m3 * (4.0 / 3.0) * np.pi * RHO_SULFATE * r_m ** 3 * np.exp(4.5 * np.log(sigma_g) ** 2)

    mass_total = mass_conc(N1_m3, r1_m, sg1) + mass_conc(N3_m3, r3_m, sg3)
    N_total_m3 = np.maximum(N1_m3 + N3_m3, 1.0)

    ln_sg = np.log(mode_width)
    r_combined_m = (
        mass_total / (N_total_m3 * (4.0 / 3.0) * np.pi * RHO_SULFATE * np.exp(4.5 * ln_sg ** 2))
    ) ** (1.0 / 3.0)

    out = dict(profiles)
    out["sulfate_a1_N_cm3"] = N_total_m3 / 1e6
    out["sulfate_a1_r_um"] = r_combined_m * 1e6
    out["sulfate_a1_sigma"] = mode_width
    out["sulfate_a3_N_cm3"] = np.zeros_like(N_total_m3)
    out["sulfate_a3_r_um"] = np.full_like(N_total_m3, 0.1)
    out["sulfate_a3_sigma"] = 1.2
    return out


VARIANT_BUILDERS = {
    "baseline": lambda p: p,
    "no_ozone": _make_no_ozone,
    "single_mode": combine_to_single_mode,
}


def run_one_variant(profiles, alt_grid_m, sim_geometry, simulator, noise_model,
                     ali_wavelengths, variant_name: str) -> pd.DataFrame | None:
    from cesm_hawc.convergence import extract_l2_native_diagnostics
    from cesm_hawc.simulation import DEFAULT_PRODUCTS, run_ali_simulation_from_profiles

    var_profiles = VARIANT_BUILDERS[variant_name](profiles)

    t0 = time.perf_counter()
    data, true_ext = run_ali_simulation_from_profiles(
        var_profiles, alt_grid_m, sim_geometry, simulator=simulator,
        products=DEFAULT_PRODUCTS, noise_model=noise_model,
        return_extinction=True, truth_wavelengths_nm=ali_wavelengths,
    )
    elapsed = time.perf_counter() - t0

    diag = extract_l2_native_diagnostics(data.get("l2"))
    log.info("  [%s] done in %.1fs (l2 iterations=%s, cost=%s)",
              variant_name, elapsed, diag["l2_num_iterations"], diag["l2_final_cost"])

    l2_ext = data["l2"]["stratospheric_aerosol_extinction_per_m"]
    ret_altitude = l2_ext["altitude"].values
    ret_values = l2_ext.values

    true_total = (true_ext["aerosol_accum_reference_extinction_per_m"]
                  + true_ext["aerosol_coarse_reference_extinction_per_m"])
    true_interp = np.interp(ret_altitude, alt_grid_m, true_total)

    return pd.DataFrame({
        "variant": variant_name,
        "altitude_m": ret_altitude,
        "true_extinction_per_m": true_interp,
        "retrieved_extinction_per_m": ret_values,
        "bias_per_m": ret_values - true_interp,
    })


def main(config_path: str, n_profiles: int, lat_span_deg: float, out_dir: str | None,
         use_noise: bool) -> None:
    from cesm_hawc.calibration import warm_calibration_database
    from cesm_hawc.config import load_config
    from cesm_hawc.constituents import warm_mode_databases
    from cesm_hawc.noise import default_noise_model
    from cesm_hawc.waccm import WACCMAtmosphere

    cfg = load_config(config_path)
    if cfg.single is None or cfg.geometry is None:
        raise SystemExit("config.toml needs [single] and [geometry] for this script.")
    s, geo, ins = cfg.single, cfg.geometry, cfg.instrument

    alt_grid_m = ins.altitude_grid_m()
    ali_wavelengths = np.array(ins.wavelengths_nm)
    out_dir = out_dir or os.path.join(s.out_dir, "diagnostic_bias_hypotheses")
    os.makedirs(out_dir, exist_ok=True)

    log.info("Warming calibration/Mie databases...")
    warm_calibration_database()
    warm_mode_databases()

    from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator
    simulator = IdealALISimulator()
    noise_model = default_noise_model() if use_noise else None

    lats = np.clip(
        geo.tangent_lat + np.linspace(-lat_span_deg / 2, lat_span_deg / 2, n_profiles),
        -89.0, 89.0,
    )
    log.info("Background case: %s", s.waccm_background)
    log.info("Sampling %d columns at lon=%.1f, lats=%s", n_profiles, geo.tangent_lon,
              np.round(lats, 1).tolist())

    waccm = WACCMAtmosphere(s.waccm_background, alt_grid_km=alt_grid_m / 1e3)

    sim_geometry = {
        "tangent_latitude": geo.tangent_lat,  # overwritten per-column below
        "tangent_longitude": geo.tangent_lon,
        "tangent_solar_zenith_angle": geo.sza_deg,
        "tangent_solar_azimuth_angle": geo.saa_deg,
        "altitude_grid": alt_grid_m,
        "polarization_states": ["I", "dolp"],
        "sample_wavelengths": ali_wavelengths,
        "time": pd.Timestamp(s.obs_time),
    }

    all_rows = []
    for i, lat in enumerate(lats):
        log.info("Column %d/%d: lat=%.2f", i + 1, len(lats), lat)
        profiles = waccm.get_column_profiles(lat, geo.tangent_lon, s.time_idx)
        col_geometry = {**sim_geometry, "tangent_latitude": float(lat)}

        for variant_name in VARIANT_BUILDERS:
            df = run_one_variant(profiles, alt_grid_m, col_geometry, simulator,
                                  noise_model, ali_wavelengths, variant_name)
            df["lat"] = float(lat)
            all_rows.append(df)

    results = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(out_dir, "bias_hypotheses_raw.csv")
    results.to_csv(csv_path, index=False)
    log.info("Saved %d rows to %s", len(results), csv_path)

    plot_path = os.path.join(out_dir, "bias_vs_true_extinction.png")
    make_bias_plot(results, plot_path)
    log.info("Saved plot to %s", plot_path)


def make_bias_plot(results: pd.DataFrame, plot_path: str, n_bins: int = 12) -> None:
    """Bias (retrieved - true) vs. true extinction magnitude, log-spaced
    bins, one line per variant. Restricted to true_extinction > 0 (the
    aerosol layer's vertical footprint) so log-binning is well-defined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = results[results["true_extinction_per_m"] > 0].copy()
    if df.empty:
        log.warning("No rows with positive true extinction -- skipping plot.")
        return

    lo, hi = df["true_extinction_per_m"].quantile([0.01, 0.99])
    bin_edges = np.logspace(np.log10(max(lo, 1e-10)), np.log10(hi), n_bins + 1)
    df["mag_bin"] = pd.cut(df["true_extinction_per_m"], bin_edges)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"baseline": "tab:red", "no_ozone": "tab:blue", "single_mode": "tab:green"}
    for variant_name, group in df.groupby("variant"):
        binned = group.groupby("mag_bin", observed=True).agg(
            bin_center=("true_extinction_per_m", "median"),
            bias_median=("bias_per_m", "median"),
            n=("bias_per_m", "size"),
        ).dropna()
        ax.plot(binned["bin_center"], binned["bias_median"], marker="o",
                 label=f"{variant_name} (n={int(binned['n'].sum())})",
                 color=colors.get(variant_name))

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(f"True extinction @ {REFERENCE_WAVELENGTH_NM:.0f} nm [m$^{{-1}}$]")
    ax.set_ylabel("Retrieved - True extinction [m$^{-1}$]")
    ax.set_title("Retrieval bias vs. true extinction magnitude, by hypothesis variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--n-profiles", type=int, default=4,
                         help="Number of columns to sample across latitude (default 4). "
                              "Each column x 3 variants = one full L2 retrieval "
                              "(100-600s each per benchmark_l2_retrieval.py) -- keep small.")
    parser.add_argument("--lat-span", type=float, default=30.0,
                         help="Total latitude spread [deg] sampled around [geometry] "
                              "tangent_lat (default 30.0).")
    parser.add_argument("--out-dir", default=None,
                         help="Output directory (default: <single.out_dir>/diagnostic_bias_hypotheses).")
    parser.add_argument("--no-noise", action="store_true",
                         help="Skip instrument noise (noiseless radiances) for a cleaner "
                              "systematic-bias isolation with few profiles. Default: use "
                              "cesm_hawc.noise.default_noise_model(), matching production.")
    args = parser.parse_args()
    main(args.config, args.n_profiles, args.lat_span, args.out_dir, not args.no_noise)
