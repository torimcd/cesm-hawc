"""
plot_sai_aerosols.py
--------------------
Exploratory figures for CESM2/WACCM SAI output, focused on aerosol properties.
Designed to work with both background and injection cases, and to produce figures
comparable to those in Sellitto et al. (2026, egusphere-2026-919).

Usage:
    python plot_sai_aerosols.py --case sai_background_2035_001 \
                                --archivedir /scratch/vmcd/cesm/output/archive \
                                --outdir ./figures

    # Compare injection vs background:
    python plot_sai_aerosols.py --case sai_1.0Tg_2035_001 \
                                --bgcase sai_background_2035_001 \
                                --archivedir /scratch/vmcd/cesm/output/archive \
                                --outdir ./figures

Dependencies:
    pip install xarray netCDF4 matplotlib cartopy numpy scipy
"""

import argparse
import glob
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm


# ── helpers ───────────────────────────────────────────────────────────────────

def load_h0(archivedir, casename, pattern="*.cam.h0.*.nc"):
    """Load all monthly-mean h0 files for a case into a single xarray Dataset."""
    path = os.path.join(archivedir, casename, "atm", "hist", pattern)
    files = sorted(glob.glob(path))
    if not files:
        sys.exit(f"No h0 files found at: {path}")
    print(f"Loading {len(files)} h0 file(s) for {casename}")
    ds = xr.open_mfdataset(files, combine="by_coords", use_cftime=True)
    return ds


def hybrid_to_pressure(ds, ps_name="PS"):
    """
    Compute 3-D pressure field (Pa) from hybrid sigma coefficients.
    Returns pressure array with dims (time, lev, lat, lon).
    """
    P0 = float(ds["P0"])          # reference pressure in Pa
    hyam = ds["hyam"]             # (lev,)
    hybm = ds["hybm"]             # (lev,)
    PS   = ds[ps_name]            # (time, lat, lon)
    # broadcast: P = A*P0 + B*PS
    pres = hyam * P0 + hybm * PS  # (time, lev, lat, lon)
    return pres                   # Pa


def pressure_level_interp(field, pres, target_hPa):
    """
    Interpolate a 4-D field (time, lev, lat, lon) to a fixed pressure level.
    Uses simple linear interpolation in log-pressure space.
    Returns (time, lat, lon).
    """
    target_Pa = target_hPa * 100.0
    nlev = pres.shape[1]
    out = np.full(field.shape[:1] + field.shape[2:], np.nan)

    for t in range(field.shape[0]):
        for i in range(field.shape[2]):      # lat
            for j in range(field.shape[3]):  # lon
                p_col = pres[t, :, i, j]
                f_col = field[t, :, i, j]
                # pressure decreases with level index in CESM (surface=lev0)
                # sort ascending pressure so we can interpolate
                idx = np.argsort(p_col)
                p_s = p_col[idx]
                f_s = f_col[idx]
                out[t, i, j] = np.interp(
                    np.log(target_Pa),
                    np.log(p_s),
                    f_s,
                    left=np.nan, right=np.nan
                )
    return out


def zonal_mean(field):
    """Return zonal mean of (time, lev, lat, lon) -> (time, lev, lat)."""
    return field.mean(dim="lon")


def column_burden(ds, varname, pres):
    """
    Compute column burden (kg/m2) of a mixing ratio (kg/kg) field by
    integrating over pressure:  integral( q * dp / g )
    pres: (time, lev, lat, lon) in Pa
    """
    g = 9.80665  # m/s^2
    q = ds[varname].values          # (time, lev, lat, lon)
    # dp on each level (positive downward → thick from below to above)
    dp = np.diff(pres, axis=1)      # (time, lev-1, lat, lon)
    # mid-level q for integration
    q_mid = 0.5 * (q[:, :-1, :, :] + q[:, 1:, :, :])
    burden = np.sum(q_mid * np.abs(dp) / g, axis=1)  # (time, lat, lon)
    return burden


# ── figure functions ───────────────────────────────────────────────────────────

def fig_so2_map(ds, pres, time_idx=0, level_hPa=25.0, outdir=".", suffix=""):
    """
    Fig 1a-c style: SO2 mixing ratio at ~25 hPa as a global map.
    """
    if "SO2" not in ds:
        print("SO2 not found in dataset, skipping SO2 map.")
        return

    field = ds["SO2"].values  # mol/mol, (time, lev, lat, lon)
    so2_lev = pressure_level_interp(field, pres, level_hPa)  # (time, lat, lon)

    t = time_idx
    time_label = str(ds.time.values[t])[:7]

    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.Robinson()}
    )
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    lat = ds["lat"].values
    lon = ds["lon"].values
    data = so2_lev[t] * 1e9   # convert mol/mol → nmol/mol (ppb)

    vmax = max(np.nanpercentile(np.abs(data), 99), 1e-3)
    im = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
        vmin=0, vmax=vmax
    )
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05,
                 label="SO₂ (nmol/mol)")
    ax.set_title(f"SO₂ at ~{level_hPa} hPa  |  {time_label}{suffix}")

    fname = os.path.join(outdir, f"so2_map_{time_label}{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_so4_burden_map(ds, pres, time_idx=0, outdir=".", suffix=""):
    """
    Column burden of total SO4 aerosol (so4_a1 + so4_a3) as a global map.
    Comparable to Fig 1d (spatial snapshot at one time).
    """
    missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
    if missing:
        print(f"Missing variables {missing}, skipping SO4 burden map.")
        return

    b1 = column_burden(ds, "so4_a1", pres)
    b3 = column_burden(ds, "so4_a3", pres)
    total = b1 + b3   # (time, lat, lon)

    t = time_idx
    time_label = str(ds.time.values[t])[:7]

    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.Robinson()}
    )
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    lat = ds["lat"].values
    lon = ds["lon"].values
    data = total[t] * 1e3   # kg/m² → g/m²

    vmax = np.nanpercentile(data, 99)
    im = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        vmin=0, vmax=max(vmax, 1e-6)
    )
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05,
                 label="SO₄ column burden (g m⁻²)")
    ax.set_title(f"Total SO₄ column burden  |  {time_label}{suffix}")

    fname = os.path.join(outdir, f"so4_burden_map_{time_label}{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_so4_burden_timeseries(ds, pres, outdir=".", suffix="", label=None):
    """
    Fig 1d style: zonal-mean SO4 column burden as a latitude-time Hovmöller.
    """
    missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
    if missing:
        print(f"Missing variables {missing}, skipping SO4 Hovmöller.")
        return

    b1 = column_burden(ds, "so4_a1", pres)
    b3 = column_burden(ds, "so4_a3", pres)
    total = (b1 + b3).mean(axis=2)   # (time, lat) zonal mean

    lat  = ds["lat"].values
    times = np.arange(len(ds.time))
    time_labels = [str(t)[:7] for t in ds.time.values]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(
        times, lat, total.T * 1e3,   # g/m²
        cmap="plasma",
        vmin=0
    )
    plt.colorbar(im, ax=ax, label="SO₄ column burden (g m⁻²)")
    ax.set_xticks(times)
    ax.set_xticklabels(time_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Latitude (°)")
    title = f"Zonal-mean SO₄ column burden{suffix}"
    if label:
        title = f"{label}: " + title
    ax.set_title(title)

    fname = os.path.join(outdir, f"so4_burden_hovmoller{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_aod_timeseries(ds, outdir=".", suffix="", label=None):
    """
    Global-mean and tropical-mean AOD time series.
    Uses AODSO4dn (SO4-specific AOD, day+night) and AODVISdn.
    """
    avail = [v for v in ["AODSO4dn", "AODVISdn", "AODVIS"] if v in ds]
    if not avail:
        print("No AOD variables found, skipping AOD time series.")
        return

    fig, axes = plt.subplots(len(avail), 1, figsize=(9, 3*len(avail)), sharex=True)
    if len(avail) == 1:
        axes = [axes]

    time_labels = [str(t)[:7] for t in ds.time.values]
    lat = ds["lat"].values
    weights = np.cos(np.deg2rad(lat))

    for ax, varname in zip(axes, avail):
        data = ds[varname].values   # (time, lat, lon)
        # global mean
        gm = np.average(
            data.mean(axis=2),   # zonal mean → (time, lat)
            axis=1, weights=weights
        )
        # tropical mean (30S-30N)
        trop_mask = np.abs(lat) <= 30
        tm = np.average(
            data.mean(axis=2)[:, trop_mask],
            axis=1, weights=weights[trop_mask]
        )
        ax.plot(time_labels, gm, "o-", label="Global mean", color="steelblue")
        ax.plot(time_labels, tm, "s--", label="Tropics (30S-30N)", color="tomato")
        ax.set_ylabel(varname)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(alpha=0.3)

    title = f"AOD time series{suffix}"
    if label:
        title = f"{label}: " + title
    axes[0].set_title(title)

    fname = os.path.join(outdir, f"aod_timeseries{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_extinction_profile(ds, pres, time_idx=0, lat_range=(-30, 30),
                           outdir=".", suffix=""):
    """
    Zonal-mean aerosol extinction profile (EXTINCTdn) averaged over a
    latitude band. Useful for comparing stratospheric aerosol layer height.
    """
    if "EXTINCTdn" not in ds:
        print("EXTINCTdn not found, skipping extinction profile.")
        return

    lat = ds["lat"].values
    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])

    ext = ds["EXTINCTdn"].values    # (time, lev, lat, lon)
    p   = pres                      # (time, lev, lat, lon)

    t = time_idx
    time_label = str(ds.time.values[t])[:7]

    # average over lon and selected lat band
    ext_mean = ext[t, :, lat_mask, :].mean(axis=(1,))   # (lev, lon) -> mean over lon
    ext_mean = ext[t, :, :, :].mean(axis=(1, 2))        # (lev,)
    ext_trop = ext[t, :, lat_mask, :].mean(axis=(1, 2)) # (lev,)

    # pressure in hPa for plotting
    p_mean = p[t, :, lat_mask, :].mean(axis=(1, 2)) / 100.0  # hPa

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.semilogx(ext_trop * 1e3, p_mean, "b-", lw=2,
                label=f"{lat_range[0]}–{lat_range[1]}°")
    ax.semilogx(ext[t, :, :, :].mean(axis=(1, 2)) * 1e3,
                pres[t, :, :, :].mean(axis=(1, 2)) / 100.0,
                "k--", lw=1, label="Global mean")
    ax.invert_yaxis()
    ax.set_ylim(100, 1)
    ax.set_xlabel("Extinction (×10⁻³ m⁻¹)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(f"Aerosol extinction (550 nm)  |  {time_label}{suffix}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fname = os.path.join(outdir, f"extinction_profile_{time_label}{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_so4_zonal_mean(ds, pres, outdir=".", suffix="", label=None):
    """
    Latitude-pressure cross-section of zonal-mean SO4 burden (so4_a1+so4_a3),
    time-averaged. Useful for seeing vertical distribution of aerosol.
    """
    missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
    if missing:
        print(f"Missing {missing}, skipping zonal mean cross-section.")
        return

    s1 = ds["so4_a1"].values   # (time, lev, lat, lon) kg/kg
    s3 = ds["so4_a3"].values
    total = s1 + s3

    # time mean, zonal mean
    zm = total.mean(axis=(0, 3))  # (lev, lat)
    p_zm = pres.mean(axis=(0, 3)) / 100.0  # hPa (lev, lat)

    lat = ds["lat"].values
    lev_vals = ds["lev"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    # use representative pressure column (global mean)
    p_col = p_zm.mean(axis=1)  # (lev,)

    im = ax.contourf(
        lat, p_col, zm * 1e9,   # ppb equivalent by mass
        levels=20, cmap="YlOrRd"
    )
    plt.colorbar(im, ax=ax, label="SO₄ (×10⁻⁹ kg/kg)")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_ylim(200, 1)
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Pressure (hPa)")
    title = f"Zonal-mean SO₄ (time avg){suffix}"
    if label:
        title = f"{label}: " + title
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    fname = os.path.join(outdir, f"so4_zonal_mean{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_injection_vs_background(ds_inj, ds_bg, pres_inj, pres_bg,
                                 outdir=".", time_idx=0):
    """
    Difference plots (injection - background) for key aerosol variables.
    Requires both datasets to have the same grid.
    """
    time_label = str(ds_inj.time.values[time_idx])[:7]
    suffix = f"_diff_{time_label}"

    for varname, units, scale in [
        ("AODSO4dn", "ΔAOD SO₄", 1.0),
        ("BURDENSO4dn", "ΔSO₄ burden (mg m⁻²)", 1e6),
    ]:
        if varname not in ds_inj or varname not in ds_bg:
            continue

        # find matching time index in background
        inj_time = ds_inj.time.values[time_idx]
        bg_times = ds_bg.time.values
        bg_idx = np.argmin(np.abs(bg_times - inj_time))

        diff = (ds_inj[varname].values[time_idx] -
                ds_bg[varname].values[bg_idx]) * scale

        lat = ds_inj["lat"].values
        lon = ds_inj["lon"].values
        vmax = np.nanpercentile(np.abs(diff), 99)
        if vmax == 0:
            vmax = 1e-10

        fig, ax = plt.subplots(
            figsize=(10, 5),
            subplot_kw={"projection": ccrs.Robinson()}
        )
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(
            lon, lat, diff,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r", norm=norm
        )
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05,
                     label=units)
        ax.set_title(f"{varname}: Injection − Background  |  {time_label}")

        fname = os.path.join(outdir, f"{varname}{suffix}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot aerosol diagnostics from CESM2/WACCM SAI output"
    )
    parser.add_argument("--case",       required=True,
                        help="Case name (injection or background)")
    parser.add_argument("--bgcase",     default=None,
                        help="Background case name (for difference plots)")
    parser.add_argument("--archivedir", default=".",
                        help="Root archive directory (contains <case>/atm/hist/)")
    parser.add_argument("--outdir",     default="./figures",
                        help="Output directory for figures")
    parser.add_argument("--level_hPa", type=float, default=25.0,
                        help="Pressure level (hPa) for horizontal maps")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading case: {args.case}")
    ds = load_h0(args.archivedir, args.case)
    pres = hybrid_to_pressure(ds).values   # (time, lev, lat, lon) Pa

    suffix = f"_{args.case}"

    # ── per-timestep figures ───────────────────────────────────────────────────
    ntimes = len(ds.time)
    print(f"\nProducing figures for {ntimes} time step(s)...")

    for t in range(ntimes):
        time_label = str(ds.time.values[t])[:7]
        print(f"\n  [{t+1}/{ntimes}] {time_label}")
        fig_so2_map(ds, pres, time_idx=t,
                    level_hPa=args.level_hPa,
                    outdir=args.outdir, suffix=suffix)
        fig_so4_burden_map(ds, pres, time_idx=t,
                           outdir=args.outdir, suffix=suffix)
        fig_extinction_profile(ds, pres, time_idx=t,
                               outdir=args.outdir, suffix=suffix)

    # ── multi-timestep / summary figures ──────────────────────────────────────
    print("\nProducing summary figures...")
    fig_aod_timeseries(ds, outdir=args.outdir, suffix=suffix, label=args.case)
    fig_so4_burden_timeseries(ds, pres, outdir=args.outdir,
                              suffix=suffix, label=args.case)
    fig_so4_zonal_mean(ds, pres, outdir=args.outdir,
                       suffix=suffix, label=args.case)

    # ── comparison with background ─────────────────────────────────────────────
    if args.bgcase:
        print(f"\nLoading background case: {args.bgcase}")
        ds_bg   = load_h0(args.archivedir, args.bgcase)
        pres_bg = hybrid_to_pressure(ds_bg).values

        print("Producing injection − background difference figures...")
        for t in range(ntimes):
            fig_injection_vs_background(
                ds, ds_bg, pres, pres_bg,
                outdir=args.outdir, time_idx=t
            )

    print(f"\nDone. All figures saved to: {args.outdir}")


if __name__ == "__main__":
    main()
