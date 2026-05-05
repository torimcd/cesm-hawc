"""
plot_sai_aerosols.py
--------------------
Exploratory figures for CESM2/WACCM SAI output, focused on aerosol properties.
Designed to work with both background and injection cases, and to produce figures
comparable to those in Sellitto et al. (2026, egusphere-2026-919).

Usage:
    # Single case (background or injection):
    python plot_sai_aerosols.py --case sai_background_2035_001 \
                                --archivedir ~/projects/data/cesm_output \
                                --outdir ./figures

    # Both cases together — produces per-case panels AND difference panels:
    python plot_sai_aerosols.py --case sai_1.0Tg_2035_001 \
                                --bgcase sai_background_2035_001 \
                                --archivedir /scratch/vmcd/cesm/output/archive \
                                --outdir ./figures

Per-timestep figure types (SO2 map, SO4 burden map, extinction profile) are
produced as a single multi-panel figure — one subplot per month — with a shared
colorbar so magnitudes are directly comparable across time.

When --bgcase is supplied, additional difference figures (injection − background)
are produced for: SO2 map, SO4 burden map, SO4 burden Hovmöller, SO4 zonal mean,
AODSO4dn, and BURDENSO4dn.

Dependencies:
    pip install xarray netCDF4 matplotlib cartopy numpy scipy
"""

import argparse
import glob
import math
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

def load_h0(archivedir, casename, pattern="*.cam.h1.*.nc"):
    """Load all monthly-mean h0 files for a case into a single xarray Dataset."""
    path  = os.path.join(archivedir, casename, pattern)
    files = sorted(glob.glob(path))
    if not files:
        sys.exit(f"No h0 files found at: {path}")
    print(f"Loading {len(files)} h0 file(s) for {casename}")
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(
        files, combine="by_coords", decode_times=time_coder,
        data_vars="minimal", coords="minimal", compat="override"
    )
    return ds


def hybrid_to_pressure(ds, ps_name="PS"):
    """
    Compute 3-D pressure field (Pa) from hybrid sigma coefficients.
    Returns numpy array with shape (time, lev, lat, lon).
    """
    P0   = float(np.asarray(ds["P0"]).flat[0])
    hyam = ds["hyam"].isel(time=0) if "time" in ds["hyam"].dims else ds["hyam"]
    hybm = ds["hybm"].isel(time=0) if "time" in ds["hybm"].dims else ds["hybm"]
    PS   = ds[ps_name]
    pres = hyam * P0 + hybm * PS
    pres = pres.transpose("time", "lev", "lat", "lon")
    return np.asarray(pres)


def pressure_level_interp(field, pres, target_hPa):
    """
    Interpolate a 4-D field (time, lev, lat, lon) to a fixed pressure level.
    Uses linear interpolation in log-pressure space.
    Returns (time, lat, lon).
    """
    target_Pa = target_hPa * 100.0
    out = np.full(field.shape[:1] + field.shape[2:], np.nan)
    for t in range(field.shape[0]):
        for i in range(field.shape[2]):
            for j in range(field.shape[3]):
                p_col = pres[t, :, i, j]
                f_col = field[t, :, i, j]
                idx   = np.argsort(p_col)
                out[t, i, j] = np.interp(
                    np.log(target_Pa),
                    np.log(p_col[idx]), f_col[idx],
                    left=np.nan, right=np.nan
                )
    return out


def column_burden(ds, varname, pres):
    """
    Compute column burden (kg/m²) of a mixing ratio (kg/kg) field:
        integral( q * |dp| / g )
    """
    g     = 9.80665
    q     = np.asarray(ds[varname])
    p     = np.asarray(pres)
    dp    = np.diff(p, axis=1)
    q_mid = 0.5 * (q[:, :-1, :, :] + q[:, 1:, :, :])
    return np.sum(q_mid * np.abs(dp) / g, axis=1)


def match_times(inj_times, bg_times):
    """
    For each injection time, return the index of the nearest background time.
    Returns a list of integer indices into bg_times.
    """
    return [int(np.argmin(np.abs(bg_times - t))) for t in inj_times]


def panel_grid(n):
    """Return (nrows, ncols) for a roughly square panel grid of n subplots."""
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


# ── axes-grid helpers ─────────────────────────────────────────────────────────

def _map_axes_grid(n, figsize_per=(5, 2.8)):
    """Grid of Robinson-projection axes; returns (fig, flat axes array)."""
    nrows, ncols = panel_grid(n)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=False,
    )
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat


def _profile_axes_grid(n, figsize_per=(3.2, 5)):
    """Grid of plain axes for vertical profiles; returns (fig, flat axes array)."""
    nrows, ncols = panel_grid(n)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat


def _add_map_colorbar(fig, axes_active, im, label):
    cbar = fig.colorbar(im, ax=axes_active, orientation="horizontal",
                        fraction=0.03, pad=0.04, shrink=0.6)
    cbar.set_label(label)
    return cbar


# ── individual-case figure functions ──────────────────────────────────────────

def fig_so2_map(ds, pres, level_hPa=25.0, outdir=".", suffix=""):
    """
    Multi-panel SO2 map at ~level_hPa hPa — one subplot per month,
    shared colorbar across all panels.
    """
    if "SO2" not in ds:
        print("SO2 not found, skipping SO2 map.")
        return None   # return None so diff function can detect missing data

    field    = ds["SO2"].values
    so2_lev  = pressure_level_interp(field, pres, level_hPa)  # (time, lat, lon)
    data_ppb = so2_lev * 1e9

    ntimes      = data_ppb.shape[0]
    time_labels = [str(t)[:7] for t in ds.time.values]
    lat         = ds["lat"].values
    lon         = ds["lon"].values
    vmax        = max(np.nanpercentile(np.abs(data_ppb), 99), 1e-3)

    fig, axes = _map_axes_grid(ntimes)
    ims = []
    for t, ax in enumerate(axes[:ntimes]):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.2)
        im = ax.pcolormesh(lon, lat, data_ppb[t],
                           transform=ccrs.PlateCarree(),
                           cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(time_labels[t], fontsize=9)
        ims.append(im)

    fig.suptitle(f"SO₂ at ~{level_hPa} hPa{suffix}", fontsize=12, y=1.01)
    _add_map_colorbar(fig, axes[:ntimes], ims[0], "SO₂ (nmol/mol)")

    fname = os.path.join(outdir, f"so2_map_panel{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return data_ppb   # (time, lat, lon) — returned for use in diff function


def fig_so4_burden_map(ds, pres, outdir=".", suffix=""):
    """
    Multi-panel SO4 column burden map — one subplot per month, shared colorbar.
    """
    missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
    if missing:
        print(f"Missing {missing}, skipping SO4 burden map.")
        return None

    b1    = column_burden(ds, "so4_a1", pres)
    b3    = column_burden(ds, "so4_a3", pres)
    total = (b1 + b3) * 1e3   # g/m²

    ntimes      = total.shape[0]
    time_labels = [str(t)[:7] for t in ds.time.values]
    lat         = ds["lat"].values
    lon         = ds["lon"].values
    vmax        = max(np.nanpercentile(total, 99), 1e-6)

    fig, axes = _map_axes_grid(ntimes)
    ims = []
    for t, ax in enumerate(axes[:ntimes]):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        im = ax.pcolormesh(lon, lat, total[t],
                           transform=ccrs.PlateCarree(),
                           cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(time_labels[t], fontsize=9)
        ims.append(im)

    fig.suptitle(f"Total SO₄ column burden{suffix}", fontsize=12, y=1.01)
    _add_map_colorbar(fig, axes[:ntimes], ims[0], "SO₄ column burden (g m⁻²)")

    fname = os.path.join(outdir, f"so4_burden_map_panel{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return total   # (time, lat, lon) g/m²


def fig_so4_burden_timeseries(ds, pres, outdir=".", suffix="", label=None):
    """
    Latitude-time Hovmöller of zonal-mean SO4 column burden.
    """
    missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
    if missing:
        print(f"Missing {missing}, skipping SO4 Hovmöller.")
        return None

    b1    = column_burden(ds, "so4_a1", pres)
    b3    = column_burden(ds, "so4_a3", pres)
    total = (b1 + b3).mean(axis=2) * 1e3   # (time, lat) g/m², zonal mean

    lat         = ds["lat"].values
    times       = np.arange(len(ds.time))
    time_labels = [str(t)[:7] for t in ds.time.values]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(times, lat, total.T, cmap="plasma", vmin=0)
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
    return total, lat, times, time_labels   # returned for diff function


def fig_aod_timeseries(ds, outdir=".", suffix="", label=None):
    """Global-mean and tropical-mean AOD time series."""
    avail = [v for v in ["AODSO4dn", "AODVISdn", "AODVIS"] if v in ds]
    if not avail:
        print("No AOD variables found, skipping AOD time series.")
        return

    fig, axes = plt.subplots(len(avail), 1, figsize=(9, 3 * len(avail)),
                             sharex=True)
    if len(avail) == 1:
        axes = [axes]

    time_labels = [str(t)[:7] for t in ds.time.values]
    lat         = ds["lat"].values
    weights     = np.cos(np.deg2rad(lat))

    for ax, varname in zip(axes, avail):
        data   = ds[varname].values
        gm     = np.average(data.mean(axis=2), axis=1, weights=weights)
        trop_m = np.abs(lat) <= 30
        tm     = np.average(data.mean(axis=2)[:, trop_m], axis=1,
                            weights=weights[trop_m])
        ax.plot(time_labels, gm, "o-",  label="Global mean",      color="steelblue")
        ax.plot(time_labels, tm, "s--", label="Tropics (30S–30N)", color="tomato")
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


def fig_extinction_profile(ds, pres, lat_range=(-30, 30), outdir=".", suffix=""):
    """
    Multi-panel aerosol extinction profile (EXTINCTdn) — one subplot per month,
    shared x-axis range so magnitudes are directly comparable.
    """
    if "EXTINCTdn" not in ds:
        print("EXTINCTdn not found, skipping extinction profile.")
        return

    lat      = ds["lat"].values
    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
    ext      = ds["EXTINCTdn"].values   # (time, lev, lat, lon)

    ntimes      = ext.shape[0]
    time_labels = [str(t)[:7] for t in ds.time.values]

    profiles_trop, profiles_global, p_cols = [], [], []
    for t in range(ntimes):
        # ext[t] is (lev, lat, lon); index lat first, then mean over lat+lon axes
        profiles_trop.append(ext[t][:, lat_mask, :].mean(axis=(1, 2)))
        profiles_global.append(ext[t].mean(axis=(1, 2)))
        p_cols.append(pres[t].mean(axis=(1, 2)) / 100.0)   # hPa, (lev,)

    xmax = max(
        np.nanpercentile(np.concatenate(profiles_trop),   99),
        np.nanpercentile(np.concatenate(profiles_global), 99),
    ) * 1e3

    fig, axes = _profile_axes_grid(ntimes)
    for t, ax in enumerate(axes[:ntimes]):
        ax.semilogx(profiles_trop[t]   * 1e3, p_cols[t],
                    "b-",  lw=2, label=f"{lat_range[0]}–{lat_range[1]}°")
        ax.semilogx(profiles_global[t] * 1e3, p_cols[t],
                    "k--", lw=1, label="Global mean")
        ax.invert_yaxis()
        ax.set_ylim(100, 1)
        ax.set_xlim(right=xmax * 1.1)
        ax.set_xlabel("Extinction (×10⁻³ m⁻¹)", fontsize=8)
        ax.set_ylabel("Pressure (hPa)",           fontsize=8)
        ax.set_title(time_labels[t], fontsize=9)
        ax.grid(alpha=0.3, which="both")
        if t == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Aerosol extinction (550 nm){suffix}", fontsize=12)

    fname = os.path.join(outdir, f"extinction_profile_panel{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_so4_zonal_mean(ds, pres, outdir=".", suffix="", label=None):
    """
    Latitude-pressure cross-section of time-averaged zonal-mean SO4.
    """
    missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
    if missing:
        print(f"Missing {missing}, skipping SO4 zonal mean.")
        return None

    s1    = ds["so4_a1"].values
    s3    = ds["so4_a3"].values
    total = s1 + s3
    zm    = total.mean(axis=(0, 3))         # (lev, lat)
    p_zm  = pres.mean(axis=(0, 3)) / 100.0  # hPa (lev, lat)
    p_col = p_zm.mean(axis=1)              # representative (lev,)
    lat   = ds["lat"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(lat, p_col, zm * 1e9, levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="SO₄ (×10⁻⁹ kg/kg)")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_ylim(200, 1)
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Pressure (hPa)")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    title = f"Zonal-mean SO₄ (time avg){suffix}"
    if label:
        title = f"{label}: " + title
    ax.set_title(title)

    fname = os.path.join(outdir, f"so4_zonal_mean{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return zm, p_col, lat   # returned for diff function


# ── difference figure functions ───────────────────────────────────────────────

def _diff_map_panel(diffs, lat, lon, time_labels, suptitle, cbar_label,
                    fname, cmap="RdBu_r"):
    """
    Generic helper: plot a list of (lat, lon) difference arrays as a multi-panel
    map with a single shared symmetric colorbar.
    """
    ntimes = len(diffs)
    vmax   = max(np.nanpercentile(np.abs(np.stack(diffs)), 99), 1e-10)
    norm   = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = _map_axes_grid(ntimes)
    ims = []
    for t, ax in enumerate(axes[:ntimes]):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.2)
        im = ax.pcolormesh(lon, lat, diffs[t],
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, norm=norm)
        ax.set_title(time_labels[t], fontsize=9)
        ims.append(im)

    fig.suptitle(suptitle, fontsize=12, y=1.01)
    _add_map_colorbar(fig, axes[:ntimes], ims[0], cbar_label)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_diff_so2_map(ds_inj, pres_inj, ds_bg, pres_bg,
                     level_hPa=25.0, outdir="."):
    """
    Difference panel: SO2 at level_hPa (injection − background), one subplot
    per injection month, shared symmetric colorbar.
    """
    if "SO2" not in ds_inj or "SO2" not in ds_bg:
        print("SO2 missing from one dataset, skipping SO2 diff map.")
        return

    so2_inj = pressure_level_interp(ds_inj["SO2"].values, pres_inj, level_hPa) * 1e9
    so2_bg  = pressure_level_interp(ds_bg["SO2"].values,  pres_bg,  level_hPa) * 1e9

    inj_times = ds_inj.time.values
    bg_idx    = match_times(inj_times, ds_bg.time.values)
    diffs     = [so2_inj[t] - so2_bg[bg_idx[t]] for t in range(len(inj_times))]

    _diff_map_panel(
        diffs,
        lat         = ds_inj["lat"].values,
        lon         = ds_inj["lon"].values,
        time_labels = [str(t)[:7] for t in inj_times],
        suptitle    = f"SO₂ at ~{level_hPa} hPa: Injection − Background",
        cbar_label  = "ΔSO₂ (nmol/mol)",
        fname       = os.path.join(outdir, "so2_map_diff_panel.png"),
    )


def fig_diff_so4_burden_map(ds_inj, pres_inj, ds_bg, pres_bg, outdir="."):
    """
    Difference panel: SO4 column burden (injection − background), one subplot
    per injection month, shared symmetric colorbar.
    """
    for ds, label in [(ds_inj, "inj"), (ds_bg, "bg")]:
        missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
        if missing:
            print(f"Missing {missing} in {label}, skipping SO4 burden diff map.")
            return

    b_inj = (column_burden(ds_inj, "so4_a1", pres_inj)
             + column_burden(ds_inj, "so4_a3", pres_inj)) * 1e3   # g/m²
    b_bg  = (column_burden(ds_bg,  "so4_a1", pres_bg)
             + column_burden(ds_bg,  "so4_a3", pres_bg)) * 1e3

    inj_times = ds_inj.time.values
    bg_idx    = match_times(inj_times, ds_bg.time.values)
    diffs     = [b_inj[t] - b_bg[bg_idx[t]] for t in range(len(inj_times))]

    _diff_map_panel(
        diffs,
        lat         = ds_inj["lat"].values,
        lon         = ds_inj["lon"].values,
        time_labels = [str(t)[:7] for t in inj_times],
        suptitle    = "SO₄ column burden: Injection − Background",
        cbar_label  = "ΔSO₄ burden (g m⁻²)",
        fname       = os.path.join(outdir, "so4_burden_map_diff_panel.png"),
    )


def fig_diff_so4_hovmoller(ds_inj, pres_inj, ds_bg, pres_bg, outdir="."):
    """
    Difference Hovmöller: zonal-mean SO4 column burden (injection − background).
    Both cases are regridded to the injection time axis before differencing.
    Shared symmetric colorbar.
    """
    for ds, label in [(ds_inj, "inj"), (ds_bg, "bg")]:
        missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
        if missing:
            print(f"Missing {missing} in {label}, skipping SO4 Hovmöller diff.")
            return

    # zonal-mean column burden (time, lat) g/m²
    zm_inj = ((column_burden(ds_inj, "so4_a1", pres_inj)
               + column_burden(ds_inj, "so4_a3", pres_inj))
              .mean(axis=2) * 1e3)
    zm_bg  = ((column_burden(ds_bg,  "so4_a1", pres_bg)
               + column_burden(ds_bg,  "so4_a3", pres_bg))
              .mean(axis=2) * 1e3)

    inj_times = ds_inj.time.values
    bg_idx    = match_times(inj_times, ds_bg.time.values)
    diff      = zm_inj - zm_bg[bg_idx, :]   # (time, lat)

    lat         = ds_inj["lat"].values
    times       = np.arange(len(inj_times))
    time_labels = [str(t)[:7] for t in inj_times]

    vmax = max(np.nanpercentile(np.abs(diff), 99), 1e-10)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(times, lat, diff.T, cmap="RdBu_r", norm=norm)
    plt.colorbar(im, ax=ax, label="ΔSO₄ burden (g m⁻²)")
    ax.set_xticks(times)
    ax.set_xticklabels(time_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Zonal-mean SO₄ burden: Injection − Background")

    fname = os.path.join(outdir, "so4_burden_hovmoller_diff.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_diff_so4_zonal_mean(ds_inj, pres_inj, ds_bg, pres_bg, outdir="."):
    """
    Difference latitude-pressure cross-section: time-averaged zonal-mean SO4
    (injection − background). Shared symmetric colorbar.
    """
    for ds, label in [(ds_inj, "inj"), (ds_bg, "bg")]:
        missing = [v for v in ["so4_a1", "so4_a3"] if v not in ds]
        if missing:
            print(f"Missing {missing} in {label}, skipping SO4 zonal mean diff.")
            return

    def _zm(ds, pres):
        total = ds["so4_a1"].values + ds["so4_a3"].values
        return total.mean(axis=(0, 3))   # (lev, lat)

    zm_inj = _zm(ds_inj, pres_inj)
    zm_bg  = _zm(ds_bg,  pres_bg)
    diff   = zm_inj - zm_bg             # (lev, lat) — time-avg difference

    # use injection pressure grid for plotting axes
    p_col = pres_inj.mean(axis=(0, 2, 3)) / 100.0   # (lev,) hPa
    lat   = ds_inj["lat"].values

    vmax = max(np.nanpercentile(np.abs(diff * 1e9), 99), 1e-10)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(lat, p_col, diff * 1e9,
                     levels=20, cmap="RdBu_r", norm=norm)
    plt.colorbar(im, ax=ax, label="ΔSO₄ (×10⁻⁹ kg/kg)")
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_ylim(200, 1)
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Pressure (hPa)")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_title("Zonal-mean SO₄ (time avg): Injection − Background")

    fname = os.path.join(outdir, "so4_zonal_mean_diff.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_diff_aod_burden(ds_inj, ds_bg, outdir="."):
    """
    Difference panels for AODSO4dn and BURDENSO4dn (injection − background),
    one subplot per matched month, shared symmetric colorbar per variable.
    """
    inj_times = ds_inj.time.values
    bg_idx    = match_times(inj_times, ds_bg.time.values)

    for varname, units, scale in [
        ("AODSO4dn",    "ΔAOD SO₄",             1.0),
        ("BURDENSO4dn", "ΔSO₄ burden (mg m⁻²)", 1e6),
    ]:
        if varname not in ds_inj or varname not in ds_bg:
            continue

        diffs = [(ds_inj[varname].values[t] - ds_bg[varname].values[bg_idx[t]]) * scale
                 for t in range(len(inj_times))]

        _diff_map_panel(
            diffs,
            lat         = ds_inj["lat"].values,
            lon         = ds_inj["lon"].values,
            time_labels = [str(t)[:7] for t in inj_times],
            suptitle    = f"{varname}: Injection − Background",
            cbar_label  = units,
            fname       = os.path.join(outdir, f"{varname}_diff_panel.png"),
        )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot aerosol diagnostics from CESM2/WACCM SAI output"
    )
    parser.add_argument("--case",       required=True,
                        help="Injection (or single) case name")
    parser.add_argument("--bgcase",     default=None,
                        help="Background case name")
    parser.add_argument("--archivedir", default=".",
                        help="Root archive directory containing <case>/ subdirs")
    parser.add_argument("--outdir",     default="./figures",
                        help="Output directory for figures")
    parser.add_argument("--level_hPa", type=float, default=25.0,
                        help="Pressure level (hPa) for SO2 horizontal maps")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── load cases ────────────────────────────────────────────────────────────
    print(f"\nLoading case: {args.case}")
    ds   = load_h0(args.archivedir, args.case)
    pres = hybrid_to_pressure(ds)
    suffix = f"_{args.case}"

    ds_bg   = None
    pres_bg = None
    if args.bgcase:
        print(f"Loading background case: {args.bgcase}")
        ds_bg   = load_h0(args.archivedir, args.bgcase)
        pres_bg = hybrid_to_pressure(ds_bg)

    # ── per-case panel figures ────────────────────────────────────────────────
    for _ds, _pres, _sfx, _lbl in [
        (ds,    pres,    suffix,              args.case),
        (ds_bg, pres_bg, f"_{args.bgcase}",  args.bgcase),
    ]:
        if _ds is None:
            continue
        print(f"\nProducing panel figures for {_lbl} ({len(_ds.time)} months)...")
        fig_so2_map(_ds, _pres, level_hPa=args.level_hPa,
                    outdir=args.outdir, suffix=_sfx)
        fig_so4_burden_map(_ds, _pres, outdir=args.outdir, suffix=_sfx)
        fig_extinction_profile(_ds, _pres, outdir=args.outdir, suffix=_sfx)

        print(f"  Summary figures for {_lbl}...")
        fig_aod_timeseries(_ds, outdir=args.outdir, suffix=_sfx, label=_lbl)
        fig_so4_burden_timeseries(_ds, _pres, outdir=args.outdir,
                                  suffix=_sfx, label=_lbl)
        fig_so4_zonal_mean(_ds, _pres, outdir=args.outdir,
                           suffix=_sfx, label=_lbl)

    # ── difference figures (injection − background) ───────────────────────────
    if ds_bg is not None:
        print("\nProducing injection − background difference figures...")
        fig_diff_so2_map(ds, pres, ds_bg, pres_bg,
                         level_hPa=args.level_hPa, outdir=args.outdir)
        fig_diff_so4_burden_map(ds, pres, ds_bg, pres_bg, outdir=args.outdir)
        fig_diff_so4_hovmoller(ds, pres, ds_bg, pres_bg,  outdir=args.outdir)
        fig_diff_so4_zonal_mean(ds, pres, ds_bg, pres_bg, outdir=args.outdir)
        fig_diff_aod_burden(ds, ds_bg,                    outdir=args.outdir)

    print(f"\nDone. All figures saved to: {args.outdir}")


if __name__ == "__main__":
    main()
