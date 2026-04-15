"""
cesm_hawc.waccm
===============
Read CESM2/WACCM CAM history output and extract single-column atmospheric
profiles for use with the HAWC ALI simulator.

Notes
-----
Tested with BWSSP245 / TSMLT compsets using MAM4 aerosol. The following
variables must be present in the h0 file (add to fincl in user_nl_cam):

    T, Q, PS, hyam, hybm        -- state
    O3, NO2, H2O, SO2           -- gas chemistry (mol/mol)
    so4_a1, so4_a3              -- sulfate mass mixing ratio (kg/kg)
    num_a1, num_a3              -- aerosol number mixing ratio (#/kg)

MAM4 mode sigma_g values (WACCM/BWSSP245)
------------------------------------------
    Accumulation (_a1): sigma_g = 1.8
    Coarse (_a3):       sigma_g = 1.2   ← WACCM-specific (Mills et al. 2016)
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# ── Physical constants ─────────────────────────────────────────────────────
R_DRY = 287.058       # J kg⁻¹ K⁻¹  dry air gas constant
R_H2O = 461.5         # J kg⁻¹ K⁻¹  water vapour gas constant
G0    = 9.80665       # m s⁻²       standard gravity
NA    = 6.02214076e23 # mol⁻¹       Avogadro constant
M_AIR = 0.028966      # kg mol⁻¹    mean molar mass of dry air
M_H2O = 0.018015      # kg mol⁻¹    molar mass of water

# Hydrated H₂SO₄/H₂O stratospheric aerosol density [kg m⁻³]
RHO_SULFATE = 1600.0

# MAM4 modal sigma_g values for WACCM/BWSSP245
MAM4_SIGMA = {"a1": 1.8, "a2": 1.6, "a3": 1.2}


# ── Atmospheric physics helpers ────────────────────────────────────────────

def hybrid_to_pressure(hyam: np.ndarray, hybm: np.ndarray,
                        p0: float, ps: float) -> np.ndarray:
    """Convert hybrid sigma-pressure coefficients to pressure [Pa]."""
    return hyam * p0 + hybm * ps


def pressure_to_altitude(pressure: np.ndarray, temperature: np.ndarray,
                          ps: float, z_surface: float = 0.0) -> np.ndarray:
    """
    Hydrostatic integration from pressure to geometric altitude [m].

    Parameters
    ----------
    pressure    : [Pa]  pressure on model levels
    temperature : [K]   temperature on model levels
    ps          : [Pa]  surface pressure
    z_surface   : [m]   surface elevation, default 0

    Returns
    -------
    altitude : [m]  geometric altitude of each model level
    """
    nlev = len(pressure)
    flip = pressure[0] < pressure[-1]
    if flip:
        pressure    = pressure[::-1].copy()
        temperature = temperature[::-1].copy()

    p_half = np.empty(nlev + 1)
    p_half[0]    = ps
    p_half[nlev] = pressure[-1] / 2.0
    for k in range(1, nlev):
        p_half[k] = np.sqrt(pressure[k - 1] * pressure[k])

    altitude = np.empty(nlev)
    z = z_surface
    for k in range(nlev):
        dz          = (R_DRY * temperature[k] / G0) * np.log(p_half[k] / p_half[k + 1])
        altitude[k] = z + dz / 2.0
        z          += dz

    return altitude[::-1] if flip else altitude


def blend_h2o(q_vmr: np.ndarray, chem_h2o: np.ndarray,
              pressure: np.ndarray, join_pa: float = 10000.0) -> np.ndarray:
    """
    Blend dynamics Q (troposphere) with chemistry H₂O (stratosphere).

    Uses a cosine taper centred at join_pa with a transition width of one
    decade in log-pressure. Default join_pa = 100 hPa (~tropopause).
    """
    merged = np.empty_like(q_vmr)
    lp0, tw = np.log(join_pa), np.log(10.0)
    for k, lp in enumerate(np.log(pressure)):
        if lp >= lp0 + tw / 2:
            w = 1.0
        elif lp <= lp0 - tw / 2:
            w = 0.0
        else:
            w = 0.5 * (1.0 + np.cos(np.pi * (lp0 - lp) / tw))
        merged[k] = w * q_vmr[k] + (1.0 - w) * chem_h2o[k]
    return merged


def mam4_lognormal(so4_mmr: np.ndarray, num_per_kg: np.ndarray,
                   n_air_cm3: np.ndarray, sigma_g: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive lognormal median radius [μm] and number density [cm⁻³] from
    MAM4 mass and number mixing ratios.

    Uses the lognormal mass moment relation:
        mass_conc = N · (4/3)π ρ · r_m³ · exp(4.5 · ln(σ_g)²)

    Parameters
    ----------
    so4_mmr    : [kg/kg]  sulfate mass mixing ratio
    num_per_kg : [#/kg]   number mixing ratio
    n_air_cm3  : [cm⁻³]  air number density
    sigma_g    : float    geometric standard deviation

    Returns
    -------
    r_um  : [μm]   lognormal median radius per level
    N_cm3 : [cm⁻³] number concentration per level
    """
    rho_air   = (n_air_cm3 * 1e6) * M_AIR / NA          # [kg m⁻³]
    mass_conc = np.maximum(so4_mmr, 0.0) * rho_air       # [kg m⁻³]
    N_m3      = np.maximum(num_per_kg, 0.0) * rho_air    # [# m⁻³]
    N_m3      = np.maximum(N_m3, 1.0)                    # avoid divide-by-zero

    ln_sg = np.log(sigma_g)
    r_m3  = mass_conc / (N_m3 * (4.0 / 3.0) * np.pi * RHO_SULFATE
                         * np.exp(4.5 * ln_sg ** 2))
    r_m3  = np.maximum(r_m3, (1e-9) ** 3)               # floor at 1 nm

    return r_m3 ** (1.0 / 3.0) * 1e6, N_m3 * 1e-6      # [μm], [cm⁻³]


# ── Main class ─────────────────────────────────────────────────────────────

class WACCMAtmosphere:
    """
    Read a CESM2/WACCM CAM history NetCDF file and extract single-column
    atmospheric profiles for the HAWC ALI simulator.

    Parameters
    ----------
    filepath : str or list of str
        Path(s) to WACCM CAM h0 NetCDF file(s).
    alt_grid_km : array-like, optional
        Output altitude grid [km]. Default: 0–65 km in 1 km steps.
    z_surface : float, optional
        Surface elevation [m] for hydrostatic integration. Default 0.
    h2o_join_hpa : float, optional
        Pressure [hPa] at which dynamics Q blends into chemistry H₂O.
        Default 100 hPa (~tropopause).

    Examples
    --------
    >>> atm = WACCMAtmosphere("run.cam.h0.2035-02.nc")
    >>> profiles = atm.get_column_profiles(lat=30.6, lon=180.0, time_index=0)
    >>> burden = atm.sulfate_column_burden(lat=30.6, lon=180.0)
    """

    def __init__(self, filepath, alt_grid_km=None, z_surface=0.0,
                 h2o_join_hpa=100.0):
        if isinstance(filepath, (str, bytes)):
            self.ds = xr.open_dataset(filepath, engine="netcdf4",
                                       chunks={"time": 1}, decode_times=True)
        else:
            self.ds = xr.open_mfdataset(filepath, engine="netcdf4",
                                         chunks={"time": 1},
                                         combine="by_coords", decode_times=True)

        self.z_surface   = z_surface
        self.h2o_join_pa = h2o_join_hpa * 100.0

        if alt_grid_km is None:
            self.alt_grid_m = np.arange(0.0, 65001.0, 1000.0)
        else:
            self.alt_grid_m = np.asarray(alt_grid_km) * 1e3

        self._check_required_vars()

    def _check_required_vars(self):
        have = set(self.ds.data_vars) | set(self.ds.coords)
        required = {"T", "Q", "PS", "hyam", "hybm"}
        missing  = required - have
        if missing:
            raise ValueError(f"Missing required WACCM variables: {missing}")
        for v in ["O3", "NO2", "H2O", "SO2", "so4_a1", "so4_a3",
                  "num_a1", "num_a3"]:
            if v not in have:
                warnings.warn(f"'{v}' not in file — add to fincl in user_nl_cam.")

    def _p0(self) -> float:
        if "P0" in self.ds:
            return float(self.ds["P0"].values)
        warnings.warn("P0 not found; assuming 100 000 Pa.")
        return 100_000.0

    def _interp(self, alt_m: np.ndarray, values: np.ndarray,
                log: bool = False) -> np.ndarray:
        """Interpolate a profile onto self.alt_grid_m."""
        idx = np.argsort(alt_m)
        a, v = alt_m[idx], values[idx]
        if log:
            vmin   = v[v > 0].min() if np.any(v > 0) else 1e-30
            v_safe = np.where(v > 0, v, vmin * 1e-10)
            f = interp1d(a, np.log(v_safe), kind="linear",
                         bounds_error=False,
                         fill_value=(np.log(v_safe[0]), np.log(v_safe[-1])))
            return np.exp(f(self.alt_grid_m))
        f = interp1d(a, v, kind="linear", bounds_error=False,
                     fill_value=(v[0], v[-1]))
        return f(self.alt_grid_m)

    def get_column_profiles(self, lat: float, lon: float,
                             time_index: int = 0) -> dict:
        """
        Extract and interpolate one WACCM column onto the altitude grid.

        Parameters
        ----------
        lat        : float  target latitude [degrees], nearest-neighbour
        lon        : float  target longitude [degrees], nearest-neighbour
        time_index : int    time slice index (0-based)

        Returns
        -------
        dict with keys (all arrays on self.alt_grid_m unless noted):

        altitudes_m         [m]       altitude grid
        pressure_pa         [Pa]      pressure (log-interpolated)
        temperature_k       [K]       temperature
        specific_humidity   [kg/kg]   specific humidity
        vmr_o3              [mol/mol] ozone VMR
        vmr_no2             [mol/mol] NO₂ VMR (zeros if not in file)
        vmr_h2o             [mol/mol] H₂O blended Q + chemistry
        vmr_so2             [mol/mol] gas-phase SO₂ (precursor diagnostic)
        n_air_cm3           [cm⁻³]   air number density
        sulfate_a1_N_cm3    [cm⁻³]   accumulation mode number
        sulfate_a1_r_um     [μm]     accumulation mode median radius
        sulfate_a1_sigma    float    1.8 (scalar)
        sulfate_a3_N_cm3    [cm⁻³]   coarse mode number  ← ALI primary signal
        sulfate_a3_r_um     [μm]     coarse mode median radius
        sulfate_a3_sigma    float    1.2 (scalar, WACCM-specific)
        """
        col  = self.ds.isel(time=time_index).sel(lat=lat, lon=lon, method="nearest")
        p0   = self._p0()
        ps   = float(col["PS"].values)
        hyam = col["hyam"].values
        hybm = col["hybm"].values
        T    = col["T"].values
        Q    = col["Q"].values

        pressure = hybrid_to_pressure(hyam, hybm, p0, ps)
        T_v      = T * (1.0 + (R_H2O / R_DRY - 1.0) * Q)   # virtual temperature
        altitude = pressure_to_altitude(pressure, T_v, ps, self.z_surface)

        # Air number density [cm⁻³]: p / (k_B · T) · 1e-6
        n_air = (pressure / (1.380649e-23 * T)) * 1e-6

        # H₂O: blend dynamics Q with chemistry H₂O across the tropopause
        q_vmr = Q * (M_AIR / M_H2O)
        h2o_vmr = (blend_h2o(q_vmr, col["H2O"].values, pressure, self.h2o_join_pa)
                   if "H2O" in col else q_vmr)

        def chem_vmr(var):
            if var in col:
                return np.maximum(col[var].values, 0.0)
            warnings.warn(f"'{var}' not in file; returning zeros.")
            return np.zeros_like(T)

        # MAM4 sulfate: derive lognormal parameters from mass + number
        def sulfate_mode(suffix):
            so4v, numv = f"so4_{suffix}", f"num_{suffix}"
            if so4v not in col or numv not in col:
                return {"N_cm3": np.zeros_like(T),
                        "r_um":  np.full_like(T, 0.1),
                        "sigma_g": MAM4_SIGMA[suffix]}
            r_um, N_cm3 = mam4_lognormal(
                col[so4v].values, col[numv].values, n_air, MAM4_SIGMA[suffix]
            )
            return {"N_cm3": N_cm3, "r_um": r_um, "sigma_g": MAM4_SIGMA[suffix]}

        a1 = sulfate_mode("a1")
        a3 = sulfate_mode("a3")

        gi = self._interp
        return {
            "altitudes_m":       self.alt_grid_m,
            "pressure_pa":       gi(altitude, pressure,           log=True),
            "temperature_k":     gi(altitude, T,                  log=False),
            "specific_humidity": gi(altitude, Q,                  log=False),
            "vmr_o3":            gi(altitude, chem_vmr("O3"),     log=True),
            "vmr_no2":           gi(altitude, chem_vmr("NO2"),    log=True),
            "vmr_h2o":           gi(altitude, h2o_vmr,            log=True),
            "vmr_so2":           gi(altitude, chem_vmr("SO2"),    log=True),
            "n_air_cm3":         gi(altitude, n_air,              log=True),
            "sulfate_a1_N_cm3":  gi(altitude, a1["N_cm3"],        log=True),
            "sulfate_a1_r_um":   gi(altitude, a1["r_um"],         log=False),
            "sulfate_a1_sigma":  a1["sigma_g"],
            "sulfate_a3_N_cm3":  gi(altitude, a3["N_cm3"],        log=True),
            "sulfate_a3_r_um":   gi(altitude, a3["r_um"],         log=False),
            "sulfate_a3_sigma":  a3["sigma_g"],
        }

    def sulfate_column_burden(self, lat: float, lon: float,
                               time_index: int = 0,
                               alt_range_km: tuple = (15.0, 35.0)) -> dict:
        """
        Compute stratospheric sulfate column burden and peak aerosol properties.

        Parameters
        ----------
        lat, lon      : column coordinates [degrees]
        time_index    : time slice index
        alt_range_km  : (lower, upper) altitude bounds [km]

        Returns
        -------
        dict with:
            burden_mg_m2  [mg m⁻²]  SO₄ column burden (a1 + a3 modes)
            N_column_cm2  [cm⁻²]    number column
            peak_alt_km   [km]      altitude of peak number concentration
            peak_r_um     [μm]      median radius at peak
            dominant_mode str       "a1" (fresh) or "a3" (aged)
        """
        p    = self.get_column_profiles(lat, lon, time_index)
        alt  = p["altitudes_m"]
        lo, hi = alt_range_km[0] * 1e3, alt_range_km[1] * 1e3
        mask = (alt >= lo) & (alt <= hi)

        if not np.any(mask):
            return {"burden_mg_m2": 0.0, "N_column_cm2": 0.0,
                    "peak_alt_km": 0.0, "peak_r_um": 0.0,
                    "dominant_mode": "none"}

        dz = np.gradient(alt)

        def mode_mass_kgm3(N_cm3, r_um, sg):
            ln_sg = np.log(sg)
            return (N_cm3 * 1e6 * (4.0 / 3.0) * np.pi * RHO_SULFATE
                    * (r_um * 1e-6) ** 3 * np.exp(4.5 * ln_sg ** 2))

        mass = (mode_mass_kgm3(p["sulfate_a1_N_cm3"], p["sulfate_a1_r_um"],
                                p["sulfate_a1_sigma"])
              + mode_mass_kgm3(p["sulfate_a3_N_cm3"], p["sulfate_a3_r_um"],
                                p["sulfate_a3_sigma"]))

        burden  = np.sum((mass * dz)[mask]) * 1e6        # kg m⁻² → mg m⁻²
        N_total = p["sulfate_a1_N_cm3"] + p["sulfate_a3_N_cm3"]
        N_col   = np.sum((N_total * dz * 100.0)[mask])   # cm⁻³·m → cm⁻²

        peak_idx = np.argmax(N_total * mask)
        dominant = ("a3" if p["sulfate_a3_N_cm3"][peak_idx]
                          > p["sulfate_a1_N_cm3"][peak_idx] else "a1")

        return {
            "burden_mg_m2":  burden,
            "N_column_cm2":  N_col,
            "peak_alt_km":   alt[peak_idx] / 1e3,
            "peak_r_um":     p[f"sulfate_{dominant}_r_um"][peak_idx],
            "dominant_mode": dominant,
        }

    def save_column_profiles(self, lat: float, lon: float,
                              output_path: str, time_index: int = 0) -> None:
        """
        Extract a column and save to a small self-contained NetCDF.

        Parameters
        ----------
        lat, lon     : column coordinates [degrees]
        output_path  : output NetCDF path
        time_index   : time slice index
        """
        import os
        p = self.get_column_profiles(lat, lon, time_index)

        arrays  = {k: v for k, v in p.items()
                   if not np.isscalar(v) and k != "altitudes_m"}
        scalars = {k: v for k, v in p.items() if np.isscalar(v)}

        ds = xr.Dataset(
            {k: ("altitude_m", v) for k, v in arrays.items()},
            coords={"altitude_m": p["altitudes_m"]},
            attrs={
                "latitude":    float(lat),
                "longitude":   float(lon),
                "time_index":  int(time_index),
                "sigma_a1":    scalars.get("sulfate_a1_sigma", 1.8),
                "sigma_a3":    scalars.get("sulfate_a3_sigma", 1.2),
                "description": "WACCM column profiles from cesm-hawc-ali",
            }
        )
        ds.to_netcdf(output_path)
        size_kb = os.path.getsize(output_path) / 1e3
        print(f"Saved {output_path}  ({size_kb:.0f} KB)")

    def list_variables(self) -> list[str]:
        """Print all chemistry and aerosol variables found in the file."""
        known = ["O3", "NO2", "H2O", "SO2", "HNO3", "CH4", "N2O", "CO",
                 "H2SO4", "OH",
                 "so4_a1", "so4_a2", "so4_a3",
                 "num_a1", "num_a2", "num_a3",
                 "soa1_a1", "soa2_a1", "soa3_a1", "soa4_a1", "soa5_a1"]
        critical = {"so4_a1", "so4_a3", "num_a1", "num_a3", "SO2", "H2O", "O3"}
        present  = [v for v in known if v in self.ds]
        print("Variables found in file:")
        for v in present:
            units = self.ds[v].attrs.get("units", "?")
            lname = self.ds[v].attrs.get("long_name", "")
            flag  = "  ← KEY" if v in critical else ""
            print(f"  {v:15s}  [{units:8s}]  {lname}{flag}")
        missing = critical - set(present)
        if missing:
            print(f"\nMissing: {missing} — add to fincl in user_nl_cam")
        return present
