# cesm-hawc

Feed CESM2/WACCM atmosphere model output into the
[HAWC ALI simulator](https://github.com/usask-arg/hawc-simulator) to
simulate what the HAWCSat Aerosol Limb Imager would observe.

## What this does

Given CESM/WACCM history output (monthly `h0` or daily `h2` files), this
package:

1. Extracts one or many atmospheric columns (T, P, O₃, MAM4 sulfate aerosol)
2. Converts MAM4 modal aerosol to extinction profiles using the ALI Mie database
3. Runs the `IdealALISimulator` forward model + L2 retrieval
4. Outputs retrieved aerosol extinction and median radius profiles

It has two tiers:

- **Base** (`pip install cesm-hawc`, Python >=3.10) — WACCM column
  extraction and saving simulator/L2 inputs at monthly or daily scale.
  Only needs numpy/xarray/scipy/pandas.
- **`[sim]` extra** (`pip install cesm-hawc[sim]`, Python >=3.11) — the
  full forward model and L2 retrieval, via `hawcsimulator` + `sasktran2`.

## Install

```bash
pip install cesm-hawc          # base tier
pip install cesm-hawc[sim]     # + full simulator (requires Python >=3.11)
```

**From source (development):**

```bash
git clone https://github.com/torimcd/cesm-hawc
cd cesm-hawc
micromamba env create -f environment.yml
micromamba activate hawc_env
pip install -e ".[sim,dev]"
```

**Alliance Canada HPC (Fir/Rorqual/Narval):**

```bash
git clone https://github.com/torimcd/cesm-hawc
cd cesm-hawc
bash scripts/setup/create_env.sh
```

That script creates the `hawc_env` micromamba environment, installs all
dependencies, and registers the package. It takes 5–10 minutes on first run.

## Quick start

### Python API

```python
from cesm_hawc.waccm import WACCMAtmosphere
from cesm_hawc.constituents import build_waccm_constituents
from cesm_hawc.simulation import run_ali_simulation

# Point at your WACCM h0 or h2 file
result = run_ali_simulation(
    background_file = "path/to/background.cam.h0.nc",
    injection_file  = "path/to/injection.cam.h0.nc",
    lat=30.6, lon=180.0,
    time_index=0,
)

print(result["peak_extinction_anomaly_m"])   # m⁻¹
print(result["peak_radius_anomaly_nm"])       # nm
print(result["delta_burden_mg_m2"])           # mg SO₄ m⁻²
```

### CLI

Copy `config.example.toml` to `config.toml` and fill in your paths, then:

```bash
# Base tier: extract and save WACCM column inputs, no [sim] extra needed
cesm-hawc save-inputs --config config.toml --mode single

# [sim] tier: run the full forward model + L2 retrieval end to end
cesm-hawc run --config config.toml --mode single
```

`--mode` selects the scale:

| Mode | Scale | Config section |
|------|-------|-----------------|
| `single` | one column, one file | `[single]` + `[geometry]` |
| `batch` | a directory of monthly h0 files | `[batch]` + `[geometry]` |
| `orbit-track` | a real orbit ground track matched to one CESM case's daily h2 files | `[orbit]` |
| `orbit-file` | real per-orbit-file, per-pixel observations matched to daily h2 files | `[orbit_real]` |

Add `--dry-run` to see the job count without running anything, `--n-workers N`
to override the config's worker count, and `--out-dir PATH` to override the
output directory. Run `cesm-hawc save-inputs --help` / `cesm-hawc run --help`
for the full flag list.

Library users calling into `cesm_hawc.orbit_files`, `cesm_hawc.calibration`,
or `cesm_hawc.simulation` directly (rather than through the CLI) should call
`cesm_hawc.configure_environment()` once at startup — it disables astropy's
IERS auto-download, silences noisy third-party logging, and patches a known
`hawcsimulator` calibration-cache race condition. The CLI calls this
automatically.

## Consuming saved inputs externally

`save-inputs` doesn't just save the raw WACCM profile (T, P, humidity, gas
VMRs, per-mode sulfate number density/radius) — when `sasktran2` is
importable at save time (i.e. you're in the `[sim]`-capable environment),
it *also* saves everything needed to reconstruct the simulator's aerosol/gas
constituent objects: per-mode extinction (at the 745 nm reference
wavelength `ExtinctionScatterer` uses, plus a multi-wavelength "truth"
array) and clipped median radius, alongside the Mie database's build
parameters (`mie_refractive_index`, `mie_wavelength_grid_nm`,
`mie_median_radius_grid_nm`, `mode_width_accum`, `mode_width_coarse`) as
file attrs. Check a file's `includes_constituents` attr to see which shape
it has (`--profiles-only` skips this even when `sasktran2` is available,
for minimal-footprint massive batch runs).

`sasktran2`'s constituent objects themselves can't be serialized to a file since they wrap live Mie-database state built from real scattering
calculations, so this is the closest a file format
can get: everything **except** the Mie database build itself is
precomputed and saved. You can go straight from a saved file to a
simulator run using only *native* `sasktran2` calls with this output, no `cesm_hawc` import required at all:

```python
import xarray as xr
import sasktran2 as sk
from hawcsimulator.ali.configurations.ideal_spectrograph import IdealALISimulator

ds = xr.open_dataset("background_column.nc")
assert ds.attrs["includes_constituents"], "file was saved with --profiles-only"
alt_m = ds["altitude_m"].values

def mode_constituent(mode: str) -> "sk.constituent.ExtinctionScatterer":
    mode_width = ds.attrs[f"mode_width_{'accum' if mode == 'aerosol_accum' else 'coarse'}"]
    mode_db = sk.database.MieDatabase(
        sk.mie.distribution.LogNormalDistribution().freeze(mode_width=mode_width),
        sk.mie.refractive.H2SO4(),          # ds.attrs["mie_refractive_index"]
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
    "o3":  sk.constituent.VMRAltitudeAbsorber(sk.optical.O3DBM(), altitudes_m=alt_m, vmr=ds["vmr_o3"].values),
    "no2": sk.constituent.VMRAltitudeAbsorber(sk.optical.NO2Vandaele(), altitudes_m=alt_m, vmr=ds["vmr_no2"].values),
    "aerosol_accum":  mode_constituent("aerosol_accum"),
    "aerosol_coarse": mode_constituent("aerosol_coarse"),
}

sim_input = {
    "tangent_latitude": ds.attrs["latitude"],
    "tangent_longitude": ds.attrs["longitude"],
    "altitude_grid": alt_m,
    "polarization_states": ["I", "dolp"],
    "sample_wavelengths": [470.0, 745.0, 1020.0],
    "time": "2035-02-01T12:00:00Z",   # your own observation time
    "constituents": constituents,
}
data = IdealALISimulator().run(["l2", "front_end_radiance", "l1b"], sim_input)
```


## Required WACCM output variables

Add these to `fincl` in `user_nl_cam` if not already present:

| Variable | Description |
|----------|-------------|
| `T`, `Q`, `PS` | Temperature, humidity, surface pressure |
| `O3`, `NO2`, `SO2` | Gas chemistry (mol/mol) |
| `so4_a1`, `so4_a3` | Sulfate mass mixing ratio (kg/kg) |
| `num_a1`, `num_a3` | Aerosol number mixing ratio (#/kg) |

See [docs/waccm_variables.md](docs/waccm_variables.md) for the full list.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests that need `sasktran2`/`hawcsimulator` are automatically skipped if
those aren't installed. A small bundled example column
(`src/cesm_hawc/data/example_column_*.nc`) is used for fixture-based tests
and doesn't require any external data.
