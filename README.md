# cesm-hawc

Feed CESM2/WACCM stratospheric aerosol injection (SAI) output into the
[HAWC ALI simulator](https://github.com/usask-arg/hawc-simulator) to
simulate what the HAWCSat Aerosol Limb Imager would observe.

## What this does

Given a WACCM h0 history file, this package:

1. Extracts a single atmospheric column (T, P, O₃, MAM4 sulfate aerosol)
2. Converts MAM4 modal aerosol to extinction profiles using the ALI Mie database
3. Runs the `IdealALISimulator` forward model + L2 retrieval
4. Outputs retrieved aerosol extinction and median radius profiles

## Quick start

```bash
# Install environment (Alliance HPC / Linux)
micromamba create -n hawc_env -c conda-forge python=3.11 sasktran2 xarray scipy matplotlib pandas -y
micromamba activate hawc_env
pip install hawcsimulator
pip install -e .
```

```python
from cesm_hawc.waccm import WACCMAtmosphere
from cesm_hawc.constituents import build_waccm_constituents
from cesm_hawc.simulation import run_ali_simulation

# Point at your WACCM h0 file
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

## Installation

**Local (Linux/Mac):**
```bash
git clone https://github.com/torimcd/cesm-hawc
cd cesm-hawc
micromamba env create -f environment.yml
micromamba activate hawc_env
pip install -e .
```

**Alliance Canada HPC (Fir/Rorqual/Narval):**

```bash
git clone https://github.com/torimcd/cesm-hawc
cd cesm-hawc
bash scripts/setup/create_env.sh
```

That script creates the `hawc_env` micromamba environment, installs all
dependencies, and registers the package. It takes 5–10 minutes on first run.


## Required WACCM output variables

Add these to `fincl` in `user_nl_cam` if not already present:

| Variable | Description |
|----------|-------------|
| `T`, `Q`, `PS` | Temperature, humidity, surface pressure |
| `O3`, `NO2`, `SO2` | Gas chemistry (mol/mol) |
| `so4_a1`, `so4_a3` | Sulfate mass mixing ratio (kg/kg) |
| `num_a1`, `num_a3` | Aerosol number mixing ratio (#/kg) |

See [docs/waccm_variables.md](docs/waccm_variables.md) for the full list.

## Physics notes

- MAM4 coarse mode σ_g = 1.2 (WACCM-specific; Mills et al. 2016)
- The ALI Mie database (`aerosol_median_radius_db()`) is used for all
  optical calculations, ensuring consistency with the ALI retrieval algorithm
- The `atmosphere_method="default"` Hamilton config is used; `"omps_calipso_era5"`
  requires network access and will crash on compute nodes
