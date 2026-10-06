# cesm-hawc

Feed CESM2/WACCM atmosphere model output into the
[HAWC ALI simulator](https://github.com/usask-arg/hawc-simulator) to
simulate what the HAWCSat Aerosol Limb Imager would observe.

## What this does

Given CESM/WACCM history output (monthly `h0` or daily `h2` files), this
package:

1. Extracts one or many atmospheric columns (T, P, O₃, MAM4 sulfate aerosol)
2. Converts MAM4 modal aerosol to extinction profiles using mode-matched Mie databases
3. Runs the `IdealALISimulator` forward model and, optionally, the L2 retrieval
4. Outputs retrieved aerosol extinction and median radius profiles

Requires Python >=3.11. It has two tiers:

- **Base** (`pip install cesm-hawc`) — WACCM column extraction and saving
  simulator-ready inputs. Only needs
  numpy/xarray/scipy/pandas.
- **`[sim]` extra** (`pip install cesm-hawc[sim]`) — the full forward model
  and L2 retrieval, via `hawcsimulator` + `sasktran2`.

## Install

```bash
pip install cesm-hawc          # base tier
pip install cesm-hawc[sim]     # + full simulator
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

### CLI

Copy `config.example.toml` to `config.toml` and fill in your paths. Each run
processes one model case, described by the `[case]` table, and writes under
`out_dir/<case name>/`.

```bash
# Base tier: extract and save WACCM column inputs, no [sim] extra needed
cesm-hawc save-inputs --config config.toml --mode fixed

# [sim] tier: run the forward model (+ L2 retrieval if run_l2 = true)
cesm-hawc run --config config.toml --mode fixed
```

`--mode` selects how the model is sampled:

| Mode | Samples | Config table |
|------|---------|--------------|
| `fixed` | one column at a fixed tangent point and solar geometry, per history file | `[fixed]` |
| `orbit` | columns along a real HAWC orbit ground track, moved onto the case's dates | `[orbit]` |

Add `--dry-run` to see the job count without running anything, and
`--case-name NAME` to run a different case from the same config. Run
`cesm-hawc save-inputs --help` / `cesm-hawc run --help` for the full flag
list.

### Python API

```python
import cesm_hawc
from cesm_hawc.noise import default_noise_model
from cesm_hawc.simulation import run_ali_simulation

cesm_hawc.configure_environment()   # once per process; the CLI does this for you

result = run_ali_simulation(
    "path/to/my_case.cam.h0.2035-02.nc",
    lat=30.6, lon=180.0,
    run_l2=True,
    noise_model=default_noise_model(),
)
l2 = result["data"]["l2"]   # retrieved extinction, median radius, ...
```

Library users calling into `cesm_hawc.orbit_files`, `cesm_hawc.calibration`,
or `cesm_hawc.simulation` directly (rather than through the CLI) should call
`cesm_hawc.configure_environment()` once at startup — it disables astropy's
IERS auto-download, silences noisy third-party logging, and patches a known
`hawcsimulator` calibration-cache race condition.

## Using saved inputs without cesm-hawc

When `sasktran2` is installed, each file written by `save-inputs` contains
everything needed to rebuild the simulator's aerosol and gas constituents
with native `sasktran2` calls: per-mode reference and truth extinction,
clipped median radius, and the Mie database build parameters as attributes.
The documentation's "Using saved inputs" page has a complete example.

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
(`src/cesm_hawc/data/example_column.nc`) is used for fixture-based tests
and doesn't require any external data.
