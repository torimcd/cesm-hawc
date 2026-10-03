# Quickstart

This walks through one column, one WACCM file, end to end.

## 1. Write a config file

Copy the template from the repository root and edit the paths:

```bash
cp config.example.toml config.toml
```

For a single column you need the `[single]`, `[geometry]` and `[instrument]`
tables:

```toml
[single]
waccm_file = "/path/to/casename.cam.h0.2035-02.nc"
time_idx         = 0
obs_time         = "2035-02-01T12:00:00Z"
out_dir          = "~/results/hawc_ali/"

[geometry]
tangent_lat = 30.6
tangent_lon = 180.0
sza_deg     = 60.0
saa_deg     = 0.0

[instrument]
wavelengths_nm = [470.0, 745.0, 1020.0]
```

See [Configuration](user-guide/configuration.md) for every key.

## 2. Save the inputs (base tier)

```bash
cesm-hawc save-inputs --config config.toml --mode single
```

This writes `background_column.nc` (and `injection_column.nc` if an
injection file is set) to `out_dir`. Each file holds the extracted WACCM
profile and, if `sasktran2` is installed, everything needed to rebuild the
simulator's constituents. See [Using saved inputs](user-guide/saved-inputs.md).

## 3. Run the simulator (`[sim]` tier)

```bash
cesm-hawc run --config config.toml --mode single
```

This runs the forward model and L2 retrieval for the model output column and writes the L2 output, CESM's own extinction for
comparison, and a `summary.txt`. See [Outputs](user-guide/outputs.md).

Add `--dry-run` to any command to report what would be done without running
anything.

## The same thing from Python

```python
import cesm_hawc
from cesm_hawc.noise import default_noise_model
from cesm_hawc.simulation import run_ali_simulation

cesm_hawc.configure_environment()   # once per process; the CLI does this for you

result = run_ali_simulation(
    waccm_file="path/to/casename.cam.h0.2035-02.nc",

    lat=30.6, lon=180.0,
    time_index=0,
    noise_model=default_noise_model(),
)

print(result["peak_extinction_anomaly_m"])   # m⁻¹, above 15 km
print(result["peak_radius_anomaly_nm"])      # nm, above 15 km
print(result["delta_burden_mg_m2"])          # mg SO₄ m⁻², 15–35 km
```

`noise_model` is required: the ALI imager model has no noiseless mode.

## A fuller worked example

`examples/quickstart.py` in the repository runs one CESM h2 file against a
real orbit file, saves every intermediate product, and compares the
retrieval against the model truth.

% TODO: update once examples/quickstart.py is switched to ideal_dolp_imager
% and passes a noise model.
