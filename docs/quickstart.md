# Quickstart

This walks through one column from one WACCM history file, end to end.

## 1. Write a config file

Copy the template from the repository root and edit the paths:

```bash
cp config.example.toml config.toml
```

For a fixed column you need the `[case]` and `[fixed]` tables (`[instrument]`
is optional):

```toml
[case]
name      = "my_case"
waccm_dir = "/path/to/archive/my_case/atm/hist/"
pattern   = "my_case.cam.h0.2035-02.nc"     # one file; use a wildcard for many
out_dir   = "~/results/cesm_hawc/"
run_l2    = true

[fixed]
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
cesm-hawc save-inputs --config config.toml --mode fixed
```

This writes `my_case.cam.h0.2035-02.nc` to `out_dir/my_case/`. It holds the
extracted WACCM profile and, if `sasktran2` is installed, everything needed
to rebuild the simulator's constituents. See
[Using saved inputs](user-guide/saved-inputs.md).

## 3. Run the simulator (`[sim]` tier)

```bash
cesm-hawc run --config config.toml --mode fixed
```

This runs the forward model and, because `run_l2 = true`, the L2 retrieval.
It writes the L1b output with the model's truth extinction, the L2 output,
CESM's own extinction for comparison, and a `summary.txt` to
`out_dir/my_case/my_case.cam.h0.2035-02/`. See
[Outputs](user-guide/outputs.md).

Add `--dry-run` to any command to report what would be done without running
anything.

## The same thing from Python

```python
import cesm_hawc
from cesm_hawc.noise import default_noise_model
from cesm_hawc.simulation import run_ali_simulation

cesm_hawc.configure_environment()   # once per process; the CLI does this for you

result = run_ali_simulation(
    "path/to/my_case.cam.h0.2035-02.nc",
    lat=30.6, lon=180.0,
    time_index=0,
    run_l2=True,
    noise_model=default_noise_model(),
)

l2 = result["data"]["l2"]
print(l2["stratospheric_aerosol_extinction_per_m"])   # m⁻¹
print(l2["stratospheric_aerosol_median_radius"])      # nm
print(result["burden"]["burden_mg_m2"])               # mg SO₄ m⁻², 15–35 km
```

`noise_model` is required: the ALI imager model has no noiseless mode.

For every step on one example column, with plots, see the
[Tutorial](tutorial.ipynb).
