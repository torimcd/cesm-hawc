# Python API

The CLI covers most uses. These are the main entry points for calling
cesm-hawc from your own scripts. Full parameter descriptions are in each
function's docstring (`help(...)` in Python).

```{note}
Call `cesm_hawc.configure_environment()` once at the start of any script
that uses the `[sim]` tier. It turns off astropy's automatic IERS downloads,
quiets the simulator's per-step error logging, and works around a cache race
in `hawcsimulator` when many processes run at once. The CLI calls it for you.
```

## Base tier

### `cesm_hawc.waccm.WACCMAtmosphere`

Reads a CAM history file (or a list of files) and extracts columns.

| Method | Returns |
|--------|---------|
| `WACCMAtmosphere(filepath, alt_grid_km=None, z_surface=0.0, h2o_join_hpa=100.0)` | Opens the file(s) and checks the required variables |
| `.get_column_profiles(lat, lon, time_index=0)` | Dict of profiles on the altitude grid (see [Using saved inputs](../user-guide/saved-inputs.md)) |
| `.sulfate_column_burden(lat, lon, time_index=0, alt_range_km=(15, 35))` | Burden [mg m⁻²], number column, peak altitude and radius, dominant mode |
| `.extract_cesm_extinction(lat, lon, time_index, alt_m)` | CESM's own `EXTINCTdn`/`EXTINCTUVdn`/`EXTINCTNIRdn` on `alt_m` |
| `.save_column_profiles(lat, lon, output_path, time_index=0)` | Writes the profiles to NetCDF |
| `.list_variables()` | Prints and returns the chemistry and aerosol variables present |

### `cesm_hawc.save_inputs.save_column_inputs`

`save_column_inputs(waccm, lat, lon, output_path, time_index, alt_m, wavelengths_nm=None, profiles_only=False, obs_time=None)`

Writes one column as a simulator-ready file, adding constituent data when
`sasktran2` is available.

### `cesm_hawc.config.load_config`

`load_config(path)` reads and validates a config file and returns a
`CesmHawcConfig` with one attribute per table. Raises `ConfigError` for a
missing file or required key.

## Simulation tier (`[sim]`)

### `cesm_hawc.simulation.run_ali_simulation`

`run_ali_simulation(waccm_file, *, lat, lon, time_index=0, sza_deg=60.0, saa_deg=0.0, obs_time, wavelengths_nm=None, alt_grid_m=None, run_l2=False, noise_model)`

Runs one column of a history file through the forward model (and the L2
retrieval when `run_l2=True`) at a fixed tangent point and solar geometry.
Returns a dict with the simulator output (`data`), the truth extinction
(`true_extinction`) and the column's sulfate burden (`burden`).

### `cesm_hawc.simulation.run_ali_simulation_from_profiles`

`run_ali_simulation_from_profiles(profiles, alt_m, sim_geometry, *, simulator=None, products=DEFAULT_PRODUCTS, noise_model, return_extinction=False, truth_wavelengths_nm=None)`

Lower-level version that takes profiles you have already extracted and a
geometry dict, so you can reuse one `WACCMAtmosphere` and one simulator
across many observations.

### `cesm_hawc.constituents.build_waccm_constituents`

`build_waccm_constituents(profiles, alt_m, return_extinction=False, truth_wavelengths_nm=None)`

Builds the `sasktran2` constituents dict (`o3`, `no2`, `aerosol_accum`,
`aerosol_coarse`) to pass to `simulator.run()` under the `constituents` key.
With `return_extinction=True` it also returns the truth extinction.

### `cesm_hawc.noise.default_noise_model`

Returns the `ALINoiseModel` cesm-hawc uses everywhere. Pass it as
`noise_model`.

## Orbit utilities

`cesm_hawc.orbit_files` reads orbit geometry: `load_orbit_files`,
`build_orbit_day_index` and `extract_observations`. See [Orbit files](../inputs/orbit-files.md).

% TODO: replace this hand-written page with generated API docs once the
% docstrings are standardized on numpydoc.
