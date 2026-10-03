# Configuration

The CLI reads a TOML file, `config.toml` by default. Start from
`config.example.toml` in the repository root, which lists every table with
comments. Each table is optional; a mode reports which tables it needs if
they are missing. Paths in `out_dir`, `orbit_dir`, `waccm_data_dir` and the
`orbit_real` directories may start with `~`.

Keys without a default are required whenever their table is present.

## `[single]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `waccm_background` | path | — | WACCM history file for the reference run |
| `waccm_injection` | path | `""` | WACCM history file for the injection run; `""` skips it |
| `time_idx` | int | `0` | Time index within the file |
| `obs_time` | ISO time | — | Observation time, used for solar position |
| `out_dir` | path | — | Output directory |

## `[batch]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `waccm_background_dir` | path | — | Directory of background h0 files |
| `waccm_injection_dir` | path | `""` | Directory of injection h0 files; `""` skips them |
| `h0_pattern` | glob | `"*.cam.h0.*.nc"` | File pattern in both directories |
| `month_filter` | list of `"YYYY-MM"` | `[]` | Months to run; `[]` runs all |
| `out_dir` | path | — | Output directory |
| `n_workers` | int | `1` | Worker processes |

## `[geometry]`

Used by `single` and `batch`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tangent_lat` | degrees | — | Tangent-point latitude |
| `tangent_lon` | degrees | — | Tangent-point longitude (−180–180 or 0–360) |
| `sza_deg` | degrees | `60.0` | Solar zenith angle at the tangent point |
| `saa_deg` | degrees | `0.0` | Solar azimuth angle at the tangent point |

## `[orbit]`

Used by `orbit-track`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `orbit_dir` | path | — | Directory of orbit files |
| `orbit_pattern` | glob | `"orbit_*.nc"` | Orbit file pattern |
| `orbit_epoch` | date | `"2019-08-01"` | Time origin of the orbit file set; orbit `time` values are seconds since this date |
| `center_pixel` | int | `256` | Across-track pixel used as the tangent point |
| `waccm_data_dir` | path | — | CESM archive root containing `<case_name>/atm/hist/` |
| `case_name` | string | — | The CESM case this run processes |
| `h2_pattern` | glob | `"*.cam.h2.*.nc"` | h2 file pattern |
| `h2_cadence` | `"daily"` or `"subdaily"` | `"daily"` | Use `"subdaily"` when there is more than one h2 file per date |
| `obs_cadence_s` | seconds | `60.0` | Spacing between sampled observations |
| `run_start_date`, `run_end_date` | `"YYYY-MM-DD"` | `""` | Optional date range; `""` means unbounded |
| `run_l2` | bool | `false` | Also run the L2 retrieval for every observation |
| `strip_ozone` | bool | `false` | Zero WACCM ozone before simulating (diagnostic) |
| `out_dir` | path | — | Output directory |
| `n_workers` | int | `1` | Worker processes (one job per day) |

## `[orbit_real]`

Used by `orbit-file`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `orbit_dir` | path | — | Directory of orbit files |
| `orbit_pattern` | glob | `"orbit_*.nc"` | Orbit file pattern |
| `waccm_background_dir` | path | — | Directory of background h2 files |
| `waccm_injection_dir` | path | `""` | Directory of injection h2 files; `""` skips them |
| `h2_pattern` | glob | `"*.cam.h2.*.nc"` | h2 file pattern |
| `across_indices` | list of int | `[]` | Across-track pixels to simulate; `[]` means all |
| `time_stride` | int | `1` | Simulate every *N*-th orbit time step |
| `out_dir` | path | — | Output directory |
| `n_workers` | int | `1` | Worker processes (one job per orbit file) |

## `[instrument]`

Used by every mode.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `wavelengths_nm` | list of float | `[470.0, 745.0, 1020.0]` | Simulated wavelengths |
| `alt_grid_start_m` | m | `0.0` | Altitude grid start |
| `alt_grid_stop_m` | m | `65000.0` | Altitude grid end (inclusive) |
| `alt_grid_step_m` | m | `1000.0` | Altitude grid spacing |

The instrument noise model is fixed (see [Methods](../background/methods.md#instrument-noise))
and is not configurable.
