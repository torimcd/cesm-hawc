# Configuration

The CLI reads a TOML file, `config.toml` by default. Start from
`config.example.toml` in the repository root, which lists every table with
comments.

- `[case]` is required.
- `[fixed]` and `[orbit]` are needed only by their own mode.
- `[instrument]` is optional; its defaults are listed below.

Paths may start with `~`. Keys without a default are required whenever their
table is present.

## `[case]`

The model case to process, where to write, and options shared by both modes.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | — | Free-form label for the case; output goes to `out_dir/<name>/` |
| `waccm_dir` | path | — | Directory of CAM history files. `{name}` is replaced by the case name |
| `pattern` | glob | — | Which files to read, e.g. `"*.cam.h0.*.nc"` or `"*.cam.h2.*.nc"` |
| `out_dir` | path | — | Output root |
| `n_workers` | int | `1` | Worker processes |
| `time_index` | int | `0` | Time slice to read within each history file |
| `start_date`, `end_date` | `"YYYY-MM-DD"` | `""` | Optional date range, using the date in each file name; `""` means unbounded |
| `run_l2` | bool | `false` | Also run the L2 retrieval (`run` only). Takes minutes per profile |
| `strip_ozone` | bool | `false` | Zero WACCM ozone before simulating (`run` only); output goes to `<name>_no_ozone/` |

The `{name}` placeholder lets one config serve several cases stored side by
side. With

```toml
waccm_dir = "/archive/{name}/atm/hist/"
```

`cesm-hawc run --mode orbit --case-name another_case` reads
`/archive/another_case/atm/hist/` and writes to `out_dir/another_case/`.

```{note}
`time_index` applies to every file. If your history files hold more than one
time sample each (`mfilt` > 1), only that one sample is used.
```

## `[fixed]`

Used by `--mode fixed`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tangent_lat` | degrees | — | Tangent-point latitude |
| `tangent_lon` | degrees | — | Tangent-point longitude (−180–180 or 0–360) |
| `sza_deg` | degrees | `60.0` | Solar zenith angle at the tangent point |
| `saa_deg` | degrees | `0.0` | Solar azimuth angle at the tangent point |
| `obs_time` | ISO time | from file name | Observation time for every file (see [Run modes](run-modes.md#fixed)) |

## `[orbit]`

Used by `--mode orbit`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `orbit_dir` | path | — | Directory of orbit files |
| `orbit_pattern` | glob | `"orbit_*.nc"` | Orbit file pattern |
| `orbit_epoch` | date | `"2019-08-01"` | Time origin of the orbit file set; orbit `time` values are seconds since this date |
| `center_pixel` | int | `256` | Across-track pixel used as the tangent point |
| `obs_cadence_s` | seconds | `60.0` | Spacing between sampled observations |
| `h2_cadence` | `"daily"` or `"subdaily"` | `"daily"` | Use `"subdaily"` when there is more than one history file per date |

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
