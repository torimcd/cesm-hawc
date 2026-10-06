# Outputs

Everything is written under `out_dir/<case name>/` (`out_dir/<case
name>_no_ozone/` with `strip_ozone`). Below, `<file>` is a history file's
name without `.nc`, and `<date>` is `YYYY-MM-DD` (or `YYYY-MM-DD-SSSSS` for
sub-daily output).

## `save-inputs`

| Mode | Files |
|------|-------|
| `fixed` | `<file>.nc` |
| `orbit` | `<date>/column_<HHMMSS>.nc`, one per observation, with the observation time and satellite position as attributes |

Every file is a single column on the `altitude_m` grid. The variables are
described in [Using saved inputs](saved-inputs.md).

## `run --mode fixed`

Per history file, under `<file>/`:

| File | Contents |
|------|----------|
| `l1b.nc` | L1b radiance, DoLP and their noise (`wavelength` × `altitude_m`), plus per-mode truth extinction on the instrument grid and on the model grid (`*_atm`, dim `atm_altitude_m`) |
| `l2.nc` | L2 retrieval output (`run_l2 = true` only) |
| `cesm_extinction.nc` | CESM's own aerosol extinction (`EXTINCTdn` etc.) on the same grid, if present in the file |
| `summary.txt` | File, time, location and 15–35 km sulfate burden |

## `run --mode orbit`

Per job, under `<date>/`:

| File | Contents |
|------|----------|
| `curtain.nc` | The `l1b.nc` contents above for every observation, along a new `along_track` dimension with `lat`, `lon` and `time` coordinates |
| `orbit_track.csv` | Time, latitude and longitude of every successful observation |
| `l2_retrieval.nc` | L2 output concatenated along the track (`run_l2 = true` only) |
| `l2_profiles/<case>_<time>.nc` | One L2 file per observation, written as each finishes (`run_l2 = true` only) |
| `l2_diagnostics.csv` | Per-observation convergence, function evaluations, cost, runtime and errors (`run_l2 = true` only) |

An `.orbit_day_index_cache.json` file in `out_dir` caches the orbit-file
index between runs and can be deleted safely.

## L2 variables

The L2 datasets come from the retrieval in `hawcsimulator`/`aliprocessing`.
The main ones are:

| Variable | Units | Description |
|----------|-------|-------------|
| `stratospheric_aerosol_extinction_per_m` | m⁻¹ | Retrieved extinction at 745 nm |
| `stratospheric_aerosol_median_radius` | nm | Retrieved lognormal median radius |

% TODO: list the remaining L2 variables and their units.
