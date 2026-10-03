# Outputs

All paths below are relative to the mode's `out_dir` (or `--out-dir`).

## `save-inputs`

| Mode | Files |
|------|-------|
| `single` | `background_column.nc`, `injection_column.nc` |
| `batch` | `background/<YYYY-MM>.nc`, `injection/<YYYY-MM>.nc` |
| `orbit-track` | `<case_name>/<YYYY-MM-DD>/column_<HHMMSS>.nc` |
| `orbit-file` | `<YYYY-MM-DD>/<orbit>_t<time>_a<across>_bg.nc` and `…_inj.nc` |

Every file is a single column on the `altitude_m` grid. The variables are
described in [Using saved inputs](saved-inputs.md).

## `run --mode single`

| File | Contents |
|------|----------|
| `l2_background.nc`, `l2_injection.nc` | L2 retrieval output |
| `cesm_extinction_background.nc`, `cesm_extinction_injection.nc` | CESM's own aerosol extinction (`EXTINCTdn` etc.) on the same grid, if present in the file |
| `summary.txt` | Sulfate burden and, with an injection run, anomaly diagnostics |

## `run --mode batch`

Per month:

- `background/<YYYY-MM>/l2_background.nc`, `cesm_extinction_background.nc`
- `injection/<YYYY-MM>/l2_injection.nc`, `cesm_extinction_injection.nc`
- `diff/<YYYY-MM>/summary_diff.txt` with an injection run, otherwise
  `background/<YYYY-MM>/summary.txt`

## `run --mode orbit-track`

Per day, under `<case_name>/<YYYY-MM-DD>/` (or `<case_name>_no_ozone/…`
with `--strip-ozone`):

| File | Contents |
|------|----------|
| `curtain.nc` | L1b radiance, DoLP and their noise along the track (dim `along_track` × `wavelength` × `altitude_m`), plus per-mode truth extinction on the instrument grid and on the model grid (`*_atm`, dim `atm_altitude_m`) |
| `orbit_track.csv` | Time, latitude and longitude of every successful observation |
| `l2_retrieval.nc` | L2 output concatenated along the track (`run_l2 = true` only) |
| `l2_profiles/<case>_<time>.nc` | One L2 file per observation, written as each finishes (`run_l2 = true` only) |
| `l2_diagnostics.csv` | Per-observation convergence, function evaluations, cost, runtime and errors (`run_l2 = true` only) |

With `h2_cadence = "subdaily"`, the day folder is named
`<YYYY-MM-DD>-<seconds>` after the h2 snapshot instead.

An `.orbit_day_index_cache.json` file in `out_dir` caches the orbit-file
index between runs and can be deleted safely.

## `run --mode orbit-file`

Per orbit file, under `<YYYY-MM-DD>/`:

- `<orbit>_l2_bg.nc` — background L2 output with dimension `obs`, and
  coordinates `obs_time`, `across_idx`, `lat`, `lon`
- `<orbit>_l2_inj.nc` — the same for the injection run, if configured

## L2 variables

The L2 datasets come from the retrieval in `hawcsimulator`/`aliprocessing`.
The two variables cesm-hawc uses in its own summaries are:

| Variable | Units | Description |
|----------|-------|-------------|
| `stratospheric_aerosol_extinction_per_m` | m⁻¹ | Retrieved extinction at 745 nm |
| `stratospheric_aerosol_median_radius` | nm | Retrieved lognormal median radius |

% TODO: list the remaining L2 variables and their units.
