# Run modes

The `cesm-hawc` command has two subcommands, each with the same four modes:

```bash
cesm-hawc save-inputs --config config.toml --mode {single,batch,orbit-track,orbit-file}
cesm-hawc run         --config config.toml --mode {single,batch,orbit-track,orbit-file}
```

- **`save-inputs`** extracts WACCM columns and saves them as simulator-ready
  NetCDF files. It needs only the base install.
- **`run`** runs the forward model and, depending on the mode, the L2
  retrieval. It needs the `[sim]` extra.

## Choosing a mode

| Mode | Samples | WACCM files | Config tables |
|------|---------|-------------|---------------|
| `single` | One column at a fixed location | One file (any stream) | `[single]`, `[geometry]` |
| `batch` | One fixed column per monthly file | A directory of h0 files | `[batch]`, `[geometry]` |
| `orbit-track` | Many columns along a real orbit ground track, one CESM case per run | Daily (or sub-daily) h2 files | `[orbit]` |
| `orbit-file` | Every chosen pixel of every orbit file | Daily h2 files, background and optional injection | `[orbit_real]` |

All modes also read `[instrument]` for wavelengths and the altitude grid.

### single

Runs one background column, and optionally the matching injection column,
at the fixed tangent point and solar angles in `[geometry]`. With an
injection file, it also reports the peak extinction and radius anomalies
above 15 km and the change in 15–35 km sulfate burden.

### batch

Runs the `single` calculation once per month for a directory of h0 files.
Months are matched between the background and injection directories by the
`YYYY-MM` in the file name. The observation time is set to the 15th of each
month at 12:00 UTC.

### orbit-track

Replays a real HAWC orbit ground track over one CESM case's h2 output. For
each simulated date it takes one orbit day's files, samples observations at
the `center_pixel` every `obs_cadence_s` seconds, and moves each
observation's time onto the simulated date while keeping its real time of
day and satellite geometry. Solar angles are then computed from that time
and position. Night-side observations are skipped.

Simulated dates are paired with orbit days by position: the *n*-th h2 date
in the case (sorted) uses orbit day *n* modulo the number of orbit days. See
[Methods](../background/methods.md#orbit-sampling) for details.

Run this mode once per CESM case; `--case-name` overrides `case_name` so one
config can drive several cases. With `run_l2 = true` it also runs the L2
retrieval for every observation, which is slow (minutes per profile).

Set `h2_cadence = "subdaily"` if your h2 files are written more than once per
day. Each observation is then assigned to the nearest h2 snapshot in time
instead of to its calendar date's file.

### orbit-file

Runs the forward model and L2 retrieval for each selected across-track pixel
(`across_indices`) at every `time_stride`-th time step of each orbit file,
using the h2 file for that orbit file's calendar date. Unlike `orbit-track`,
dates are not remapped: orbit files and h2 files are matched by actual date.

## Common flags

| Flag | Effect |
|------|--------|
| `--config PATH` | Config file to read (default `config.toml`) |
| `--mode MODE` | One of the four modes above (required) |
| `--out-dir PATH` | Override the mode's `out_dir` |
| `--n-workers N` | Override the mode's `n_workers` |
| `--case-name NAME` | Override `[orbit] case_name` (`orbit-track` only) |
| `--strip-ozone` | Zero WACCM ozone before simulating; writes to `<case_name>_no_ozone/` (`run --mode orbit-track` only) |
| `--profiles-only` | Save only the raw WACCM profiles, skipping constituents (`save-inputs` only) |
| `--dry-run` | Report how many jobs would run, then stop |
| `-v`, `--verbose` | Debug-level logging |

## Parallelism and resuming

`batch`, `orbit-track` and `orbit-file` split work into jobs (a month, a day
or an orbit file) and run them across `n_workers` processes; `n_workers = 1`
runs serially, which is easiest to debug.

`run --mode orbit-track` is resumable. A day whose outputs already exist is
skipped, and with `run_l2 = true` each finished profile is recorded in
`l2_diagnostics.csv` as it completes, so a killed job picks up where it
stopped. Re-submit the same command to continue.
