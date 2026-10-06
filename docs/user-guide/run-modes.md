# Run modes

The `cesm-hawc` command has two subcommands, each with two modes:

```bash
cesm-hawc save-inputs --config config.toml --mode {fixed,orbit}
cesm-hawc run         --config config.toml --mode {fixed,orbit}
```

- **`save-inputs`** extracts WACCM columns and saves them as simulator-ready
  NetCDF files. It needs only the base install.
- **`run`** runs the ALI forward model and, if `run_l2 = true`, the L2
  retrieval. It needs the `[sim]` extra.

Each run processes **one model case**: the history files that `[case]`
points to. Output goes under `out_dir/<case name>/`. To process several
cases, run once per case. `--case-name` lets one config serve them all (see
[Configuration](configuration.md#case)).

## Choosing a mode

| Mode | Samples | One job per | Config table |
|------|---------|-------------|--------------|
| `fixed` | One column at a fixed tangent point and solar geometry | History file | `[fixed]` |
| `orbit` | Many columns along a real HAWC orbit ground track | Day (or history file, if sub-daily) | `[orbit]` |

Both modes also read `[case]` and `[instrument]`.

### fixed

For every history file matched by `[case]`, simulates the column at
`tangent_lat`, `tangent_lon` with the given solar zenith and azimuth angles.
It works with any output stream, for example monthly h0 files for a
seasonal cycle at one location, or a single file for a quick look.

The observation time is `obs_time` if set, otherwise it comes from each file
name: `YYYY-MM` files use the 15th at 12:00 UTC, and `YYYY-MM-DD-SSSSS`
files use that date and time of day.

### orbit

Replays a real HAWC orbit ground track over the case's daily (or sub-daily)
output. For each model date it takes one orbit day's files, samples an
observation at `center_pixel` every `obs_cadence_s` seconds, and moves each
observation onto the model date while keeping its real time of day and
satellite geometry. Solar angles are computed from that time and position,
and night-side observations are skipped.

Model dates are paired with orbit days by position: the *n*-th date (sorted)
uses orbit day *n* modulo the number of orbit days. See
[Methods](../background/methods.md#orbit-sampling) for details.

Set `h2_cadence = "subdaily"` if your history files are written more than
once per day. Each observation is then assigned to the nearest snapshot in
time instead of to its calendar date's file.

## Command-line options

| Option | Effect |
|--------|--------|
| `--config PATH` | Config file to read (default `config.toml`) |
| `--mode MODE` | `fixed` or `orbit` (required) |
| `--case-name NAME` | Override `[case] name`, which also fills `{name}` in `waccm_dir` |
| `--out-dir PATH` | Override `[case] out_dir` |
| `--n-workers N` | Override `[case] n_workers` |
| `--dry-run` | Report how many jobs would run, then stop |
| `--profiles-only` | `save-inputs` only: save the raw WACCM profiles, skipping constituents |
| `--strip-ozone` | `run` only: zero WACCM ozone before simulating; writes to `<case name>_no_ozone/` |
| `-v`, `--verbose` | Debug-level logging (before the subcommand: `cesm-hawc -v run …`) |

## Parallelism and resuming

Jobs run across `n_workers` processes; `n_workers = 1` runs serially, which is
easiest to debug.

`run` is resumable in both modes. A job whose outputs already exist is
skipped, so re-submitting the same command continues where a killed run
stopped. In `orbit` mode with `run_l2 = true`, each finished profile within a
day is also recorded in `l2_diagnostics.csv` as it completes, so a partly
finished day resumes too.
