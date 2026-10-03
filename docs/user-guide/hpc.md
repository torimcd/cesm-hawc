# Running on HPC

Large batch and orbit runs are designed for a single many-core cluster node:
one process per core, with jobs (months, days or orbit files) spread across
them.

## Setting up the environment (Alliance Canada)

On Alliance Canada clusters (Fir, Rorqual, Narval), with `micromamba`
installed in `~/bin`:

```bash
git clone https://github.com/torimcd/cesm-hawc
cd cesm-hawc
bash scripts/setup/create_env.sh
```

This creates the `hawc_env` environment, installs cesm-hawc with the `[sim]`
extra, and checks that `sasktran2`, `hawcsimulator` and `cesm_hawc` import.
It takes 5–10 minutes the first time.

## SLURM templates

The `scripts/` directory has SLURM job scripts to copy and adapt:

| Script | Runs |
|--------|------|
| `scripts/submit.sh` | `cesm-hawc run --mode single` on one core |
| `scripts/submit_orbit_daily.sh` | `cesm-hawc run --mode orbit-track` on a full node |
| `scripts/submit_orbit_daily_l2.sh` | The same, taking the config, case name and an optional `strip-ozone` flag as arguments |

Before submitting, set `--account` to your allocation and make sure
`n_workers` in `config.toml` equals `--cpus-per-task`.

## Practical notes

- **Pin threading to one per process.** Set `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1
  before starting Python. Otherwise each worker may start its own BLAS
  threads and oversubscribe the node.
- **Leave memory headroom.** L2 retrieval memory varies from profile to
  profile. Request memory per core with a margin rather than filling the
  node; `sacct -j <jobid> --format=JobID,MaxRSS,State` shows what a finished
  job actually used.
- **Workers are recycled after each day** when `run_l2 = true`, which limits
  memory growth within a long-lived process.
- **Interrupted runs resume.** Re-submit the same command; finished days are
  skipped and, with `run_l2 = true`, finished profiles within a day are too.
- **Compute nodes without internet are fine.** cesm-hawc turns off astropy's
  automatic Earth-orientation downloads.

% TODO: add measured per-profile runtime and memory figures once benchmarked
% on the current version.
