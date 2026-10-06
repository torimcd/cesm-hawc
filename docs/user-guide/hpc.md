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

## SLURM job script

`scripts/submit.sh` runs `cesm-hawc run --mode orbit` on one node. Before
using it, set `--account` to your allocation, and create a `logs/` directory
in the repository root; SLURM writes the job's output there and fails
without it.

```bash
mkdir -p logs
sbatch scripts/submit.sh                          # config.toml, configured case
sbatch scripts/submit.sh config.toml case_a       # a different case
sbatch scripts/submit.sh config.toml case_a strip-ozone
```

The arguments are, in order: the config file (default `config.toml`), a case
name passed to `--case-name`, and the literal word `strip-ozone` to add
`--strip-ozone`. Queuing one job per case name lets a single config drive
several cases.

The script asks for 32 cores with 12 GB each for 60 hours, and sets the
worker count to the number of cores, overriding `n_workers` in the config.
Change the resources at submit time rather than editing the script:

```bash
sbatch --cpus-per-task=16 --mem-per-cpu=8G --time=12:00:00 scripts/submit.sh
```

For `--mode fixed`, copy the script and change `--mode orbit` to
`--mode fixed`. A handful of files usually needs only one core and well under
an hour.

## Practical notes

- **Pin threading to one per process.** Set `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1
  before starting Python (`submit.sh` does this). Otherwise each worker may
  start its own BLAS threads and oversubscribe the node.
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
