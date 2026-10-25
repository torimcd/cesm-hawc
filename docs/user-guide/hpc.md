# Running on HPC

Large runs are designed for a single many-core cluster node: one worker
process per core, with jobs spread across them. A job is one history file in
`fixed` mode, and one day (or one sub-daily history file) in `orbit` mode.

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
Change the resources at submit time rather than editing the script. For
example, a test over a single day of daily output runs as one job on one
worker (see [Runtime and resources](#hpc-runtime)), so a few cores
are enough:

```bash
sbatch --cpus-per-task=8 --time=12:00:00 scripts/submit.sh
```

For `--mode fixed`, copy the script and change `--mode orbit` to
`--mode fixed`. A handful of files usually needs only one core and well under
an hour.

## Practical notes

- **Pin BLAS threading to one per process.** Set `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1
  before starting Python (`submit.sh` does this), so each worker doesn't
  start its own BLAS threads. This does not limit the radiative transfer's
  threads; see [Runtime and resources](#hpc-runtime).
- **Leave memory headroom.** L2 retrieval memory varies from profile to
  profile. Plan for about 10 GB per worker, with a margin;
  `sacct -j <jobid> --format=JobID,MaxRSS,State` shows what a finished job
  actually used.
- **Workers are replaced after each job** when `run_l2 = true`, which limits
  memory growth within a long-lived process.
- **Interrupted runs resume.** Re-submit the same command; finished jobs are
  skipped and, in `orbit` mode with `run_l2 = true`, so are finished profiles
  within a day.
- **Compute nodes without internet are fine.** cesm-hawc turns off astropy's
  automatic Earth-orientation downloads.

(hpc-runtime)=
## Runtime and resources

One day of `orbit` observations with the L2 retrieval at 6 wavelengths
(1,440 observations sampled at one-minute spacing, about half of them in
daylight and retrieved) took ~9.6 hours on one worker:

| Measure | Value |
|---------|-------|
| Wall time per retrieved profile | about 48 s |
| CPU time per retrieved profile | about 3.7 core-minutes |
| Average cores in use by the worker | about 4.5 |
| Peak memory of the worker | about 10 GB |

Runtime grows with the number of wavelengths.

When planning a job, keep in mind:

- **A job runs on one worker.** Work is divided into jobs: one history file
  in `fixed` mode, one day (or one sub-daily history file) in `orbit` mode.
  More workers than jobs adds nothing, so a run over a single day of daily
  output uses one worker however many cores are requested.
- **Each worker uses several cores.** `hawcsimulator` runs the radiative
  transfer with up to 8 threads, which the BLAS settings above don't limit,
  so a worker averages several cores during an L2 run. With one worker per
  core, as `submit.sh` sets up, workers running in parallel share cores.
- **Each worker needs about 10 GB.** Size memory per worker, not per core.

