#!/bin/bash
#SBATCH --account=rrg-czg
#SBATCH --job-name=orbit_daily_l2
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/orbit_daily_l2_%j.out
#SBATCH --error=logs/orbit_daily_l2_%j.err

# IMPORTANT: config.toml's n_workers must match --cpus-per-task above (90).
# If you change one, change the other — a mismatch either leaves allocated
# cores idle (n_workers too low) or oversubscribes them (n_workers too high).
#
# WORKER COUNT (90, not 182): Fir's standard CPU nodes have 192 cores but
# only 750G total memory. At the proven 8G/core (unchanged from the
# original 8-worker config -- per-worker memory footprint was never
# directly measured, so this keeps the known-safe ratio rather than
# guessing it down to fit 182 workers), 90 workers x 8G = 720G leaves
# headroom under 750G for the main orchestrating process + OS. Requesting
# enough memory for 182 workers at 8G/core (1,456G) would force scheduling
# onto one of only 8 large-memory nodes cluster-wide instead of the 864
# standard nodes, trading a small further walltime reduction for a much
# longer queue wait.
#
# WALLTIME (30h): the L2 benchmark (l2_benchmark_results.csv, n=48) gives
# ~2449 CPU-hours total for the full 6-month/4-case production run, so
# ~27h estimated at 90 workers. 30h leaves margin for slower-than-observed
# days -- the benchmark sample is modest (48 profiles across 3 dates) and
# extreme-value behavior across the true 182 production days could exceed
# what that sample happened to catch. Still nowhere near Fir's 168h cap.
#
# WANT IT FASTER LATER? After this run (or a short test run), check actual
# peak memory per worker via `seff $SLURM_JOB_ID` or `sacct -j $SLURM_JOB_ID
# --format=JobID,MaxRSS,NNodes`. If real usage is well under 8G/core, you
# can lower --mem-per-cpu and raise --cpus-per-task/n_workers together (up
# to 182, Fir's per-node core count) for a further walltime reduction --
# see run_orbit_daily.py's day-level/profile-level resume, which makes
# this safe to experiment with incrementally rather than committing to a
# larger worker count on the very first full run.

set -euo pipefail

# Pin BLAS/OpenMP threading to 1 per process. Without this, numpy/scipy/
# sasktran2's underlying BLAS library may spawn its own internal threads
# per worker (e.g. 4 each), which at 90 independent worker PROCESSES could
# oversubscribe the 90 allocated cores several times over and collapse
# performance via thread-thrashing rather than actually running 90-way
# parallel. Must be set before Python starts.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

eval "$(micromamba shell hook --shell bash)"
micromamba activate hawc_env
unset PYTHONPATH

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Workers requested: 90 (cpus-per-task) -- confirm config.toml's n_workers matches"

cd /project/6079534/vmcd/cesm-hawc/scripts
python run_orbit_daily.py

echo "Job finished: $(date)"