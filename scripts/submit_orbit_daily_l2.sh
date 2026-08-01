#!/bin/bash
#SBATCH --account=rrg-czg
#SBATCH --job-name=orbit_daily_l2
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/orbit_daily_l2_%j.out
#SBATCH --error=logs/orbit_daily_l2_%j.err

# IMPORTANT: config.toml's n_workers must match --cpus-per-task above (50).
# If you change one, change the other — a mismatch either leaves allocated
# cores idle (n_workers too low) or oversubscribes them (n_workers too high).
#
# WORKER COUNT / MEMORY (revised after a real OOM kill): a previous attempt
# at 90 workers x 8G/core (720G, only ~4% under the 750G node ceiling) was
# OOM-killed ~5h in. sacct showed MaxRSS = 754968848K =~720GiB -- almost
# EXACTLY the requested 720G, not a wild overshoot, meaning the aggregate
# usage across 90 concurrent workers genuinely reached the requested
# ceiling with essentially no margin, and some transient spike (plausibly
# one of the slow max_nfev cases -- several profiles in that run's log ran
# 1000-1268s, meaning ~200 RT/Jacobian evaluations' worth of accumulated
# state in that one worker at once) pushed it over.
#
# This config trades some walltime for real headroom instead of requesting
# right up to the node's edge: 50 workers x 12G/core = 600G, ~80% of the
# 750G node total, leaving ~20% margin for the OS, the main orchestrating
# process, and per-worker spikes -- vs. the previous config's ~4% margin.
# Given the ~2449 CPU-hour total cost is negligible against a 1,600
# core-year allocation, this is a deliberate choice: memory safety over
# shaving walltime, now that we have real evidence 8G/core wasn't enough
# margin under peak load.
#
# process_day() also now uses max_tasks_per_child=1 (each worker is
# recycled after one day, capping any cross-day memory accumulation within
# a single long-lived worker process) as a complementary safeguard --
# doesn't hurt regardless of whether the OOM was from a single-day spike,
# cross-day growth, or both.
#
# STILL HITS OOM? Check `sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,
# NNodes,State` again. If MaxRSS is still pinned near the requested total,
# a single day's peak footprint may exceed even 12G/core for some worker --
# increase --mem-per-cpu further and reduce --cpus-per-task accordingly
# (day-level/profile-level resume make this safe to iterate on incrementally
# rather than guessing right on the first try).

set -euo pipefail

# Pin BLAS/OpenMP threading to 1 per process. Without this, numpy/scipy/
# sasktran2's underlying BLAS library may spawn its own internal threads
# per worker (e.g. 4 each), which at 50 independent worker PROCESSES could
# oversubscribe the 50 allocated cores several times over and collapse
# performance via thread-thrashing rather than actually running 50-way
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
echo "Workers requested: 50 (cpus-per-task) -- confirm config.toml's n_workers matches"

cd /project/6079534/vmcd/cesm-hawc/scripts
python run_orbit_daily.py

echo "Job finished: $(date)"
