#!/bin/bash
# SLURM job script for `cesm-hawc run --mode orbit` on one node.
#
# The logs/ directory must exist before you submit (SLURM opens the log
# files before this script runs). Submit from the repo root:
#   sbatch scripts/submit.sh [config.toml] [case_name] [strip-ozone]
#
# config.toml defaults to "config.toml" in the submit directory if omitted.
# case_name, if given, overrides [case] name via --case-name (and fills
# {name} in [case] waccm_dir), so you can queue one job per case off a
# single shared config.toml instead of maintaining a config file per case:
#   sbatch scripts/submit.sh config.toml case_a
#   sbatch scripts/submit.sh config.toml case_b
#
# strip-ozone, if given as the literal string "strip-ozone", passes
# --strip-ozone through (zeroes WACCM ozone before simulating). Reads the same case's h2 files but
# writes to <case_name>_no_ozone/ instead of <case_name>/, so it's safe to
# queue alongside a normal run of the same case off the same config.toml:
#   sbatch scripts/submit.sh config.toml case_a strip-ozone
#SBATCH --account=def-yourPI          # ← change to your PI's allocation account
#SBATCH --job-name=cesm_hawc
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/cesm_hawc_%j.out
#SBATCH --error=logs/cesm_hawc_%j.err

# The worker count is set from --cpus-per-task (overriding n_workers in
# config.toml). Override resources at submit time rather than editing this
# file, e.g.  sbatch --cpus-per-task=16 --time=12:00:00 scripts/submit.sh
set -euo pipefail

# Pin BLAS/OpenMP threading to 1 per process. Must be set before Python starts.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG="${1:-config.toml}"
CASE_NAME="${2:-}"
STRIP_OZONE_FLAG="${3:-}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate hawc_env
unset PYTHONPATH

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG"
echo "Case name override: ${CASE_NAME:-<none -- using the configured [case] name>}"
echo "Strip ozone: ${STRIP_OZONE_FLAG:-<none -- using the configured strip_ozone>}"
echo "Workers: $SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
CESM_HAWC_ARGS=(run --config "$CONFIG" --mode orbit --n-workers "$SLURM_CPUS_PER_TASK")
if [ -n "$CASE_NAME" ]; then
    CESM_HAWC_ARGS+=(--case-name "$CASE_NAME")
fi
if [ "$STRIP_OZONE_FLAG" = "strip-ozone" ]; then
    CESM_HAWC_ARGS+=(--strip-ozone)
fi
cesm-hawc "${CESM_HAWC_ARGS[@]}"

echo "Job finished: $(date)"
