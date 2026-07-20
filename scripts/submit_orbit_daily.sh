#!/bin/bash
#SBATCH --account=rrg-czg
#SBATCH --job-name=orbit_daily
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/orbit_daily_%j.out
#SBATCH --error=logs/orbit_daily_%j.err

# IMPORTANT: config.toml's n_workers must match --cpus-per-task above (8).
# If you change one, change the other — a mismatch either leaves allocated
# cores idle (n_workers too low) or oversubscribes them (n_workers too high).

set -euo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate hawc_env
unset PYTHONPATH

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /project/6079534/vmcd/cesm-hawc/scripts
python run_orbit_daily.py

echo "Job finished: $(date)"