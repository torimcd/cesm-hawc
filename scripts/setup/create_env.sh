#!/bin/bash
# create_env.sh
# =============
# Create the cesm_hawc working environment on Alliance Canada HPC.
#
# Run this after build_wheels.sh has completed successfully.
# Uses micromamba for the environment and installs sasktran2/hawcsimulator
# from the pre-built wheels in ~/wheels/.
#
# Usage:
#   bash scripts/setup/create_env.sh
#
# After this completes, activate the environment with:
#   micromamba activate hawc_env

set -e

WHEEL_DIR="$HOME/wheels"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== Loading modules ==="
module load python/3.11

echo "=== Setting up micromamba ==="
export PATH="$HOME/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"

echo "=== Creating hawc_env ==="
micromamba create -n hawc_env -c conda-forge \
    python=3.11 \
    xarray \
    scipy \
    numpy \
    matplotlib \
    pandas \
    cartopy \
    numba \
    -y

micromamba activate hawc_env

echo "=== Installing sasktran2 and hawcsimulator from wheels ==="
pip install --no-index --find-links="$WHEEL_DIR" sasktran2 hawcsimulator

echo "=== Installing cesm-hawc-ali in editable mode ==="
pip install -e "$REPO_DIR"

echo ""
echo "Environment created successfully."
echo "Activate with:  micromamba activate hawc_env"
