#!/bin/bash
# create_env.sh
# =============
# Create the cesm_hawc working environment on Alliance Canada HPC.
#
# Usage:
#   bash scripts/setup/create_env.sh
#
# After this completes, activate the environment with:
#   micromamba activate hawc_env


set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== Setting up micromamba ==="
export PATH="$HOME/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"

echo "=== Creating hawc_env ==="
micromamba create -n hawc_env -c conda-forge \
    python=3.11 \
    sasktran2 \
    xarray \
    netcdf4 \
    scipy \
    numpy \
    matplotlib \
    pandas \
    cartopy \
    numba \
    dask \
    scikit-learn \
    -y

micromamba activate hawc_env

echo "=== Installing cesm-hawc (editable, with the [sim] extra) ==="
# hawcsimulator 2026.10 requires sasktran2 2026.10.1, which conda-forge doesn't
# have yet; keep the conda-forge sasktran2 rather than a pip wheel.
pip install -e "$REPO_DIR[sim]" "hawcsimulator>=2026.6.0,<2026.10"

echo ""
echo "Testing imports..."
python -c "import sasktran2; print('  sasktran2 OK')"
python -c "import hawcsimulator; print('  hawcsimulator OK')"
python -c "import cesm_hawc; print('  cesm_hawc OK')"
echo ""
echo "Environment created successfully."
echo "Activate with:  micromamba activate hawc_env"
