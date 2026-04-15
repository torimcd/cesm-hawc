#!/bin/bash
# build_wheels.sh
# ===============
# Compile sasktran2 and hawcsimulator wheels on Alliance Canada HPC.
#
# Run this ONCE after cloning the repo, before create_env.sh.
# Wheels are saved to ~/wheels/ and reused by create_env.sh.
#
# This step is needed because sasktran2 has a Rust extension that must
# be compiled from source — it is not available as a pre-built Alliance wheel.
#
# Usage:
#   bash scripts/setup/build_wheels.sh
#
# Expected runtime: 5–15 minutes depending on node load.

set -e   # exit on first error

WHEEL_DIR="$HOME/wheels"
mkdir -p "$WHEEL_DIR"

echo "=== Loading build modules ==="
# StdEnv/2023 already provides gcc, flexiblas, openblas — do not purge or
# reload the standard environment. Only add cmake and rust on top.
module load python/3.11
module load cmake rust

echo "=== Creating temporary build environment ==="
TMPENV="/tmp/build_env_$$"
virtualenv --no-download "$TMPENV"
source "$TMPENV/bin/activate"
pip install --no-index --upgrade pip

echo "=== Building sasktran2 wheel ==="
pip wheel --no-deps -w "$WHEEL_DIR" sasktran2

echo "=== Building hawcsimulator wheel ==="
pip wheel --no-deps -w "$WHEEL_DIR" hawcsimulator

echo "=== Cleaning up temporary environment ==="
deactivate
rm -rf "$TMPENV"

echo ""
echo "Wheels saved to $WHEEL_DIR:"
ls "$WHEEL_DIR"
echo ""
echo "Next step: run scripts/setup/create_env.sh"
