#!/bin/bash
# submit.sh — SLURM batch script for Alliance Canada clusters (Fir/Rorqual/Narval)
#
# Edit ACCOUNT, MAIL, and the paths in config.toml (copy config.example.toml
# to get started), then submit from the repo root:
#   sbatch scripts/submit.sh
#
#SBATCH --job-name=waccm_ali
#SBATCH --account=def-yourPI          # ← change to your PI's allocation account
#SBATCH --time=00:20:00               # single column converges in ~5–10 min
#SBATCH --cpus-per-task=1             # retrieval is serial; 1 core is optimal
#SBATCH --mem=18G
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@institution.ca  # ← change to your email

mkdir -p logs

# ── Activate micromamba environment ──────────────────────────────────────────
export PATH="$HOME/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate hawc_env

# Single-threaded — matches --cpus-per-task=1
export OMP_NUM_THREADS=1

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"
cesm-hawc run --config config.toml --mode single
