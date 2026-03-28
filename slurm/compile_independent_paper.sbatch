#!/bin/bash
#SBATCH --job-name=ind-paper
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-ind-paper-%j.out
#SBATCH --error=slurm-ind-paper-%j.err

set -euo pipefail

if [[ -z "${PAPER_DIR:-}" ]]; then
  echo "PAPER_DIR is required" >&2
  exit 1
fi

if [[ -z "${MAIN_TEX:-paper.tex}" ]]; then
  echo "MAIN_TEX is required" >&2
  exit 1
fi

cd "$HOME/edge-detection-filter-critique/$PAPER_DIR"

echo "Building $PAPER_DIR/$MAIN_TEX at $(date)"
echo "Node: $(hostname)"

tectonic --keep-logs "$MAIN_TEX"

echo
echo "Build complete:"
ls -lh "${MAIN_TEX%.tex}.pdf" "${MAIN_TEX%.tex}.log"
