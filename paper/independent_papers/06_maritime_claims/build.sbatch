#!/bin/bash
#SBATCH --job-name=maritime-paper
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-maritime-paper-%j.out
#SBATCH --error=slurm-maritime-paper-%j.err

set -euo pipefail

cd $HOME/edge-detection-filter-critique/paper/independent_papers/06_maritime_claims

echo "Building maritime claims paper at $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

tectonic --keep-logs paper.tex

echo
echo "Build complete:"
ls -lh paper.pdf paper.log
