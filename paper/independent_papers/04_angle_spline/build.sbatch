#!/bin/bash
#SBATCH --job-name=angle-paper
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-angle-paper-%j.out
#SBATCH --error=slurm-angle-paper-%j.err

set -euo pipefail

cd "$HOME/edge-detection-filter-critique/paper/independent_papers/04_angle_spline"

echo "Building angle-spline paper at $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

tectonic --keep-logs paper.tex

echo
echo "Build complete:"
ls -lh paper.pdf paper.log
