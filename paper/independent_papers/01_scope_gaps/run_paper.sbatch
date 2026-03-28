#!/bin/bash
#SBATCH --job-name=scope-gaps-paper
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-scope-gaps-%j.out
#SBATCH --error=slurm-scope-gaps-%j.err

set -euo pipefail

cd $HOME/edge-detection-filter-critique/paper/independent_papers/01_scope_gaps

echo "Building scope-gaps paper at $(date)"
echo "Node: $(hostname)"
tectonic --keep-logs paper.tex

echo
echo "Build complete:"
ls -lh paper.pdf paper.log
