#!/bin/bash
#SBATCH --job-name=paper-build
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-paper-%j.out
#SBATCH --error=slurm-paper-%j.err

set -euo pipefail

cd $HOME/edge-detection-filter-critique/paper

echo "Building paper at $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

tectonic --keep-logs critique.tex

echo
echo "Build complete:"
ls -lh critique.pdf critique.log
