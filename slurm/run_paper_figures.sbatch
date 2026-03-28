#!/bin/bash
#SBATCH --job-name=paper-figs
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-paper-figs-%j.out
#SBATCH --error=slurm-paper-figs-%j.err

set -euo pipefail

module load python3/anaconda/3.12
export PYTHONNOUSERSITE=1

cd $HOME/edge-detection-filter-critique

echo "Generating paper figures at $(date)"
echo "Node: $(hostname)"
python -u paper/generate_figures.py

echo
echo "Generated figures:"
ls -lh paper/figures
