#!/bin/bash
#SBATCH --job-name=gen-figs
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-figs-%j.out
#SBATCH --error=slurm-figs-%j.err

module load python3/anaconda/3.12
export PYTHONNOUSERSITE=1
export PYTHONPATH=$HOME/.bench_pkgs:$PYTHONPATH

cd $HOME/edge-detection-filter-critique
python -u uofsc-beamer/generate_figures.py

echo "Done. Figures:"
ls -lh uofsc-beamer/figures/*.pdf 2>/dev/null
