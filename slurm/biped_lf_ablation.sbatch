#!/bin/bash
#SBATCH --job-name=biped-lf-abl
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-biped-lf-abl-%j.out
#SBATCH --error=slurm-biped-lf-abl-%j.err

module load python3/anaconda/3.12

echo "BIPED LF ablation at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique

# Install package to user site
pip install --user -e . 2>&1 | tail -1

python3 -u scripts/run_biped_ablation_full.py
