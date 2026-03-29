#!/bin/bash
#SBATCH --job-name=full-ablation
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --output=slurm-full-ablation-%j.out
#SBATCH --error=slurm-full-ablation-%j.err

module load python3/anaconda/3.12

echo "Full dataset ablation at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1

python3 -u scripts/run_full_dataset_ablation.py
