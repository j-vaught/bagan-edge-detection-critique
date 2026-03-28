#!/bin/bash
#SBATCH --job-name=bagan-critique
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load python3/anaconda/3.12
module load cuda/12.1
export PYTHONNOUSERSITE=1

cd $HOME/edge-detection-filter-critique

echo "Starting critique analysis at $(date)"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

python3 src/analyze_results.py

echo "Completed at $(date)"
echo "Results in: $HOME/edge-detection-filter-critique/results/"
ls -la results/
