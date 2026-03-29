#!/bin/bash
#SBATCH --job-name=canny-test
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-canny-test-%j.out
#SBATCH --error=slurm-canny-test-%j.err

module load python3/anaconda/3.12

echo "Canny post-processing test at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1

python3 -u scripts/canny_postprocess_test.py
