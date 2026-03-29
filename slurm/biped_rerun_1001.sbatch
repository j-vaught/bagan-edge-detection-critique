#!/bin/bash
#SBATCH --job-name=biped-1001
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-biped-1001-%j.out
#SBATCH --error=slurm-biped-1001-%j.err

module load python3/anaconda/3.12

echo "BIPED rerun (1001 thresholds) at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1

python3 -u scripts/rerun_biped_1001.py
