#!/bin/bash
#SBATCH --job-name=wfn-full
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-wfn-full-%j.out
#SBATCH --error=slurm-wfn-full-%j.err

module load python3/anaconda/3.12

echo "WideFieldNet full-res at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique
pip install --user -e . 2>&1 | tail -1

python3 -u scripts/run_widefieldnet_sliding.py
