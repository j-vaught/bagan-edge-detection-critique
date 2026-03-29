#!/bin/bash
#SBATCH --job-name=wvf-cuda-test
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-wvf-cuda-test-%j.out
#SBATCH --error=slurm-wvf-cuda-test-%j.err

module load python3/anaconda/3.12

export PYTHONNOUSERSITE=1

echo "WVF CUDA speed test at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

cd $HOME/edge-detection-filter-critique/src
python -u wvf_cuda.py
