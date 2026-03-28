#!/bin/bash
#SBATCH --job-name=gpu-probe
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-probe-%j.out
#SBATCH --error=slurm-probe-%j.err

module load python3/anaconda/3.12
module load cuda/12.1

export PYTHONNOUSERSITE=1

echo "Starting GPU probe at $(date)"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

cd $HOME/edge-detection-filter-critique
export PROBE_TAG=${PROBE_TAG:-${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}}
python -u benchmarks/gpu_env_probe.py

echo "Completed at $(date)"
