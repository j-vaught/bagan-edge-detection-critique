#!/bin/bash
#SBATCH --job-name=edge-bench
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-bench-%j.out
#SBATCH --error=slurm-bench-%j.err

module load python3/anaconda/3.12
module load cuda/12.1

pip install --user 'numpy==2.0.2' opencv-python-headless kornia 'scikit-image>=0.22' 2>&1 | tail -3

cd $HOME/edge-detection-filter-critique

echo "Starting benchmark at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run with 50 images initially (set BENCH_MAX_IMAGES for full)
export BENCH_MAX_IMAGES=${BENCH_MAX_IMAGES:-50}

python benchmarks/benchmark_all.py

echo "Completed at $(date)"
echo "Results:"
ls -la benchmarks/results/
