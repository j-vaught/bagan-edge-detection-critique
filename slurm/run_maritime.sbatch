#!/bin/bash
#SBATCH --job-name=maritime
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-maritime-%j.out
#SBATCH --error=slurm-maritime-%j.err

module load python3/anaconda/3.12
module load cuda/12.1
export PYTHONNOUSERSITE=1
mkdir -p $HOME/.bench_pkgs
export PYTHONPATH=$HOME/.bench_pkgs:$PYTHONPATH
export MARITIME_TRIALS=${MARITIME_TRIALS:-10}
export MARITIME_BASE_SEED=${MARITIME_BASE_SEED:-1234}

echo "Starting maritime benchmark at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

cd $HOME/edge-detection-filter-critique
python -u benchmarks/maritime_benchmark.py

echo "Completed at $(date)"
ls -la benchmarks/maritime_results/
