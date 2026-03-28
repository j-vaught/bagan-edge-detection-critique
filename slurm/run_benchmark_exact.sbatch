#!/bin/bash
#SBATCH --job-name=edge-exact
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-exact-%j.out
#SBATCH --error=slurm-exact-%j.err

module load python3/anaconda/3.12
module load cuda/12.1

export PYTHONNOUSERSITE=1
mkdir -p $HOME/.bench_pkgs
if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("cv2") else 1)
PY
then
    pip install --target=$HOME/.bench_pkgs --no-deps kornia kornia_rs opencv-python-headless 2>&1 | tail -3
fi
export PYTHONPATH=$HOME/.bench_pkgs:$PYTHONPATH

echo "Starting exact-threshold benchmark at $(date)"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

cd $HOME/edge-detection-filter-critique

export BENCH_OUTPUT_TAG=${BENCH_OUTPUT_TAG:-_exact1001}
export BENCH_DATASETS=${BENCH_DATASETS:-BSDS500:UDED}
export BENCH_MAX_IMAGES=${BENCH_MAX_IMAGES:-50}
export BENCH_N_THRESHOLDS=${BENCH_N_THRESHOLDS:-1001}
export BENCH_MATCH_RADIUS=${BENCH_MATCH_RADIUS:-3}
export RUN_TRADITIONAL=${RUN_TRADITIONAL:-1}
export RUN_WVF_LF=${RUN_WVF_LF:-0}
export RUN_ML=${RUN_ML:-1}

python -u benchmarks/benchmark_all.py

echo "Completed at $(date)"
ls -la benchmarks/results/
