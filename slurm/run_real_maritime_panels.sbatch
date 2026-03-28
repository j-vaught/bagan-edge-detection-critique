#!/bin/bash
#SBATCH --job-name=real-maritime
#SBATCH --partition=gpu-H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-real-maritime-%j.out
#SBATCH --error=slurm-real-maritime-%j.err

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

echo "Starting real maritime panels at $(date)"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

cd $HOME/edge-detection-filter-critique

valid_count=$(find datasets/maritime/images -maxdepth 1 -type f -size +0c | wc -l)
if [ "$valid_count" -eq 0 ]; then
    echo "No non-empty maritime images are available in datasets/maritime/images"
    exit 1
fi

export REAL_MARITIME_LIMIT=${REAL_MARITIME_LIMIT:-6}
export REAL_MARITIME_MAX_SIDE=${REAL_MARITIME_MAX_SIDE:-192}
export REAL_MARITIME_TAG=${REAL_MARITIME_TAG:-paper}
export REAL_MARITIME_WVF_NP=${REAL_MARITIME_WVF_NP:-15}
export REAL_MARITIME_LF_HALF_WIDTH=${REAL_MARITIME_LF_HALF_WIDTH:-3}
export REAL_MARITIME_ORIENTATIONS=${REAL_MARITIME_ORIENTATIONS:-18}
export REAL_MARITIME_LF_SUBSAMPLE=${REAL_MARITIME_LF_SUBSAMPLE:-2}

python -u benchmarks/real_maritime_panels.py

echo "Completed at $(date)"
ls -la benchmarks/real_maritime_results/
