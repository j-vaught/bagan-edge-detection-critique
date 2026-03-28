#!/bin/bash
#SBATCH --job-name=wvf-scale
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-wvf-scale-%j.out
#SBATCH --error=slurm-wvf-scale-%j.err

module load python3/anaconda/3.12

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WVF_SWEEP_WORKERS=${WVF_SWEEP_WORKERS:-${SLURM_CPUS_PER_TASK}}

echo "Starting WVF/LF scale sweep at $(date)"
echo "Node: $(hostname)"
echo "Workers: ${WVF_SWEEP_WORKERS}"

cd $HOME/edge-detection-filter-critique

export SWEEP_DATASET=${SWEEP_DATASET:-UDED}
export SWEEP_MAX_IMAGES=${SWEEP_MAX_IMAGES:-5}
export SWEEP_NP_COUNTS=${SWEEP_NP_COUNTS:-15,50,100,150,250}
export SWEEP_LF_HALF_WIDTHS=${SWEEP_LF_HALF_WIDTHS:-3,7,14}
export SWEEP_RESIZE_SHAPES=${SWEEP_RESIZE_SHAPES:-64x64,96x96}
export SWEEP_N_ORIENTATIONS=${SWEEP_N_ORIENTATIONS:-18}
export SWEEP_ORDER=${SWEEP_ORDER:-4}
export SWEEP_N_THRESHOLDS=${SWEEP_N_THRESHOLDS:-1001}
export SWEEP_MATCH_RADIUS=${SWEEP_MATCH_RADIUS:-3}
export SWEEP_LF_SUBSAMPLE=${SWEEP_LF_SUBSAMPLE:-2}
export SWEEP_TAG=${SWEEP_TAG:-${SWEEP_DATASET,,}_paperlike}

python -u benchmarks/wvf_scale_sweep.py

echo "Completed at $(date)"
ls -la benchmarks/wvf_scale_results/
