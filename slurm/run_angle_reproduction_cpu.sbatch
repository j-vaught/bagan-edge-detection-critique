#!/bin/bash
#SBATCH --job-name=angle-sweep
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-angle-%j.out
#SBATCH --error=slurm-angle-%j.err

module load python3/anaconda/3.12

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ANGLE_SWEEP_WORKERS=${ANGLE_SWEEP_WORKERS:-${SLURM_CPUS_PER_TASK}}

echo "Starting angle reproduction sweep at $(date)"
echo "Node: $(hostname)"
echo "Workers: ${ANGLE_SWEEP_WORKERS}"

cd $HOME/edge-detection-filter-critique

export ANGLE_SWEEP_NP_COUNTS=${ANGLE_SWEEP_NP_COUNTS:-15,50,100,150,250}
export ANGLE_SWEEP_ORIENTATIONS=${ANGLE_SWEEP_ORIENTATIONS:-18,36,72}
export ANGLE_SWEEP_SNRS=${ANGLE_SWEEP_SNRS:-2.0,1.0,0.75,0.5}
export ANGLE_SWEEP_TEST_ANGLES=${ANGLE_SWEEP_TEST_ANGLES:-0,23,63.5,90,135,174}
export ANGLE_SWEEP_IMAGE_SIZE=${ANGLE_SWEEP_IMAGE_SIZE:-128}
export ANGLE_SWEEP_ORDER=${ANGLE_SWEEP_ORDER:-4}
export ANGLE_SWEEP_SAMPLE_POINTS=${ANGLE_SWEEP_SAMPLE_POINTS:-9}
export ANGLE_SWEEP_TAG=${ANGLE_SWEEP_TAG:-paperlike}

python -u benchmarks/angle_reproduction_sweep.py

echo "Completed at $(date)"
ls -la benchmarks/angle_sweep_results/
