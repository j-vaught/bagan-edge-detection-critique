#!/bin/bash
#SBATCH --job-name=uded-paper
#SBATCH --output=build_paper.%j.out
#SBATCH --error=build_paper.%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /home/jvaught/edge-detection-filter-critique/paper/independent_papers/03_uded_exact
tectonic paper.tex
