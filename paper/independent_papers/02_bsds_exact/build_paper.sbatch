#!/bin/bash
#SBATCH --job-name=bsds-exact-paper
#SBATCH --output=/home/jvaught/edge-detection-filter-critique/paper/independent_papers/02_bsds_exact/build_%j.out
#SBATCH --error=/home/jvaught/edge-detection-filter-critique/paper/independent_papers/02_bsds_exact/build_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

module load singularity/4.1.4

WORKDIR=/home/jvaught/edge-detection-filter-critique/paper/independent_papers/02_bsds_exact
IMAGE=docker://ghcr.io/xu-cheng/texlive-small:latest
TEXMFDIR="$WORKDIR/texmfhome"

cd "$WORKDIR"

mkdir -p "$TEXMFDIR" "$TEXMFDIR/var" "$TEXMFDIR/config"

mkdir -p "$TEXMFDIR/tex/latex/ieeetran"
curl -fsSL -o "$TEXMFDIR/tex/latex/ieeetran/IEEEtran.cls" https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls

singularity exec --bind "$WORKDIR":"$WORKDIR" "$IMAGE" env TEXMFHOME="$TEXMFDIR" TEXMFVAR="$TEXMFDIR/var" TEXMFCONFIG="$TEXMFDIR/config" pdflatex -interaction=nonstopmode -halt-on-error paper.tex
singularity exec --bind "$WORKDIR":"$WORKDIR" "$IMAGE" env TEXMFHOME="$TEXMFDIR" TEXMFVAR="$TEXMFDIR/var" TEXMFCONFIG="$TEXMFDIR/config" pdflatex -interaction=nonstopmode -halt-on-error paper.tex
