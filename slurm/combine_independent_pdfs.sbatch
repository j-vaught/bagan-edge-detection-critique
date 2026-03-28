#!/bin/bash
#SBATCH --job-name=ind-pdf-merge
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-ind-merge-%j.out
#SBATCH --error=slurm-ind-merge-%j.err

set -euo pipefail

cd "$HOME/edge-detection-filter-critique"

OUTFILE="${OUTFILE:-paper/independent_papers/independent_bundle.pdf}"

mapfile -t pdfs < <(find paper/independent_papers -mindepth 2 -maxdepth 2 -name 'paper.pdf' | sort)

if [[ "${#pdfs[@]}" -eq 0 ]]; then
  echo "No paper.pdf files found under paper/independent_papers" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTFILE")"

echo "Merging ${#pdfs[@]} PDFs into $OUTFILE"
pdfunite "${pdfs[@]}" "$OUTFILE"

ls -lh "$OUTFILE"
