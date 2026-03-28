#!/bin/bash
#SBATCH --job-name=agent-bundle
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-agent-bundle-%j.out
#SBATCH --error=slurm-agent-bundle-%j.err

set -euo pipefail

cd $HOME/edge-detection-filter-critique/paper/agent_drafts

echo "Building agent bundle at $(date)"
echo "Node: $(hostname)"

tectonic --keep-logs agent_bundle.tex

echo
echo "Build complete:"
ls -lh agent_bundle.pdf agent_bundle.log
