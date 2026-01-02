#!/bin/bash -l

#SBATCH --job-name=rcrl_visual
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH -e rcrl_visual.err
#SBATCH -o rcrl_visual.out

set -euo pipefail

REPO=/scratch_root/wy524/Reachability_Constrained_RL

export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR"

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate rcrl

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TF_ENABLE_ONEDNN_OPTS=0

cd "$REPO/visualize_scripts"

# python visualize_quadrotor_trajectory.py

python visualize_region_quadrotor.py