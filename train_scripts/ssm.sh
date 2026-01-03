#!/bin/bash -l
#SBATCH --job-name=ssm
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH -e ssm.err
#SBATCH -o ssm.out

set -euo pipefail

# -------------------------
# Paths / Env
# -------------------------
REPO=/scratch_root/wy524/Reachability_Constrained_RL

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate rcrl

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# matplotlib cache (avoid HOME quota)
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"

# -------------------------
# TF: make logs quieter + avoid oneDNN (optional)
# -------------------------
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2   # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# H200 (sm_90): enable CUDA JIT cache to avoid repeated "compile from PTX"
export CUDA_CACHE_PATH=/scratch_root/wy524/.cuda_cache
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB
mkdir -p "$CUDA_CACHE_PATH"

# Optional: control TF GPU memory behavior (pick ONE)
# export TF_FORCE_GPU_ALLOW_GROWTH=true   # safer for multi-proc
# export TF_GPU_ALLOCATOR=cuda_malloc_async # sometimes improves fragmentation

# -------------------------
# Ray: make it cluster-friendly
#  - put temp files in scratch
#  - disable dashboard/metrics exporter to avoid rpc_code:14 spam
# -------------------------
export RAY_TMPDIR=/scratch_root/wy524/tmp/ray_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR"

export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0

# Log dedup is on by default; keep it on (less spam)
# export RAY_DEDUP_LOGS=1

# -------------------------
# Reduce warning spam (gym / deprecations)
# -------------------------
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning:gym.*"

# -------------------------
# Threading: avoid oversubscription (important with Ray + TF)
# -------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Optional reproducibility
export PYTHONHASHSEED=0

# -------------------------
# Diagnostics (nice to have)
# -------------------------
echo "===== Job Info ====="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python -V
python -c "import ray, tensorflow as tf, tensorflow_probability as tfp; import google.protobuf; \
print('ray', ray.__version__); print('tf', tf.__version__); print('tfp', tfp.__version__); print('protobuf', google.protobuf.__version__); \
print('gpus', tf.config.list_physical_devices('GPU'))" || true
echo "===================="

# -------------------------
# Run training
# Use srun to bind resources properly under Slurm
# -------------------------
cd "$REPO/train_scripts"
srun --cpu-bind=cores python ./train_scripts4ssm.py
