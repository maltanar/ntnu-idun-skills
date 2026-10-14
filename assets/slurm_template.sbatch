#!/bin/bash
#SBATCH --job-name=idun-job
#SBATCH --partition=GPUQ                # CPUQ for CPU-only, GPUQ for GPU jobs
#SBATCH --account=share-<your-group>    # Group account; share-* for higher priority
#SBATCH --time=01-00:00:00              # DD-HH:MM:SS (GPUQ limit: 14 days)
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1               # Supported architectures: h100, a100, v100, p100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/cluster/work/%u/logs/%x-%j.out
#SBATCH --error=/cluster/work/%u/logs/%x-%j.err

# Safe strictness: DO NOT enable 'set -u' before module setup
set -eo pipefail

echo "=== IDUN Job Diagnostics ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
command -v nvidia-smi >/dev/null && nvidia-smi

# 1. Purge and load base Python/Anaconda module
module purge
module load Anaconda3/2025.06-1

# 2. Activate Environment (Choose venv or Conda)
# --- Option A: Python venv (Recommended for cross-machine portability) ---
ENV_PATH="/cluster/work/$USER/venvs/my-project"
source "$ENV_PATH/bin/activate"

# --- Option B: Conda environment (uncomment if using Conda) ---
# conda activate /cluster/work/$USER/conda-envs/my-project

# 3. Environment variables & caching
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="/cluster/work/$USER/.cache/pip"
export WANDB_MODE=offline
export WANDB_DIR="/cluster/work/$USER/wandb"

# 4. Run training
# Source code in /cluster/home/$USER, heavy data & checkpoints in /cluster/work/$USER
PROJECT_DIR="/cluster/home/$USER/my-project"
WORK_DIR="/cluster/work/$USER"

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/checkpoints" "$WANDB_DIR"

cd "$PROJECT_DIR"
python -u train.py \
    --data-dir "$WORK_DIR/datasets" \
    --checkpoint-dir "$WORK_DIR/checkpoints"
