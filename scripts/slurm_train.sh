#!/bin/bash
# =====================================================================
# SLURM Job Script — FinAgent QLoRA Training on L40S
# =====================================================================
#
# HOW TO USE:
#   sbatch scripts/slurm_train.sh                        # basic run
#   sbatch scripts/slurm_train.sh --use_wandb             # with W&B tracking
#   sbatch scripts/slurm_train.sh --push_to_hub --use_wandb  # full pipeline
#
# MONITOR:
#   squeue -u $USER                      # check job status
#   tail -f logs/finagent_<JOB_ID>.out   # watch training output
#   scancel <JOB_ID>                     # cancel if needed
#
# WHAT HAPPENS:
#   1. SLURM allocates an L40S GPU node
#   2. We load CUDA & Python modules
#   3. Create (or reuse) a Python venv with Unsloth
#   4. Download training data & run fine-tuning
#   5. Copy results from fast local disk ($SLURM_TMPDIR) to your home
# =====================================================================

# ── SLURM resource request ─────────────────────────────────────────
#SBATCH --job-name=finagent-qlora
#SBATCH --partition=gpu           # GPU partition (change if yours is named differently)
#SBATCH --gres=gpu:l40s:1        # 1x NVIDIA L40S (48 GB VRAM)
#SBATCH --cpus-per-task=8        # 8 CPU cores (for data loading workers)
#SBATCH --mem=64G                # 64 GB RAM (model + data + overhead)
#SBATCH --time=02:00:00          # 2h max (training takes ~15 min, rest is setup/saving)
#SBATCH --output=logs/finagent_%j.out   # stdout → logs/finagent_<JOB_ID>.out
#SBATCH --error=logs/finagent_%j.err    # stderr → logs/finagent_<JOB_ID>.err

# Exit on any error (-e), undefined variable (-u), or pipe failure (-o pipefail)
set -euo pipefail

echo "============================================================"
echo "  FinAgent QLoRA — SLURM Job"
echo "============================================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  Partition:  $SLURM_JOB_PARTITION"
echo "  GPUs:       $SLURM_GPUS_ON_NODE"
echo "  CPUs:       $SLURM_CPUS_PER_TASK"
echo "  Memory:     $SLURM_MEM_PER_NODE MB"
echo "  Date:       $(date)"
echo "============================================================"
echo ""

# ── 1. Load cluster modules ────────────────────────────────────────
# 'module load' makes CUDA and Python available on the compute node.
# These module names vary by cluster — check `module avail` on yours.
echo "[1/7] Loading modules ..."
module load cuda python
echo "  CUDA:   $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'unknown')"
echo "  Python: $(python --version)"
echo ""

# ── 2. Create or reuse Python virtual environment ──────────────────
# We install all dependencies in a venv so they persist between jobs.
# First run: creates venv + installs everything (~5 min).
# Subsequent runs: just activates the existing venv (~instant).
echo "[2/7] Setting up Python venv ..."
VENV_DIR="$HOME/venvs/finagent"
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating new venv at $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    echo "  Venv created"
else
    echo "  Reusing existing venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "  Python: $(which python)"
echo "  pip:    $(pip --version | awk '{print $2}')"
echo ""

# ── 3. Install dependencies ────────────────────────────────────────
# Unsloth: optimized QLoRA training (2x faster than standard PEFT)
# huggingface_hub: pinned to 0.34.0 because:
#   - Unsloth pins 0.27.1 (too old for current transformers)
#   - Latest versions remove HF_HUB_ENABLE_HF_TRANSFER that Unsloth needs
#   - 0.34.0 is the sweet spot that works with both
# wandb: experiment tracking (optional but recommended)
echo "[3/7] Installing dependencies ..."
pip install --quiet unsloth
echo "  Unsloth installed"
pip install --quiet --force-reinstall huggingface_hub==0.34.0
echo "  huggingface_hub pinned to 0.34.0"
pip install --quiet wandb
echo "  wandb installed"
echo ""

# ── 4. GPU sanity check ────────────────────────────────────────────
# Verify we actually got a GPU and it's an L40S.
echo "[4/7] GPU check ..."
nvidia-smi
echo ""

# ── 5. Prepare directories ─────────────────────────────────────────
# Resolve project directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create log directory (SLURM needs it to exist for --output/--error)
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"

# ── 6. Run training ────────────────────────────────────────────────
# We train from $SLURM_TMPDIR (fast local NVMe on the node) instead
# of the network filesystem. This avoids I/O bottlenecks when saving
# checkpoints and the merged model (~5 GB write).
echo "[5/7] Starting training ..."
WORK_DIR="${SLURM_TMPDIR:-/tmp}/finagent_$$"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
echo "  Working directory: $WORK_DIR (fast local disk)"
echo "  Project directory: $PROJECT_DIR"
echo ""

# Load .env if it exists (for WANDB_API_KEY and other secrets)
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "  Loading .env file ..."
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Launch the training script
# "$@" passes through any extra flags from sbatch command line:
#   sbatch scripts/slurm_train.sh --use_wandb --push_to_hub
python "$PROJECT_DIR/scripts/train_qlora.py" \
    --output_dir "$WORK_DIR/finagent-checkpoints" \
    --adapter_path "$WORK_DIR/finagent-7b-lora" \
    --merged_path "$WORK_DIR/finagent-7b-merged" \
    "$@"

# ── 7. Copy results back to home ───────────────────────────────────
# $SLURM_TMPDIR is wiped when the job ends, so we need to copy
# the trained models back to the project directory.
echo ""
echo "[6/7] Copying results to $PROJECT_DIR/results/ ..."

if [ -d "$WORK_DIR/finagent-7b-lora" ]; then
    cp -r "$WORK_DIR/finagent-7b-lora" "$PROJECT_DIR/results/"
    echo "  LoRA adapter copied"
else
    echo "  [!] LoRA adapter not found — training may have failed"
fi

if [ -d "$WORK_DIR/finagent-7b-merged" ]; then
    cp -r "$WORK_DIR/finagent-7b-merged" "$PROJECT_DIR/results/"
    echo "  Merged model copied"
else
    echo "  [!] Merged model not found — training may have failed"
fi

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Cleanup"
echo "============================================================"
echo "  Job finished:  $(date)"
echo "  Results in:    $PROJECT_DIR/results/"
echo "  LoRA adapter:  $PROJECT_DIR/results/finagent-7b-lora/"
echo "  Merged model:  $PROJECT_DIR/results/finagent-7b-merged/"
echo ""
echo "  Next step: use the merged model in Module 3 (agentic tool-use)"
echo "============================================================"
