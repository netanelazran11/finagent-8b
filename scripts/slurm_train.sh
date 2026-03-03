#!/bin/bash
# =====================================================================
# SLURM Job Script — FinAgent QLoRA Training
# =====================================================================
#
# HOW TO USE:
#   sbatch scripts/slurm_train.sh
#
# MONITOR:
#   squeue -u $USER                      # check job status
#   tail -f logs/finagent_<JOB_ID>.out   # watch training output
#   scancel <JOB_ID>                     # cancel if needed
#
# FIRST TIME SETUP (run once on the login node BEFORE sbatch):
#   cd /sci/labs/arieljaffe/dan.abergel1
#   git clone https://github.com/DanAbergel/finagent-8b.git
#   cd finagent-8b
#   python3 -m venv /sci/labs/arieljaffe/dan.abergel1/finagent_env
#   source /sci/labs/arieljaffe/dan.abergel1/finagent_env/bin/activate
#   pip install unsloth
#   pip install --force-reinstall huggingface_hub==0.34.0
#   pip install wandb
#   # Create .env with your API keys (see README)
# =====================================================================

# ── SLURM resource request ─────────────────────────────────────────
#SBATCH --job-name=finagent-qlora
#SBATCH --gres=gpu:l40s:1        # 1x NVIDIA L40S (48 GB VRAM)
#SBATCH --cpus-per-task=8        # 8 CPU cores (for data loading workers)
#SBATCH --mem=64G                # 64 GB RAM (model + data + overhead)
#SBATCH --time=02:00:00          # 2h max (training takes ~15 min, rest is setup/saving)
#SBATCH --output=logs/finagent_%j.out   # stdout → logs/finagent_<JOB_ID>.out
#SBATCH --error=logs/finagent_%j.err    # stderr → logs/finagent_<JOB_ID>.err

set -euo pipefail

# ── Paths (adapt to your cluster) ──────────────────────────────────
LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/finagent/finagent-8b"
VENV_DIR="$LAB_DIR/repos/finagent/finagent_env"

echo "============================================================"
echo "  FinAgent QLoRA — SLURM Job"
echo "============================================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  Project:    $PROJECT_DIR"
echo "  Venv:       $VENV_DIR"
echo "============================================================"
echo ""

# ── 1. Activate virtual environment ────────────────────────────────
# The venv should already exist (created once on the login node).
# It contains unsloth, huggingface_hub==0.34.0, wandb, etc.
echo "[1/5] Activating venv ..."
source "$VENV_DIR/bin/activate"
echo "  Python: $(which python3)"
echo "  Version: $(python3 --version)"
echo ""

# ── 2. Update code from GitHub ─────────────────────────────────────
# Pull latest code so the training script matches what's on main.
echo "[2/5] Updating code ..."
cd "$PROJECT_DIR"
git fetch --all
git reset --hard origin/main
echo "  Commit: $(git rev-parse --short HEAD)"
echo "  Message: $(git log -1 --pretty=%s)"
echo ""

# ── 3. GPU check ───────────────────────────────────────────────────
# Verify we got a GPU. The Python script auto-detects GPU type
# and adjusts precision (bf16 vs fp16) accordingly.
echo "[3/5] GPU check ..."
nvidia-smi
echo ""

# ── 4. Load .env and run training ──────────────────────────────────
# .env contains WANDB_API_KEY and other secrets.
# The training script reads hyperparameters from CLI flags.
echo "[4/5] Starting training ..."
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"

# Load .env (exports WANDB_API_KEY, etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "  Loading .env ..."
    set -a
    source "$PROJECT_DIR/.env"
    set +a
else
    echo "  [!] No .env found — W&B and HF push won't work"
fi

# Launch training with W&B enabled
python3 -u "$PROJECT_DIR/scripts/train_qlora.py" \
    --output_dir "$PROJECT_DIR/results/finagent-checkpoints" \
    --adapter_path "$PROJECT_DIR/results/finagent-7b-lora" \
    --merged_path "$PROJECT_DIR/results/finagent-7b-merged" \
    --use_wandb

# ── 5. Done ─────────────────────────────────────────────────────────
echo ""
echo "[5/5] Done!"
echo "============================================================"
echo "  Job finished:  $(date)"
echo "  LoRA adapter:  $PROJECT_DIR/results/finagent-7b-lora/"
echo "  Merged model:  $PROJECT_DIR/results/finagent-7b-merged/"
echo ""
echo "  Next step: use the merged model in Module 3 (agentic tool-use)"
echo "============================================================"
