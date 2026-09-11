#!/usr/bin/env bash
# run_ablation.sh
# ================
# Downloads MedMNIST dataset files (native 224px release) and then runs the
# 8-config SPIRAL-ViT ablation (C1-C8, learned_dct in the C3/C5/C7/C8 slots)
# for each dataset given.
#
# Download-then-train, not interleaved: all requested datasets are fully
# downloaded first (download_medmnist_datasets.py), then training starts.
# medmnist's own download check is idempotent (skips a file that's already
# cached), so re-running this script after an interruption only fetches
# what's still missing and just resumes training.
#
# Both resume layers already live inside ablation_medmnist.py itself (no
# extra logic needed here):
#   - per-epoch "latest" checkpoint within a config
#   - a small state JSON tracking which configs are already completed per
#     dataset, skipped automatically on re-run
#
# num_workers=4 is safe here (this targets Linux) -- the shm.dll Windows
# DataLoader-worker issue this project's original Windows environment had
# doesn't apply.
#
# Run this inside tmux/screen (or with nohup) so it survives an SSH
# disconnect:
#   tmux new -s medmnist
#   ./run_ablation.sh
#   # detach: Ctrl-b d -- reattach later with: tmux attach -t medmnist
#
# Usage
# -----
#   ./run_ablation.sh
#   ./run_ablation.sh --datasets organa,organc,organs --epochs 100
#   ./run_ablation.sh --datasets organa --skip-download   # data already cached

set -euo pipefail

DATASETS="organa,organc,organs"
EPOCHS=100
DEVICE="auto"
SEED=42
DATA_ROOT="data/MedMnist_fullres"
OUT_DIR="results/ablation_medmnist"
CKPT_DIR="checkpoints/medmnist"
NATIVE_SIZE=224
NUM_WORKERS=4
SKIP_DOWNLOAD=0
MAX_RETRIES=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)      DATASETS="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --device)        DEVICE="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --data_root)     DATA_ROOT="$2"; shift 2 ;;
    --out_dir)       OUT_DIR="$2"; shift 2 ;;
    --ckpt_dir)      CKPT_DIR="$2"; shift 2 ;;
    --native_size)   NATIVE_SIZE="$2"; shift 2 ;;
    --num_workers)   NUM_WORKERS="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

export PYTHONIOENCODING="utf-8"

echo "======================================================"
echo "  MedMNIST Foveated ViT -- ablation run"
echo "======================================================"
echo "  Datasets    : $DATASETS"
echo "  Configs     : C1-C8 (learned_dct in C3/C5/C7/C8)"
echo "  Data root   : $DATA_ROOT  (native_size=$NATIVE_SIZE)"
echo "  Ckpt dir    : $CKPT_DIR"
echo "  Out dir     : $OUT_DIR"
echo "  Epochs      : $EPOCHS  |  Device: $DEVICE  |  Seed: $SEED  |  Workers: $NUM_WORKERS"
echo "======================================================"
echo ""

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  echo "------------------------------------------------------"
  echo "  Phase 1: Downloading all datasets (no training yet)"
  echo "------------------------------------------------------"
  python src/experiments/download_medmnist_datasets.py \
    --datasets "$DATASETS" --data_root "$DATA_ROOT" --native_size "$NATIVE_SIZE"
  echo ""
  echo "  [OK] All datasets downloaded and cached."
  echo ""
else
  echo "  [SKIP] --skip-download passed -- assuming all data is already cached."
  echo ""
fi

echo "------------------------------------------------------"
echo "  Phase 2: Training (8-config ablation per dataset)"
echo "------------------------------------------------------"
echo ""

IFS=',' read -ra DATASET_LIST <<< "$DATASETS"

for ds in "${DATASET_LIST[@]}"; do
  ds="$(echo "$ds" | xargs)"  # trim whitespace
  echo ""
  echo "------------------------------------------------------"
  echo "  Dataset: $ds"
  echo "------------------------------------------------------"

  attempt=0
  while true; do
    attempt=$((attempt + 1))
    set +e
    python src/experiments/ablation_medmnist.py \
      --dataset "$ds" --epochs "$EPOCHS" --data_root "$DATA_ROOT" \
      --out_dir "$OUT_DIR" --ckpt_dir "$CKPT_DIR" --device "$DEVICE" \
      --seed "$SEED" --num_workers "$NUM_WORKERS" --native_size "$NATIVE_SIZE"
    exit_code=$?
    set -e

    if [[ $exit_code -eq 0 ]]; then
      echo "  [OK] $ds completed."
      break
    fi

    echo "  [FAIL] $ds exited with code $exit_code (attempt $attempt/$MAX_RETRIES)."
    if [[ $attempt -ge $MAX_RETRIES ]]; then
      echo "  [ABORT] Max retries reached for $ds -- stopping."
      exit 1
    fi
    echo "  Retrying in 10s (already-completed configs are skipped automatically)..."
    sleep 10
  done
done

echo ""
echo "======================================================"
echo "  [OK] All requested datasets completed."
echo "  Checkpoints : $CKPT_DIR/"
echo "  Results     : $OUT_DIR/"
echo "======================================================"
