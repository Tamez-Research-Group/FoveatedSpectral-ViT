"""
src/experiments/ablation_medmnist.py
=====================================
2×2×2 factorial ablation study on the five MedMNIST benchmark datasets.

This script mirrors ``ablation_full.py`` exactly in its experimental design
(same 8 SPIRAL-VIT configurations, same VIT_ARCH) but targets the MedMNIST
datasets used by the supervisor's ViTDCT work:

    blood     — BloodMNIST,     8-class blood cell type (RGB)
    breast    — BreastMNIST,    2-class ultrasound malignancy (grayscale→GET)
    pneumonia — PneumoniaMNIST, 2-class chest X-ray (grayscale→GET)
    retina    — RetinaMNIST,    5-class retinal vascular grading (RGB)
    derma     — DermaMNIST,     7-class skin lesion (RGB)

Using the same datasets enables direct, fair comparison between SPIRAL-VIT
configurations and the supervisor's ViTDCT results for the comparison paper.

Training protocol (matches supervisor's ViTDCT for fair comparison)
--------------------------------------------------------------------
* **Stochastic MC validation**: validation is run ``n_val_reps`` times per
  epoch using the same stochastic ``RandomShiftAndNoise`` transform applied
  to the val loader, giving a Monte Carlo estimate of the expected loss under
  augmentation.  ``n_val_reps = max(1, n_train // (2 * n_val) + 1)``

* **Weighted monitoring loss**: ``wavgloss = 0.65 * val_loss + 0.35 * train_loss``
  This discounts the 35 % training component to make the criterion less
  sensitive to over-fit signals.  Patience and Top-K averaging are driven by
  ``wavgloss``, not raw val loss.

* **Top-K model averaging**: the ``SAVE_TOP_K`` best model states (by
  ``wavgloss``) are accumulated in a rolling ``deque``.  At the end of
  training — or when triggered by mid-patience — the in-memory states are
  averaged to produce the final model weights.  This acts as an implicit
  ensemble with negligible inference cost.

* **Mid-patience averaging**: when ``patience_counter == patience // 2``, the
  current top-K states are averaged and loaded into the model before training
  continues.  This often restores useful capacity after a stuck plateau.

* **1 % improvement threshold** (``wlos = 0.99``): a new minimum only resets
  patience when ``wavgloss < 0.99 * best_wavgloss``, preventing patience from
  being consumed by trivial noise improvements.

Experimental design
-------------------
Three binary axes yield 8 configurations (identical to ablation_full.py):

+-----------+--------------+------------+---------------+
| Config ID | token_order  | embed_type | gaussian_bias |
+===========+==============+============+===============+
| C1        | raster       | conv       | False         |  ← ViT baseline
| C2        | spiral       | conv       | False         |
| C3        | raster       | dct_fast   | False         |
| C4        | raster       | conv       | True          |
| C5        | spiral       | dct_fast   | False         |
| C6        | spiral       | conv       | True          |
| C7        | raster       | dct_fast   | True          |
| C8        | spiral       | dct_fast   | True          |  ← full SPIRAL-VIT
+-----------+--------------+------------+---------------+

Images are resized to 224×224; with grid_size=16 this gives patch_size=14,
matching the standard ViT-B/16 patch resolution.

Outputs
-------
* ``results/ablation_medmnist/ablation_results_<dataset>_<timestamp>.csv``
* ``results/ablation_medmnist/loss_curves_<dataset>_<timestamp>.csv``
* ``checkpoints/medmnist/vit_<dataset>_best_<embed>_<order>_<gauss>.pth``

Usage
-----
::

    # Single dataset
    python src/experiments/ablation_medmnist.py --dataset blood --epochs 30

    # Resume an interrupted run (reads checkpoint + state JSON automatically)
    python src/experiments/ablation_medmnist.py --dataset blood --epochs 30

    # Custom data cache location
    python src/experiments/ablation_medmnist.py --dataset derma \\
        --data_root D:/datasets/medmnist --epochs 30
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import sys
import gc
from collections import deque

# stdout/stderr default to the system ANSI codepage (not UTF-8) whenever this
# process isn't attached to a real interactive console -- piped, redirected to
# a file, or run under a watchdog wrapper. Every "->"/"..."/checkmark print in
# this file would otherwise raise UnicodeEncodeError the moment output isn't
# a live terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from datetime import datetime
from typing import Dict, Deque, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.medmnist_dataset import MEDMNIST_INFO, MULTI_LABEL_DATASETS, get_medmnist_loaders
from models.vit import ViT
from utils.training_multilabel import evaluate_multilabel, train_one_epoch_multilabel
from utils.training_tracked import evaluate, train_one_epoch


# ---------------------------------------------------------------------------
# Configurations (identical to ablation_full.py for cross-study consistency)
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: List[Dict] = [
    # Learned-DCT paper round: dct_fast replaced by learned_dct in the same
    # 4 slots it previously occupied (C3/C5/C7/C8) -- back to 8 configs
    # total, matching the original 2x2x2 factorial design. C1/C2/C4/C6
    # (conv) are unchanged.
    {"config_id": "C1", "token_order": "raster", "embed_type": "conv",        "gaussian_bias": False},
    {"config_id": "C2", "token_order": "spiral", "embed_type": "conv",        "gaussian_bias": False},
    {"config_id": "C3", "token_order": "raster", "embed_type": "learned_dct", "gaussian_bias": False},
    {"config_id": "C4", "token_order": "raster", "embed_type": "conv",        "gaussian_bias": True},
    {"config_id": "C5", "token_order": "spiral", "embed_type": "learned_dct", "gaussian_bias": False},
    {"config_id": "C6", "token_order": "spiral", "embed_type": "conv",        "gaussian_bias": True},
    {"config_id": "C7", "token_order": "raster", "embed_type": "learned_dct", "gaussian_bias": True},
    {"config_id": "C8", "token_order": "spiral", "embed_type": "learned_dct", "gaussian_bias": True},
]

# Per-dataset hyper-parameters.
# resize_to=224, grid_size=16 → patch_size=14 (standard ViT-B/16 resolution).
DATASET_HP: Dict[str, Dict] = {
    "blood": {
        "lr": 1e-4, "batch_size": 32, "num_classes": 8,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    "breast": {
        # Very small dataset (~546 train); small batch, higher patience.
        "lr": 1e-4, "batch_size": 8, "num_classes": 2,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 10, "multi_label": False,
    },
    "pneumonia": {
        "lr": 3e-4, "batch_size": 32, "num_classes": 2,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    "retina": {
        "lr": 1e-4, "batch_size": 16, "num_classes": 5,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    "derma": {
        "lr": 1e-4, "batch_size": 32, "num_classes": 7,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    # --- 7 new datasets, added for the full 12-dataset benchmark comparison ---
    "path": {
        # 89,996 train images — batch scaled up vs. the original 5 to keep
        # wall-clock reasonable at the same epoch budget.
        "lr": 1e-4, "batch_size": 64, "num_classes": 9,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    "chest": {
        # 78,468 train images. Multi-label: 14 independent, non-mutually-
        # exclusive disease flags per image (NIH ChestX-ray14). num_classes
        # here means "output dimensionality" (14), not mutually-exclusive
        # class count. See MULTI_LABEL_DATASETS / train_one_epoch_multilabel.
        "lr": 1e-4, "batch_size": 64, "num_classes": 14,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": True,
    },
    "oct": {
        # 97,477 train images — batch scaled up.
        "lr": 1e-4, "batch_size": 64, "num_classes": 4,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    "tissue": {
        # 165,466 train images — the largest dataset in this project by far;
        # batch scaled up the most.
        "lr": 1e-4, "batch_size": 128, "num_classes": 8,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    "organa": {
        "lr": 1e-4, "batch_size": 32, "num_classes": 11,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    "organc": {
        "lr": 1e-4, "batch_size": 32, "num_classes": 11,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
    "organs": {
        "lr": 1e-4, "batch_size": 32, "num_classes": 11,
        "grid_size": 16, "resize_to": 224, "weight_decay": 1e-4,
        "patience": 7, "multi_label": False,
    },
}

# Shared ViT architecture — identical to ablation_full.py for fair comparison.
VIT_ARCH = dict(
    dim=256, depth=6, heads=8, mlp_dim=512,
    channels=3, dropout=0.1, emb_dropout=0.1, freq_keep=16,
)

# Number of top model states kept in memory for final weight averaging.
SAVE_TOP_K = 5


# ---------------------------------------------------------------------------
# Training protocol helpers (ported from ViTDCT supervisor's protocol)
# ---------------------------------------------------------------------------

def _stochastic_val(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_reps: int,
    epoch: int = 0,
    desc: str = "val",
    multi_label: bool = False,
) -> Tuple[float, float, float]:
    """Run the validation loop ``n_reps`` times and return the averages.

    Because the val loader uses ``RandomShiftAndNoise`` with ``shuffle=True``,
    each pass yields a different perturbation of the val set, giving a Monte
    Carlo estimate of the expected loss under augmentation.

    Args:
        n_reps: Number of forward passes over the val set.
        multi_label: Dispatch to ``evaluate_multilabel`` instead of
            ``evaluate`` (ChestMNIST — independent per-label metrics rather
            than argmax/macro-F1).

    Returns:
        ``(avg_loss, avg_acc, avg_f1)``
    """
    eval_fn = evaluate_multilabel if multi_label else evaluate
    losses: List[float] = []
    accs:   List[float] = []
    f1s:    List[float] = []
    for rep in range(n_reps):
        rep_desc = desc if rep == 0 else ""
        loss, acc, f1, _ = eval_fn(
            model, val_loader, criterion, device, epoch, desc=rep_desc,
        )
        losses.append(loss)
        accs.append(acc)
        f1s.append(f1)
    return (
        sum(losses) / len(losses),
        sum(accs)   / len(accs),
        sum(f1s)    / len(f1s),
    )


def _average_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Element-wise mean of a list of model state dicts.

    Averages are computed in float32 regardless of the stored dtype so that
    the operation is numerically stable for half-precision weights.
    """
    avg: Dict[str, torch.Tensor] = {}
    n = len(state_dicts)
    for key in state_dicts[0]:
        avg[key] = sum(sd[key].float() for sd in state_dicts) / n
    return avg


# ---------------------------------------------------------------------------
# Checkpoint + log helpers
# ---------------------------------------------------------------------------

LOG_FIELDS = [
    "saved_at", "config_id", "checkpoint_name", "epoch",
    "val_f1", "val_acc", "val_loss", "wavgloss",
    "token_order", "embed_type", "gaussian_bias",
    "dataset", "total_epochs",
    "lr", "batch_size", "weight_decay", "patience",
    "scheduler", "n_val_reps", "save_top_k",
]


def _cpu_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Move a model state_dict to CPU to free GPU memory while keeping it alive."""
    return {k: v.cpu() for k, v in state_dict.items()}


def _safe_torch_save(payload: Dict, path: str) -> None:
    """Write checkpoint atomically: save to .tmp then rename, so a mid-write
    crash never leaves a corrupted file at the final path."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _write_info(path: str, payload: Dict) -> None:
    """Write a human-readable .info.json sidecar alongside a checkpoint.

    Contains epoch, timestamp, and metrics — readable without loading the .pth.
    """
    info = {
        "checkpoint":  os.path.basename(path),
        "saved_at":    payload.get("saved_at", ""),
        "epoch":       payload.get("epoch"),
        "dataset":     payload.get("cfg", {}).get("dataset", ""),
        "config_id":   payload.get("cfg", {}).get("config_id", ""),
        "embed_type":  payload.get("cfg", {}).get("embed_type", ""),
        "token_order": payload.get("cfg", {}).get("token_order", ""),
        "gaussian_bias": payload.get("cfg", {}).get("gaussian_bias", ""),
        "best_val_f1":   round(float(payload.get("best_val_f1",  0.0)), 6),
        "best_val_acc":  round(float(payload.get("best_val_acc",  0.0)), 6),
        "best_val_loss": round(float(payload.get("best_val_loss", 0.0)), 6),
        "best_wavgloss": round(float(payload.get("best_wavgloss", 0.0)), 6),
        "freq_keep":   payload.get("cfg", {}).get("freq_keep", ""),
        "patch_size":  payload.get("cfg", {}).get("patch_size", ""),
        "image_size":  payload.get("cfg", {}).get("image_size", ""),
    }
    info_path = path.replace(".pth", ".info.json")
    tmp = info_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, info_path)


def _ckpt_name(cfg: Dict, dataset: str) -> str:
    gauss = "gaussian" if cfg["gaussian_bias"] else "nogaussian"
    return f"vit_{dataset}_best_{cfg['embed_type']}_{cfg['token_order']}_{gauss}.pth"


def _latest_ckpt_name(cfg: Dict, dataset: str) -> str:
    gauss = "gaussian" if cfg["gaussian_bias"] else "nogaussian"
    return f"vit_{dataset}_latest_{cfg['embed_type']}_{cfg['token_order']}_{gauss}.pth"


def _log_checkpoint(
    log_path: str,
    cfg: Dict,
    hp: Dict,
    dataset: str,
    ckpt_name: str,
    epoch: int,
    val_f1: float,
    val_acc: float,
    val_loss: float,
    wavgloss: float,
    total_epochs: int,
    n_val_reps: int,
) -> None:
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "saved_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_id":       cfg["config_id"],
            "checkpoint_name": ckpt_name,
            "epoch":           epoch,
            "val_f1":          round(val_f1,   6),
            "val_acc":         round(val_acc,   6),
            "val_loss":        round(val_loss,  6),
            "wavgloss":        round(wavgloss,  6),
            "token_order":     cfg["token_order"],
            "embed_type":      cfg["embed_type"],
            "gaussian_bias":   cfg["gaussian_bias"],
            "dataset":         dataset,
            "total_epochs":    total_epochs,
            "lr":              hp.get("lr", ""),
            "batch_size":      hp.get("batch_size", ""),
            "weight_decay":    hp.get("weight_decay", ""),
            "patience":        hp.get("patience", ""),
            "scheduler":       "cosine_annealing",
            "n_val_reps":      n_val_reps,
            "save_top_k":      SAVE_TOP_K,
        })


# ---------------------------------------------------------------------------
# Resume / state persistence (mirrors ablation_full.py)
# ---------------------------------------------------------------------------

def _state_path(ckpt_dir: str, dataset: str) -> str:
    return os.path.join(ckpt_dir, f"ablation_medmnist_state_{dataset}.json")


def _fresh_state(dataset: str, total_epochs: int) -> Dict:
    return {
        "dataset":       dataset,
        "total_epochs":  total_epochs,
        "completed":     [],
        "epochs_done":   {},
        "best_metrics":  {},
        "summaries":     {},
        "curves":        {},
    }


def _try_load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            content = f.read().strip()
        if not content:
            raise ValueError("empty")
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None


def _infer_completed_from_disk(ckpt_dir: str, dataset: str) -> List[str]:
    """Infer completed configs from .info.json sidecars when state JSON is missing.

    A config is considered complete if its best checkpoint has an .info.json
    AND no corresponding latest (crash-recovery) checkpoint exists.
    """
    completed = []
    for cfg in ABLATION_CONFIGS:
        gauss    = "gaussian" if cfg["gaussian_bias"] else "nogaussian"
        best     = f"vit_{dataset}_best_{cfg['embed_type']}_{cfg['token_order']}_{gauss}.pth"
        latest   = f"vit_{dataset}_latest_{cfg['embed_type']}_{cfg['token_order']}_{gauss}.pth"
        info     = best.replace(".pth", ".info.json")
        has_info = os.path.exists(os.path.join(ckpt_dir, info))
        has_lat  = os.path.exists(os.path.join(ckpt_dir, latest))
        if has_info and not has_lat:
            completed.append(cfg["config_id"])
    return completed


def _load_state(path: str, dataset: str, total_epochs: int, ckpt_dir: str = "") -> Dict:
    bak = path + ".bak"
    state = _try_load_json(path)
    if state is None and os.path.exists(bak):
        print("[RESUME] Main state corrupt — restoring from backup.")
        state = _try_load_json(bak)
    if state is None:
        state = _fresh_state(dataset, total_epochs)
        # Fall back to disk inference when both state files are unusable.
        if ckpt_dir:
            inferred = _infer_completed_from_disk(ckpt_dir, dataset)
            if inferred:
                print(f"[RESUME] State file missing/corrupt — inferred from disk: {inferred}")
                state["completed"] = inferred
        return state
    if state.get("total_epochs") != total_epochs:
        print(
            f"[RESUME] --epochs changed "
            f"({state['total_epochs']} → {total_epochs}); resetting epoch counters."
        )
        state["total_epochs"] = total_epochs
        state["epochs_done"]  = {}
    print(f"[RESUME] Completed configs: {state['completed']}")
    return state


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_state(path: str, state: Dict) -> None:
    tmp = path + ".tmp"
    try:
        serialised = json.dumps(state, indent=2, cls=_NumpyEncoder)
    except Exception as exc:
        print(f"[WARN] State serialisation failed ({exc}); progress not saved.")
        return
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(serialised)
            f.flush()
            os.fsync(f.fileno())   # force OS buffer → disk before rename
        os.replace(tmp, path)
        shutil.copy2(path, path + ".bak")
    except Exception as exc:
        print(f"[WARN] State write failed ({exc}); progress not saved.")


# ---------------------------------------------------------------------------
# Per-configuration training — supervisor's protocol
# ---------------------------------------------------------------------------

def _build_ckpt_payload(
    model: nn.Module,
    cfg: Dict,
    hp: Dict,
    image_size: int,
    patch_size: int,
    epoch: int,
    best_val_f1: float,
    best_val_acc: float,
    best_val_loss: float,
    best_wavgloss: float,
    val_loss_curve: List[float],
    dataset: str,
    n_val_reps: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
    patience_counter: int = 0,
) -> Dict:
    """Assemble a checkpoint dict for both 'latest' and 'best' saves."""
    payload: Dict = {
        "state_dict": model.state_dict(),
        "cfg": {
            # Architecture
            "image_size":    image_size,
            "patch_size":    patch_size,
            "num_classes":   hp["num_classes"],
            "embed_type":    cfg["embed_type"],
            "token_order":   cfg["token_order"],
            "gaussian_bias": cfg["gaussian_bias"],
            "freq_keep":     VIT_ARCH["freq_keep"],
            "dataset":       dataset,
            "config_id":     cfg["config_id"],
            # Training settings — for reproducibility
            "lr":                     hp["lr"],
            "batch_size":             hp["batch_size"],
            "weight_decay":           hp["weight_decay"],
            "patience":               hp.get("patience", 7),
            "scheduler":              "cosine_annealing",
            "early_stopping_monitor": "wavgloss",
            "n_val_reps":             n_val_reps,
            "save_top_k":             SAVE_TOP_K,
        },
        "epoch":            epoch,
        "best_val_f1":      best_val_f1,
        "best_val_acc":     best_val_acc,
        "best_val_loss":    best_val_loss,
        "best_wavgloss":    best_wavgloss,
        "val_loss_curve":   val_loss_curve,
        "patience_counter": patience_counter,
        "saved_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    return payload


def train_config(
    cfg: Dict,
    dataset: str,
    hp: Dict,
    data_root: str,
    num_epochs: int,
    device: torch.device,
    seed: int,
    ckpt_dir: str,
    log_path: str,
    state: Dict,
    num_workers: int = 0,
    use_amp: bool = False,
    native_size: Optional[int] = None,
) -> Tuple[Dict, List[float]]:
    """Train one ablation configuration on a MedMNIST dataset.

    Implements the supervisor's ViTDCT training protocol for fair comparison:
    stochastic MC validation, weighted wavgloss monitoring, Top-K model
    averaging, and mid-patience weight averaging.

    Supports resume: if a 'latest' checkpoint exists for this config it is
    loaded and training continues from the saved epoch, with patience state
    and best metrics fully restored.

    Args:
        cfg:        Ablation config dict (config_id, token_order, embed_type,
                    gaussian_bias).
        dataset:    MedMNIST dataset key.
        hp:         Hyper-parameter dict from :data:`DATASET_HP`.
        data_root:  MedMNIST cache directory.
        num_epochs: Total training epochs.
        device:     Compute device.
        seed:       Random seed for reproducibility.
        ckpt_dir:   Checkpoint directory.
        log_path:   Running CSV checkpoint log path.
        state:      Mutable run-state dict (updated in place).
        num_workers: DataLoader workers (0 on Windows).
        use_amp:    Enable AMP (CUDA only).
        native_size: MedMNIST's own native resolution to download (28/64/128/224).
                    ``None`` preserves the original 28px+upsample baseline.

    Returns:
        ``(summary_dict, val_loss_curve)``
    """
    config_id   = cfg["config_id"]
    tag         = f"{dataset}/{config_id}"
    patience    = hp.get("patience", 7)
    multi_label = hp.get("multi_label", False)

    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ----- Data -----
    train_loader, val_loader, _, _ = get_medmnist_loaders(
        dataset_name=dataset,
        batch_size=hp["batch_size"],
        resize_to=hp["resize_to"],
        data_root=data_root,
        num_workers=num_workers,
        native_size=native_size,
    )

    # Number of stochastic val reps (MC validation, supervisor's formula).
    n_val_reps = max(
        1,
        len(train_loader.dataset) // (2 * len(val_loader.dataset)) + 1,
    )

    # Fixed-resolution: image_size and patch_size from config, not batch meta.
    image_size = hp["resize_to"]
    patch_size = hp["resize_to"] // hp["grid_size"]

    # ----- Model -----
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=hp["num_classes"],
        token_order=cfg["token_order"],
        embed_type=cfg["embed_type"],
        gaussian_bias=cfg["gaussian_bias"],
        **VIT_ARCH,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=hp["lr"] / 20
    )
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    # ----- Supervisor's protocol state -----
    top_models:     Deque[Dict[str, torch.Tensor]] = deque(maxlen=SAVE_TOP_K)
    mid_patience    = patience // 2
    patience_counter = 0
    addNextToTop    = False
    last_state: Optional[Dict[str, torch.Tensor]] = None
    best_wavgloss   = float("inf")
    wlos            = 0.99          # 1 % improvement threshold

    best_val_f1   = 0.0
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    val_loss_curve: List[float] = []
    start_epoch = 1
    ckpt_path   = os.path.join(ckpt_dir, _ckpt_name(cfg, dataset))
    latest_path = os.path.join(ckpt_dir, _latest_ckpt_name(cfg, dataset))

    # ----- Resume -----
    if os.path.exists(latest_path):
        print(f"[RESUME] Loading {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_f1      = ckpt.get("best_val_f1",      0.0)
        best_val_acc     = ckpt.get("best_val_acc",      0.0)
        best_val_loss    = ckpt.get("best_val_loss",     float("inf"))
        best_wavgloss    = ckpt.get("best_wavgloss",     float("inf"))
        patience_counter = ckpt.get("patience_counter",  0)
        val_loss_curve   = ckpt.get("val_loss_curve",    [])
        if scaler and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(
            f"[RESUME] Epoch {start_epoch}  "
            f"best_wavgloss={best_wavgloss:.4f}  "
            f"patience={patience_counter}/{patience}"
        )

    # ----- Header -----
    print(f"\n{'='*64}")
    print(
        f"  {tag}  |  {cfg['embed_type']}  {cfg['token_order']}  "
        f"gauss={cfg['gaussian_bias']}"
    )
    print(
        f"  image={image_size}px  patch={patch_size}px  "
        f"classes={hp['num_classes']}  lr={hp['lr']}  "
        f"val_reps={n_val_reps}  top_k={SAVE_TOP_K}"
    )
    print(f"{'='*64}")

    # ----- Training loop -----
    for epoch in range(start_epoch, num_epochs + 1):

        # Capture pre-training state from the epoch AFTER the last improvement
        # (supervisor: addNextToTop saves the epoch immediately following a minimum).
        if addNextToTop and last_state is not None:
            top_models.append(_cpu_state(last_state))
            addNextToTop = False

        # Train one epoch.
        train_epoch_fn = train_one_epoch_multilabel if multi_label else train_one_epoch
        train_loss, _, _, _ = train_epoch_fn(
            model, train_loader, criterion, optimizer, device, epoch,
            desc=tag, scaler=scaler,
        )

        # Stochastic MC validation (n_val_reps passes, same augmentation).
        avg_val_loss, avg_val_acc, avg_val_f1 = _stochastic_val(
            model, val_loader, criterion, device, n_val_reps,
            epoch=epoch, desc=tag, multi_label=multi_label,
        )

        scheduler.step()

        # Save model state before any possible in-place replacement.
        last_state = _cpu_state(model.state_dict())

        # Weighted monitoring criterion (supervisor's formula).
        wavgloss = 0.65 * avg_val_loss + 0.35 * train_loss
        val_loss_curve.append(avg_val_loss)

        # Significant improvement check (1 % threshold).
        if wavgloss < wlos * best_wavgloss:
            best_wavgloss    = wavgloss
            patience_counter = 0
            addNextToTop     = True

            top_models.append(_cpu_state(model.state_dict()))

            # Update best-reported metrics at each new wavgloss minimum.
            best_val_f1   = avg_val_f1
            best_val_acc  = avg_val_acc
            best_val_loss = avg_val_loss

            # Save best checkpoint immediately (overwritten at each improvement).
            _best_payload = _build_ckpt_payload(
                model, cfg, hp, image_size, patch_size,
                epoch, best_val_f1, best_val_acc, best_val_loss,
                best_wavgloss, val_loss_curve, dataset, n_val_reps,
            )
            _safe_torch_save(_best_payload, ckpt_path)
            _write_info(ckpt_path, _best_payload)
            _log_checkpoint(
                log_path, cfg, hp, dataset, _ckpt_name(cfg, dataset),
                epoch, best_val_f1, best_val_acc, best_val_loss,
                best_wavgloss, num_epochs, n_val_reps,
            )
            print(
                f"  ✓ New best  wavg={best_wavgloss:.4f}  "
                f"F1={best_val_f1:.4f}  acc={best_val_acc:.4f}  "
                f"[epoch {epoch}]"
            )

        else:
            patience_counter += 1

            # Mid-patience averaging: average accumulated top-K states and
            # reload into the model to escape a local plateau.
            if patience_counter == mid_patience and top_models:
                avg_state = _average_state_dicts(list(top_models))
                model.load_state_dict(avg_state)
                last_state = _cpu_state(avg_state)
                print(
                    f"  Mid-patience avg at epoch {epoch} "
                    f"({patience_counter}/{patience}, "
                    f"K={len(top_models)})"
                )

            if patience_counter >= patience:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(patience={patience})"
                )
                break

        # Latest checkpoint (crash recovery): includes patience state.
        _safe_torch_save(
            _build_ckpt_payload(
                model, cfg, hp, image_size, patch_size,
                epoch, best_val_f1, best_val_acc, best_val_loss,
                best_wavgloss, val_loss_curve, dataset, n_val_reps,
                optimizer=optimizer, scaler=scaler,
                patience_counter=patience_counter,
            ),
            latest_path,
        )

        # Update in-flight state (written to JSON after each config).
        state["epochs_done"][config_id] = epoch
        state["best_metrics"][config_id] = {
            "val_f1":    best_val_f1,
            "val_acc":   best_val_acc,
            "val_loss":  best_val_loss,
            "wavgloss":  best_wavgloss,
        }

    # ----- Final Top-K weight averaging -----
    # Average all accumulated top-K checkpoints into the final model.
    # Overwrites the best checkpoint with the averaged weights.
    if top_models:
        print(f"  Final Top-{len(top_models)} weight averaging …")
        avg_state = _average_state_dicts(list(top_models))
        model.load_state_dict(avg_state)
        _final_payload = _build_ckpt_payload(
            model, cfg, hp, image_size, patch_size,
            -1,              # epoch=-1 signals "averaged final model"
            best_val_f1, best_val_acc, best_val_loss,
            best_wavgloss, val_loss_curve, dataset, n_val_reps,
        )
        _safe_torch_save(_final_payload, ckpt_path)
        _write_info(ckpt_path, _final_payload)

    # Clean up latest checkpoint (crash-recovery file no longer needed).
    for _f in [latest_path, latest_path + ".tmp"]:
        if os.path.exists(_f):
            os.remove(_f)

    summary = {
        "config_id":               config_id,
        "token_order":             cfg["token_order"],
        "embed_type":              cfg["embed_type"],
        "gaussian_bias":           cfg["gaussian_bias"],
        "val_f1":                  best_val_f1,
        "val_acc":                 best_val_acc,
        "val_loss":                best_val_loss,
        "wavgloss":                best_wavgloss,
        "dataset":                 dataset,
        "epochs":                  num_epochs,
        "lr":                      hp["lr"],
        "batch_size":              hp["batch_size"],
        "weight_decay":            hp["weight_decay"],
        "patience":                patience,
        "scheduler":               "cosine_annealing",
        "early_stopping_monitor":  "wavgloss",
        "n_val_reps":              n_val_reps,
        "save_top_k":              SAVE_TOP_K,
    }
    return summary, val_loss_curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPIRAL-VIT 2×2×2 ablation on MedMNIST datasets."
    )
    parser.add_argument(
        "--dataset", default="blood",
        choices=list(MEDMNIST_INFO.keys()),
        help="MedMNIST dataset to train on.",
    )
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--data_root", default="data/medmnist",
                        help="MedMNIST download/cache directory.")
    parser.add_argument("--out_dir",   default="results/ablation_medmnist")
    parser.add_argument("--ckpt_dir",  default=os.path.join("checkpoints", "medmnist"),
                        help="Checkpoint directory. Defaults to checkpoints/medmnist "
                             "(28px-upsampled baseline experiment).")
    parser.add_argument("--native_size", type=int, default=None,
                        choices=[28, 64, 128, 224],
                        help="MedMNIST's own native resolution to download (a separate, "
                             "independently-curated release, not a resize of the 28px "
                             "default). None (default) preserves the existing 28px+upsample "
                             "baseline; pass 224 to match the advisor's ViTDCT protocol.")
    parser.add_argument("--device",    default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="DataLoader worker processes. Default (None) uses 4 -- "
             "measured ~4.75x data-loading throughput vs. 0 on this "
             "machine. get_medmnist_loaders() sets persistent_workers=True "
             "so workers spawn once per config rather than every epoch, "
             "mitigating the Windows shm.dll respawn race (WinError 1114). "
             "Pass 0 to fall back to single-process loading if workers "
             "still crash.",
    )
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated subset of config IDs to run (e.g. 'C1,C6,C8'). "
             "Default: all 8.",
    )
    args = parser.parse_args()

    dataset = args.dataset
    hp      = DATASET_HP[dataset]

    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",")}
        run_configs = [c for c in ABLATION_CONFIGS if c["config_id"] in wanted]
        unknown = wanted - {c["config_id"] for c in run_configs}
        if unknown:
            raise ValueError(f"Unknown config ID(s): {sorted(unknown)}. Choose from: "
                              f"{[c['config_id'] for c in ABLATION_CONFIGS]}")
    else:
        run_configs = ABLATION_CONFIGS

    # ----- Device -----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    use_amp = device.type == "cuda"
    nw      = args.num_workers if args.num_workers is not None else 4

    # ----- Directories -----
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"ablation_results_{dataset}_{ts}.csv")
    crv_path = os.path.join(args.out_dir, f"loss_curves_{dataset}_{ts}.csv")
    log_path = os.path.join(args.out_dir, f"checkpoint_log_{dataset}_{ts}.csv")

    # ----- State (resume support) -----
    state_path = _state_path(ckpt_dir, dataset)
    state      = _load_state(state_path, dataset, args.epochs, ckpt_dir=ckpt_dir)

    print(f"\nDataset  : {dataset}  ({hp['num_classes']} classes)")
    print(f"Resize   : {hp['resize_to']}px  →  patch_size={hp['resize_to']//hp['grid_size']}px")
    print(f"Epochs   : {args.epochs}  |  LR: {hp['lr']}  |  BS: {hp['batch_size']}")
    print(f"Protocol : wavgloss monitoring  Top-K={SAVE_TOP_K}  mid-patience averaging")
    print(f"Configs  : {[c['config_id'] for c in run_configs]}  |  Seed: {args.seed}")
    print(f"Out dir  : {args.out_dir}\n")

    all_summaries: List[Dict]          = list(state.get("summaries", {}).values())
    all_curves:    Dict[str, List[float]] = dict(state.get("curves",    {}))

    for cfg in run_configs:
        config_id = cfg["config_id"]

        if config_id in state["completed"]:
            print(f"[SKIP] {config_id} already completed — skipping.")
            continue

        summary, curve = train_config(
            cfg=cfg,
            dataset=dataset,
            hp=hp,
            data_root=args.data_root,
            num_epochs=args.epochs,
            device=device,
            seed=args.seed,
            ckpt_dir=ckpt_dir,
            log_path=log_path,
            state=state,
            num_workers=nw,
            use_amp=use_amp,
            native_size=args.native_size,
        )

        all_summaries.append(summary)
        all_curves[config_id] = curve
        state["completed"].append(config_id)
        state["summaries"][config_id] = summary

        # Free GPU memory between configs.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state["curves"][config_id]    = curve
        _save_state(state_path, state)

    # ----- Write results CSV -----
    if all_summaries:
        result_fields = [
            "config_id", "token_order", "embed_type", "gaussian_bias",
            "val_acc", "val_f1", "val_loss", "wavgloss",
            "dataset", "epochs",
            "lr", "batch_size", "weight_decay", "patience",
            "scheduler", "early_stopping_monitor", "n_val_reps", "save_top_k",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"\nResults  → {csv_path}")

    # ----- Write loss-curve CSV -----
    if all_curves:
        max_len = max(len(v) for v in all_curves.values())
        with open(crv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + list(all_curves.keys()))
            for i in range(max_len):
                row = [i + 1] + [
                    round(all_curves[k][i], 6) if i < len(all_curves[k]) else ""
                    for k in all_curves
                ]
                writer.writerow(row)
        print(f"Curves   → {crv_path}")

    # ----- Summary table -----
    print(f"\n{'─'*76}")
    print(
        f"{'Config':<8} {'embed':<10} {'order':<8} {'gauss':<7} "
        f"{'F1':>7} {'Acc':>7} {'wavgloss':>10}"
    )
    print(f"{'─'*76}")
    for s in sorted(all_summaries, key=lambda x: int(x["config_id"][1:])):
        gauss = "✓" if s["gaussian_bias"] else "–"
        wl    = s.get("wavgloss", float("nan"))
        print(
            f"{s['config_id']:<8} {s['embed_type']:<10} {s['token_order']:<8} "
            f"{gauss:<7} {s['val_f1']:>7.4f} {s['val_acc']:>7.4f} {wl:>10.4f}"
        )
    print(f"{'─'*76}\n")

    # ----- Checkpoint manifest -----
    kept = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith(f"vit_{dataset}_best_") and f.endswith(".pth")
    ])
    stale = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith(f"vit_{dataset}_latest_")
    ]
    if stale:
        print(f"[CLEANUP] Removing {len(stale)} stale latest-checkpoint(s):")
        for f in stale:
            fp = os.path.join(ckpt_dir, f)
            os.remove(fp)
            print(f"  deleted {f}")
    print(f"[MANIFEST] Kept checkpoints for {dataset}:")
    for f in kept:
        info_path = os.path.join(ckpt_dir, f.replace(".pth", ".info.json"))
        if os.path.exists(info_path):
            with open(info_path) as fh:
                info = json.load(fh)
            print(
                f"  {f}\n"
                f"    epoch={info.get('epoch')}  saved_at={info.get('saved_at')}"
                f"  F1={info.get('best_val_f1'):.4f}"
            )
        else:
            print(f"  {f}")


if __name__ == "__main__":
    main()
