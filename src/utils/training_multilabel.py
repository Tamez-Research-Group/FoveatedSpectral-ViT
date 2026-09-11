"""
src/utils/training_multilabel.py
=================================
Multi-label variants of the tracked training and evaluation loops, for
datasets where each sample carries several independent, non-mutually-
exclusive binary labels (currently: ChestMNIST — 14 chest-disease flags per
image, any combination of which may be simultaneously present).

Mirrors ``src/utils/training_tracked.py``'s exact
``(avg_loss, accuracy, macro_f1, metrics_dict)`` signature and return shape
so callers (``ablation_medmnist.py``'s wavgloss/Top-K/checkpoint machinery)
need no changes beyond dispatching to these functions instead — only the
loss/prediction/metric math differs:

* ``criterion`` is expected to be ``nn.BCEWithLogitsLoss()``, not
  ``CrossEntropyLoss`` — labels are float multi-hot vectors of shape
  ``(B, num_labels)``, not integer class indices.
* Predictions are per-label sigmoid + 0.5 threshold, not softmax + argmax.
* "Accuracy" is mean per-label accuracy (elementwise match over the full
  ``(N, num_labels)`` array) — matches MedMNIST's own official multi-label
  evaluation protocol (per-label ``accuracy_score``, then averaged).
* "F1" is macro-F1 computed by ``sklearn.metrics.f1_score`` directly on the
  2-D multilabel-indicator arrays (per-label F1, then averaged) — sklearn
  supports this natively with ``average="macro"``, no custom aggregation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


EpochMetrics = Dict[str, float]


def train_one_epoch_multilabel(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    desc: str = "",
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float, float, EpochMetrics]:
    """Run one full multi-label training epoch and return per-epoch metrics.

    Args:
        model: ViT model (already on ``device``); output dim = num_labels.
        dataloader: Training DataLoader yielding ``(images, labels)`` where
            ``labels`` is a float ``(B, num_labels)`` multi-hot tensor.
        criterion: Loss function — ``nn.BCEWithLogitsLoss()``.
        optimizer: Parameter optimiser (typically ``AdamW``).
        device: Target compute device.
        epoch: Current epoch index — used in the tqdm progress bar.
        desc: Optional label prefix for the tqdm bar.

    Returns:
        A 4-tuple ``(avg_loss, accuracy, macro_f1, metrics)`` where
        ``metrics`` has keys ``train_loss``, ``train_acc``, ``train_f1``.
    """
    model.train()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    bar_label = f"{desc} " if desc else ""
    pbar = tqdm(dataloader, desc=f"{bar_label}Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        imgs, labels = batch[0], batch[1]
        imgs, labels = imgs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                l1_loss = getattr(model.patch_embed, "l1_loss", None)
                if l1_loss is not None:
                    loss = loss + l1_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            l1_loss = getattr(model.patch_embed, "l1_loss", None)
            if l1_loss is not None:
                loss = loss + l1_loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore[arg-type]
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    acc = float((y_pred == y_true).mean())
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics: EpochMetrics = {
        "train_loss": avg_loss,
        "train_acc": acc,
        "train_f1": f1,
    }
    return avg_loss, acc, f1, metrics


@torch.no_grad()
def evaluate_multilabel(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    desc: str = "",
) -> Tuple[float, float, float, EpochMetrics]:
    """Run one full multi-label evaluation pass and return per-epoch metrics.

    Args:
        model: ViT model to evaluate.
        dataloader: Validation or test DataLoader, same multi-hot label
            format as :func:`train_one_epoch_multilabel`.
        criterion: Loss function — ``nn.BCEWithLogitsLoss()``.
        device: Target compute device.
        epoch: Current epoch index — used in the tqdm progress bar.
        desc: Optional label prefix for the tqdm bar.

    Returns:
        A 4-tuple ``(avg_loss, accuracy, macro_f1, metrics)`` where
        ``metrics`` has keys ``val_loss``, ``val_acc``, ``val_f1``.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    bar_label = f"{desc} " if desc else ""
    pbar = tqdm(dataloader, desc=f"{bar_label}Epoch {epoch} [Valid]", leave=False)

    for batch in pbar:
        imgs, labels = batch[0], batch[1]
        imgs, labels = imgs.to(device), labels.to(device).float()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore[arg-type]
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    acc = float((y_pred == y_true).mean())
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics: EpochMetrics = {
        "val_loss": avg_loss,
        "val_acc": acc,
        "val_f1": f1,
    }
    return avg_loss, acc, f1, metrics
