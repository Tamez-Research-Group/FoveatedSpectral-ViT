"""
src/utils/training_tracked.py
==============================
Metric-tracked variants of the standard training and evaluation loops.

Addresses **Reviewer Concern 3**: to plot learning curves demonstrating that
accuracy gains are not an artefact of undertraining the baseline, per-epoch
metrics (loss, accuracy, macro-F1) must be captured for all model variants.

This module mirrors ``src/utils/training.py`` exactly, but each function
returns a fourth value — an ``EpochMetrics`` dict — that callers can append
to a growing history list.  The underlying training logic is unchanged; no
existing script imports from this module.

Differences from training.py
------------------------------
* ``train_one_epoch`` returns ``(loss, acc, f1, metrics_dict)``.
* ``evaluate``         returns ``(loss, acc, f1, metrics_dict)``.
* Both accept an optional ``desc`` kwarg for the tqdm bar label override.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


EpochMetrics = Dict[str, float]


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    desc: str = "",
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float, float, EpochMetrics]:
    """Run one full training epoch and return per-epoch metrics.

    Identical to ``training.train_one_epoch`` except the return value
    is extended with a metrics dict for history accumulation.

    Args:
        model: ViT model (already on ``device``).
        dataloader: Training DataLoader.  Supports 2-tuple
            ``(images, labels)`` and 3-tuple ``(images, labels, meta)``
            batch formats.
        criterion: Loss function (typically ``CrossEntropyLoss``).
        optimizer: Parameter optimiser (typically ``AdamW``).
        device: Target compute device.
        epoch: Current epoch index — used in the tqdm progress bar.
        desc: Optional label prefix for the tqdm bar (e.g. ``"Spiral+DCT"``).

    Returns:
        A 4-tuple ``(avg_loss, accuracy, macro_f1, metrics)`` where:

        * ``avg_loss``:  mean per-sample loss over the epoch.
        * ``accuracy``:  fraction of correctly classified samples.
        * ``macro_f1``:  macro-averaged F1-score.
        * ``metrics``:   dict with keys ``train_loss``, ``train_acc``,
          ``train_f1`` (mirrors the other three values for easy merging).
    """
    model.train()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    bar_label = f"{desc} " if desc else ""
    pbar = tqdm(dataloader, desc=f"{bar_label}Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        imgs, labels = batch[0], batch[1]
        imgs, labels = imgs.to(device), labels.to(device)

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
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore[arg-type]
    acc = (
        (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    )
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    metrics: EpochMetrics = {
        "train_loss": avg_loss,
        "train_acc": acc,
        "train_f1": f1,
    }
    return avg_loss, acc, f1, metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    desc: str = "",
) -> Tuple[float, float, float, EpochMetrics]:
    """Run one full evaluation pass and return per-epoch metrics.

    Identical to ``training.evaluate`` except the return value is extended
    with a metrics dict for history accumulation.

    Args:
        model: ViT model to evaluate.
        dataloader: Validation or test DataLoader.  Supports the same
            2-tuple and 3-tuple batch formats as :func:`train_one_epoch`.
        criterion: Loss function.
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
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore[arg-type]
    acc = (
        (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    )
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    metrics: EpochMetrics = {
        "val_loss": avg_loss,
        "val_acc": acc,
        "val_f1": f1,
    }
    return avg_loss, acc, f1, metrics
