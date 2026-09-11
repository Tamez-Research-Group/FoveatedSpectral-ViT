"""
src/utils/training.py
=====================
Generic training and evaluation loops for the Spiral ViT.

**Metadata-aware batch format**
When full-resolution dynamic patching is enabled, each DataLoader batch is a
3-tuple ``(images, labels, meta)`` rather than the standard 2-tuple
``(images, labels)``.  The ``meta`` dictionary carries ``image_size``,
``patch_size``, and ``grid_size`` derived from the first sample in the batch
(all images in a batch share the same patch size under the current collation
strategy).  Both ``train_one_epoch`` and ``evaluate`` detect the format by
checking ``len(batch)`` and unpack accordingly; ``meta`` is consumed by the
training script to initialise the model on the first batch and is silently
discarded here.

**Checkpoint selection strategy**
Models are saved when *macro F1-score* improves on the validation set rather
than accuracy.  Macro F1 averages per-class F1 with equal weight, making it
sensitive to minority-class performance — critical for medical datasets where
the positive class (disease) is typically underrepresented.
"""

from typing import Tuple

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


class EarlyStopping:
    """Stop training when a monitored metric has not improved for ``patience`` epochs.

    Args:
        patience:  Epochs to wait after the last improvement before stopping.
        min_delta: Minimum absolute change to count as an improvement.
        mode:      ``"max"`` (higher is better, e.g. F1) or
                   ``"min"`` (lower is better, e.g. val loss).
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        # best_f1 kept for backward compatibility with ablation_full.py
        self.best_f1: float  = -1.0
        self.best: float     = -float("inf") if mode == "max" else float("inf")
        self.counter: int    = 0
        self.should_stop: bool = False

    def step(self, metric: float) -> bool:
        """Update state with the latest monitored metric.

        Args:
            metric: Val F1 when ``mode="max"``; val loss when ``mode="min"``.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best    = metric
            self.best_f1 = metric   # keep in sync for backward compat
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    """Run one full training epoch.

    Args:
        model: The ViT model to train (must already be on ``device``).
        dataloader: Training DataLoader.  Supports both 2-tuple
            ``(images, labels)`` and 3-tuple ``(images, labels, meta)``
            batch formats.
        criterion: Loss function (typically ``CrossEntropyLoss``).
        optimizer: Parameter optimiser (typically ``AdamW``).
        device: Target compute device.
        epoch: Current epoch number — used only for the tqdm bar label.

    Returns:
        A 3-tuple ``(avg_loss, accuracy, macro_f1)`` where:
          - ``avg_loss``: Mean per-sample loss over the epoch.
          - ``accuracy``: Fraction of correctly classified samples.
          - ``macro_f1``: Macro-averaged F1-score across all classes.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        imgs, labels = batch[0], batch[1]
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader.dataset)
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    """Run one full evaluation pass without gradient computation.

    Args:
        model: The ViT model to evaluate.
        dataloader: Validation or test DataLoader.  Supports the same
            2-tuple and 3-tuple batch formats as :func:`train_one_epoch`.
        criterion: Loss function.
        device: Target compute device.
        epoch: Current epoch number — used only for the tqdm bar label.

    Returns:
        A 3-tuple ``(avg_loss, accuracy, macro_f1)`` with the same semantics
        as :func:`train_one_epoch`.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]", leave=False)

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

    avg_loss = total_loss / len(dataloader.dataset)
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1
