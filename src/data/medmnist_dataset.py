"""
src/data/medmnist_dataset.py
============================
MedMNIST dataset loaders for the SPIRAL-VIT ablation study.

Supports five datasets from the MedMNIST v2 benchmark that match the
datasets used by the supervisor's ViTDCT architecture:

    blood     — BloodMNIST,     8-class blood cell type classification (RGB)
    breast    — BreastMNIST,    2-class ultrasound malignancy (grayscale)
    pneumonia — PneumoniaMNIST, 2-class X-ray pneumonia detection (grayscale)
    retina    — RetinaMNIST,    5-class retinal vascular grading (RGB)
    derma     — DermaMNIST,     7-class skin lesion classification (RGB)

Preprocessing (matches ViTDCT supervisor's protocol for fair comparison)
------------------------------------------------------------------------
* Grayscale datasets (breast, pneumonia) use ``GrayToGET`` — a
  morphology-based transform that produces three semantically distinct
  channels (Gray, Edges, Texture) instead of naively replicating the
  grey channel three times.  DCT coefficients on edge and texture channels
  carry different frequency content than raw luminance.

* ``RandomShiftAndNoise`` (frequency-preserving augmentation) is applied
  to **both train and val** splits.  This makes validation stochastic:
  repeated passes through the val loader yield differently perturbed
  images, giving a Monte Carlo estimate of the expected loss under
  augmentation — a more stable training signal, especially for small val
  sets like BreastMNIST (78 samples).

* The test loader uses a clean deterministic transform (no augmentation)
  for final evaluation with ``evaluate_model_with_ci()``.

Requirements
------------
    pip install medmnist>=0.4.3
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

try:
    import medmnist
    from medmnist import (
        BloodMNIST,
        BreastMNIST,
        ChestMNIST,
        DermaMNIST,
        INFO,
        OCTMNIST,
        OrganAMNIST,
        OrganCMNIST,
        OrganSMNIST,
        PathMNIST,
        PneumoniaMNIST,
        RetinaMNIST,
        TissueMNIST,
    )
except ImportError as exc:
    raise ImportError(
        "medmnist is required for MedMNIST experiments.\n"
        "Install it with:  pip install medmnist>=0.4.3"
    ) from exc


# ---------------------------------------------------------------------------
# Transforms — ported from ViTDCT/utils.py (supervisor's codebase)
# ---------------------------------------------------------------------------

class GrayToGET:
    """Convert a single-channel grayscale tensor to a 3-channel GET tensor.

    Produces three semantically distinct channels from a (1, H, W) input:
      - Channel 0: Gray  — original luminance
      - Channel 1: Edges — morphological gradient (dilation − erosion),
                           squared, smoothed, and normalised to [0, 1]
      - Channel 2: Texture — erosion-subtracted residual, clamped,
                             square-rooted, and smoothed

    If the input already has 3 channels it is returned unchanged, so this
    transform is safe to apply universally (RGB images pass through).

    Ported verbatim from ``ViTDCT/utils.py::Gray_to_GET()``.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.shape[0] != 1:
            return img

        # Morphological dilation and erosion via max/min pooling.
        dilated = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
        eroded  = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

        edges   = dilated - eroded          # morphological gradient

        texture = img - eroded
        median  = torch.median(texture)
        mean_v  = torch.mean(texture)
        ethr    = 1.25 * median + 0.25 * mean_v
        texture = torch.clamp(texture, max=ethr)

        # Smooth and normalise edges.
        edges = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)
        edges = edges * edges
        emin  = edges.min()
        edges = F.avg_pool2d(edges, kernel_size=3, stride=1, padding=1)
        edges = F.avg_pool2d(edges, kernel_size=3, stride=1, padding=1)
        edges = (edges - emin) / (edges.max() - emin + 1e-8)

        # Smooth texture.
        texture = -F.max_pool2d(-texture, kernel_size=3, stride=1, padding=1)
        texture = torch.sqrt(texture)
        texture = F.avg_pool2d(texture, kernel_size=3, stride=1, padding=1)
        texture = F.avg_pool2d(texture, kernel_size=3, stride=1, padding=1)
        texture = F.avg_pool2d(texture, kernel_size=3, stride=1, padding=1)
        texture = texture / (ethr + 1e-8)

        return torch.cat([img, edges, texture], dim=0)   # (3, H, W)


class RandomShiftAndNoise:
    """Frequency-preserving stochastic augmentation.

    Applies a random spatial shift (border-replicated) and additive
    Gaussian noise.  Does NOT use flips or rotations, which would corrupt
    the phase structure of DCT coefficients.

    Applied to **both train and val** loaders so that repeated passes
    through the val set give a Monte Carlo estimate of the expected loss
    under augmentation (matching the supervisor's ViTDCT training protocol).

    Ported verbatim from ``ViTDCT/utils.py::RandomShiftAndNoise()``.

    Args:
        max_shift: Maximum pixel shift in any direction.
        noise_std: Standard deviation of additive Gaussian noise (operates
            on normalised tensors, so 0.015 ≈ 1.5 % of the normalised range).
        p_shift: Probability of applying the spatial shift.
        p_noise: Probability of applying Gaussian noise.
    """

    def __init__(
        self,
        max_shift: int   = 3,
        noise_std: float = 0.015,
        p_shift:   float = 0.75,
        p_noise:   float = 0.95,
    ) -> None:
        self.max_shift = max_shift
        self.noise_std = noise_std
        self.p_shift   = p_shift
        self.p_noise   = p_noise

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_shift:
            sy = random.randint(-self.max_shift, self.max_shift)
            sx = random.randint(-self.max_shift, self.max_shift)
            if sy != 0 or sx != 0:
                img = torch.roll(img, shifts=(sy, sx), dims=(1, 2))
                if sy > 0:
                    img[:, :sy,  :] = img[:, sy:sy+1,   :]
                elif sy < 0:
                    img[:, sy:,  :] = img[:, sy-1:sy,   :]
                if sx > 0:
                    img[:, :, :sx]  = img[:, :, sx:sx+1]
                elif sx < 0:
                    img[:, :, sx:]  = img[:, :, sx-1:sx]

        if random.random() < self.p_noise:
            img = img + torch.randn_like(img) * self.noise_std

        return img


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

#: Maps the short dataset key to (medmnist_class, is_rgb).
MEDMNIST_INFO = {
    "blood":     (BloodMNIST,     True),
    "breast":    (BreastMNIST,    False),
    "pneumonia": (PneumoniaMNIST, False),
    "retina":    (RetinaMNIST,    True),
    "derma":     (DermaMNIST,     True),
    "path":      (PathMNIST,      True),
    "chest":     (ChestMNIST,     False),
    "oct":       (OCTMNIST,       False),
    "tissue":    (TissueMNIST,    False),
    "organa":    (OrganAMNIST,    False),
    "organc":    (OrganCMNIST,    False),
    "organs":    (OrganSMNIST,    False),
}

#: Datasets where class imbalance justifies WeightedRandomSampler.
_WEIGHTED_DATASETS = {"breast", "pneumonia", "derma", "retina"}

#: Datasets where each sample carries multiple independent, non-mutually-
#: exclusive binary labels (e.g. ChestMNIST's 14 disease flags) rather than
#: one mutually-exclusive class. Drives BCEWithLogitsLoss + sigmoid throughout
#: instead of CrossEntropyLoss + argmax/softmax.
MULTI_LABEL_DATASETS = {"chest"}


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _MedMNISTWrapper(Dataset):
    """Thin wrapper that normalises medmnist label format.

    medmnist returns labels as numpy arrays of shape ``(1,)`` for single-label
    datasets — CrossEntropyLoss expects plain integer scalars, so this wrapper
    converts them. For multi-label datasets (ChestMNIST), the label is instead
    a ``(14,)`` multi-hot binary vector — kept as a float tensor for
    BCEWithLogitsLoss, never squeezed to a scalar.
    """

    def __init__(self, base_ds: medmnist.MedMNIST, multi_label: bool = False) -> None:
        self.base_ds = base_ds
        self.multi_label = multi_label

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        img, label = self.base_ds[idx]
        if self.multi_label:
            return img, torch.from_numpy(np.asarray(label, dtype=np.float32))
        return img, int(np.asarray(label).squeeze())

    @property
    def labels(self) -> np.ndarray:
        """All labels — used by WeightedRandomSampler (single-label only)."""
        if self.multi_label:
            return self.base_ds.labels
        return self.base_ds.labels.squeeze()


# ---------------------------------------------------------------------------
# Public loader factory
# ---------------------------------------------------------------------------

def get_medmnist_loaders(
    dataset_name: str,
    batch_size: int = 32,
    resize_to: int = 224,
    download: bool = True,
    data_root: str = "data/medmnist",
    num_workers: int = 0,
    use_weighted_sampling: Optional[bool] = None,
    use_gray_to_get: bool = False,
    native_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train, val, and test DataLoaders for a MedMNIST dataset.

    Transform pipeline
    ------------------
    * **Train / Val**: Resize → ToTensor → channel_op → Normalize →
      RandomShiftAndNoise.
      Both splits use the *same stochastic transform* so that repeating the
      val loop ``n_reps`` times yields a Monte Carlo estimate of the expected
      loss under augmentation (supervisor's protocol).

    * **Test**: Resize → ToTensor → channel_op → Normalize.
      Clean and deterministic for final evaluation.

    For grayscale datasets (breast, pneumonia), ``channel_op`` is one of:

    * ``use_gray_to_get=False`` (default): ``T.Grayscale(3)`` — simple
      channel replication.  Use this for the baseline ablation runs.
    * ``use_gray_to_get=True``: :class:`GrayToGET` — morphological
      Gray/Edges/Texture 3-channel conversion.  Treat this as an extra
      ablation axis ("with/without GET preprocessing") run after the
      baseline ablations complete.

    Args:
        dataset_name: One of the keys in ``MEDMNIST_INFO`` (``blood``,
            ``breast``, ``pneumonia``, ``retina``, ``derma``, ``path``,
            ``chest``, ``oct``, ``tissue``, ``organa``, ``organc``,
            ``organs``). ``chest`` is multi-label — see ``MULTI_LABEL_DATASETS``.
        batch_size: Samples per batch for all three splits.
        resize_to: Target spatial resolution.  With ``grid_size=16`` this
            gives ``patch_size = resize_to // 16``.  Default 224 → 14 px.
        download: Auto-download if not cached locally.
        data_root: MedMNIST cache directory.
        num_workers: DataLoader workers for the training loader (4 is a
            verified-safe default on current torch/Windows — ~4.75x
            data-loading throughput vs. 0 on this project's hardware; pass 0
            if workers ever crash in a different environment). Val/test
            loaders always use 0 regardless of this value — they're
            re-iterated multiple times per epoch (stochastic MC validation
            reps) and worker spawn cost would dominate for small val sets.
        use_weighted_sampling: Override the per-dataset default for
            ``WeightedRandomSampler``.  ``None`` uses the built-in default.
        native_size: MedMNIST's own native resolution to download —
            one of ``28`` (default when ``None``), ``64``, ``128``, or
            ``224``.  This is a *different, independently-curated* release
            from the classic 28px array, not a resize of it — passing
            ``224`` here downloads genuinely higher-fidelity source images
            (matching the advisor's ViTDCT protocol) rather than relying on
            ``resize_to`` to upsample the low-resolution 28px default.
            ``resize_to`` still applies afterwards (a no-op when it equals
            ``native_size``).

    Returns:
        ``(train_loader, val_loader, test_loader, class_names)``
    """
    if dataset_name not in MEDMNIST_INFO:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Choose from: {sorted(MEDMNIST_INFO)}"
        )

    MedDS, is_rgb = MEDMNIST_INFO[dataset_name]
    multi_label = dataset_name in MULTI_LABEL_DATASETS

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Grayscale → 3-channel conversion.
    # T.Grayscale is a PIL op → must go BEFORE T.ToTensor.
    # GrayToGET is a tensor op → goes AFTER T.ToTensor (future ablation axis).
    pil_ops    = []   # applied before ToTensor (PIL space)
    tensor_ops = []   # applied after  ToTensor (tensor space)

    if not is_rgb:
        if use_gray_to_get:
            tensor_ops = [GrayToGET()]                        # morphological G/E/T
        else:
            pil_ops = [T.Grayscale(num_output_channels=3)]   # standard replication

    # Shared deterministic prefix (PIL → normalised tensor).
    base_tf = [
        T.Resize((resize_to, resize_to), interpolation=T.InterpolationMode.BILINEAR),
        *pil_ops,     # T.Grayscale(3) here — PIL op before ToTensor
        T.ToTensor(),
        *tensor_ops,  # GrayToGET here — tensor op after ToTensor
        T.Normalize(mean, std),
    ]

    # Stochastic augmentation appended for train + val.
    augment = RandomShiftAndNoise()

    stochastic_tf = T.Compose([*base_tf, augment])   # train & val
    clean_tf      = T.Compose(base_tf)                # test only

    os.makedirs(data_root, exist_ok=True)
    kw = {"root": data_root, "download": download}
    if native_size is not None:
        kw["size"] = native_size

    raw_train = MedDS(split="train", transform=stochastic_tf, **kw)
    raw_val   = MedDS(split="val",   transform=stochastic_tf, **kw)
    raw_test  = MedDS(split="test",  transform=clean_tf,      **kw)

    train_ds = _MedMNISTWrapper(raw_train, multi_label=multi_label)
    val_ds   = _MedMNISTWrapper(raw_val,   multi_label=multi_label)
    test_ds  = _MedMNISTWrapper(raw_test,  multi_label=multi_label)

    # WeightedRandomSampler on training split only. Never applies to
    # multi-label datasets — its per-class-count weighting assumes one
    # mutually-exclusive scalar label per sample, not a multi-hot vector.
    weighted = (
        not multi_label
        and (
            use_weighted_sampling
            if use_weighted_sampling is not None
            else dataset_name in _WEIGHTED_DATASETS
        )
    )
    if weighted:
        labels      = train_ds.labels.astype(int)
        classes     = np.unique(labels)
        class_count = np.array([(labels == c).sum() for c in classes])
        class_w     = 1.0 / class_count
        sample_w    = torch.from_numpy(
            np.array([class_w[np.where(classes == lbl)[0][0]] for lbl in labels])
        ).double()
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        # persistent_workers keeps the same worker processes alive across
        # epochs instead of respawning them every epoch -- on Windows, each
        # respawn re-imports torch in the new process, which can
        # intermittently fail to load shm.dll (WinError 1114). Respawning
        # once per training run instead of once per epoch cuts exposure to
        # that race by orders of magnitude over a long sweep.
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=(num_workers > 0),
        )

    # Val uses shuffle=True so that different mini-batch groupings across
    # repetitions don't introduce ordering bias in the MC loss estimate.
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Class names from medmnist INFO dict. INFO's keys keep the "mnist"
    # suffix (e.g. "chestmnist", "bloodmnist") — do NOT strip it, or this
    # silently falls through to a generic range(batch_size) placeholder
    # (previously undetected because it happened to look plausible for
    # small batch sizes vs. class counts).
    info_key    = MedDS.__name__.lower()
    info        = INFO.get(info_key, {})
    class_names = list(info.get("label", {}).values()) or [
        str(i) for i in range(info.get("n_classes", batch_size))
    ]

    return train_loader, val_loader, test_loader, class_names
