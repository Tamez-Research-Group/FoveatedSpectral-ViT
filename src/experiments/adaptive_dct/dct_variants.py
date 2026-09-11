"""
src/experiments/adaptive_dct/dct_variants.py
============================================
Two adaptive DCT patch embedding variants for the Spiral ViT adaptive frequency
selection experiment.

**Scientific motivation**

The baseline ``DCTEmbedding`` (src/models/vit.py) retains a fixed top-left
``freq_keep × freq_keep`` block of DCT coefficients — the lowest spatial
frequencies — for every dataset.  This mirrors the standard JPEG quantization
philosophy but ignores the fact that different imaging modalities emphasize
different frequency bands:

  * Fundus photography (EyePACS): rich mid-frequency texture (vessel structure,
    optic disc margin).
  * Brain MRI T1:  predominantly low-frequency (tissue contrast between GM/WM).
  * Chest radiograph: broad spectrum (air/fluid/bone density transitions).

This module provides two drop-in replacements for ``DCTEmbedding`` that adapt
the coefficient selection strategy:

**Option 1 — EnergyDCTEmbedding** (zero extra parameters, JPEG-style):
    Calibrates coefficient selection before training begins using a single
    no-gradient pass over the training set.  The K highest-energy positions in
    the mean absolute DCT coefficient map are retained.  Analogous to deriving a
    custom JPEG quantization table from the dataset's perceptual energy
    statistics.  Calibrated indices are stored as a persistent buffer so the
    model can be reloaded for inference without recalibration.

**Option 2 — LearnedDCTEmbedding** (patch_size² extra parameters, end-to-end):
    A learnable scalar mask of shape ``[patch_size, patch_size]`` is multiplied
    element-wise with the full DCT coefficient grid after passing through a
    sigmoid activation.  L1 regularization on the activated mask encourages
    sparsity, making the effective selection interpretable post-training.  The
    training loop must add ``model.patch_embed.l1_loss`` to the classification
    loss after each forward pass.

Both classes accept the same constructor arguments as ``DCTEmbedding`` and
produce output of identical shape ``(B, num_patches, dim)``, making them
drop-in replacements in the ViT backbone without modifying ``src/models/vit.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# L1 regularization weight for LearnedDCTEmbedding.
# Small enough not to dominate the classification loss, large enough to drive
# unimportant frequencies toward zero over many epochs.
LAMBDA_L1: float = 1e-4


class EnergyDCTEmbedding(nn.Module):
    """JPEG-style energy-calibrated DCT patch embedding (Option 1).

    Before training begins, ``calibrate()`` makes one full no-gradient pass
    over the training DataLoader.  For each image the 2-D DCT-II is computed
    for all patches and the mean absolute coefficient value at each grid
    position is accumulated.  The ``K = freq_keep²`` highest-energy positions
    are then stored as a persistent buffer (``selected_indices``) and used in
    every subsequent forward pass instead of the fixed top-left block used by
    ``DCTEmbedding``.

    This variant adds **zero** extra learnable parameters beyond the baseline:
    the ``proj`` linear layer has the same input size ``channels × freq_keep²``
    as ``DCTEmbedding``, and ``selected_indices`` is a non-learnable buffer.

    Args:
        image_size: Side length of the square input image in pixels.
        patch_size: Side length of each square patch ``P``.
            ``image_size`` must be divisible by ``patch_size``.
        dim: Transformer embedding dimension.
        channels: Number of input channels. Defaults to 3.
        freq_keep: Number of DCT coefficient positions retained per axis.
            Total retained: ``K = freq_keep²``. Defaults to 4.

    Note:
        ``calibrate()`` must be called once on the training DataLoader before
        any forward pass.  The selected indices are included in the model
        checkpoint (``persistent=True``) and are restored on load — no
        recalibration is required for inference.  Until ``calibrate()`` is
        called the module falls back to the same top-left ``freq_keep²`` block
        used by ``DCTEmbedding``.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        channels: int = 3,
        freq_keep: int = 4,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.channels = channels
        self.freq_keep = freq_keep
        self.K: int = freq_keep * freq_keep  # total retained coefficients
        self.num_patches: int = (image_size // patch_size) ** 2

        # Pre-computed DCT-II analysis matrix.  Non-persistent — re-derived
        # on load from patch_size, not stored in checkpoints.
        C = self._create_dct_matrix(patch_size)
        self.register_buffer("dct_C", C, persistent=False)

        # Default selection mirrors the fixed top-left freq_keep×freq_keep block
        # of DCTEmbedding (row-major flat indices into the P×P coefficient grid).
        # Overwritten by calibrate() before training.
        rows = torch.arange(freq_keep).unsqueeze(1).expand(freq_keep, freq_keep)
        cols = torch.arange(freq_keep).unsqueeze(0).expand(freq_keep, freq_keep)
        default_idx = (rows * patch_size + cols).reshape(-1)  # (K,)
        # Persistent so calibrated indices survive checkpointing.
        self.register_buffer("selected_indices", default_idx, persistent=True)

        # Same projection size as DCTEmbedding — zero parameter overhead.
        self.proj = nn.Linear(channels * self.K, dim)

    @staticmethod
    def _create_dct_matrix(P: int) -> torch.Tensor:
        """Build the orthonormal DCT-II analysis matrix of size ``P × P``.

        ``C[k, n] = alpha * cos(pi * (n + 0.5) * k / P)``
        where ``alpha = sqrt(1/P)`` for ``k=0``, else ``sqrt(2/P)``.

        Args:
            P: Patch side length in pixels.

        Returns:
            DCT matrix of shape ``(P, P)``.
        """
        C = torch.zeros(P, P)
        for k in range(P):
            alpha = math.sqrt(1.0 / P) if k == 0 else math.sqrt(2.0 / P)
            for n in range(P):
                C[k, n] = alpha * math.cos(math.pi * (n + 0.5) * k / P)
        return C

    def _dct2d(self, patches: torch.Tensor) -> torch.Tensor:
        """Apply separable 2-D DCT-II to a batch of patches.

        Args:
            patches: Shape ``(B, N, C, P, P)`` — batch of ``N`` patches.

        Returns:
            DCT coefficient grid of shape ``(B, N, C, P, P)``.
        """
        Cmat = self.dct_C           # (P, P)
        Ct = Cmat.t()               # (P, P)
        # Row-wise DCT, then column-wise DCT (separable 2-D DCT-II).
        tmp = torch.einsum("ij,bncpj->bncpi", Cmat, patches)
        return torch.einsum("bncpj,ij->bncpi", tmp, Ct)

    @torch.no_grad()
    def calibrate(
        self,
        dataloader: DataLoader,
        K: Optional[int] = None,
        device: Optional[torch.device] = None,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Calibrate coefficient selection from training-set energy statistics.

        Runs one full pass over ``dataloader`` without gradients.  For each
        batch, non-overlapping patches are extracted, the 2-D DCT-II is applied,
        and the mean absolute value per coefficient position is accumulated.
        The ``K`` positions with the highest energy are stored in
        ``selected_indices``.

        Args:
            dataloader: Training DataLoader.  Must NOT include test data (no
                data leakage).  Supports both 2-tuple ``(images, labels)`` and
                3-tuple ``(images, labels, meta)`` batch formats.
            K: Number of coefficient positions to select.  Defaults to
                ``freq_keep²`` (matching the fixed-baseline count).
            device: Compute device for accumulation.  Defaults to the device
                of the module's ``dct_C`` buffer.
            save_path: Optional file path.  If provided, saves the normalized
                energy map as a ``(P, P)`` numpy ``.npy`` array for later
                visualization.

        Returns:
            Energy map of shape ``(patch_size, patch_size)``, normalized so
            that the maximum value is 1.0.
        """
        if K is None:
            K = self.K

        # Determine target device; fall back to CPU if no buffers exist yet.
        if device is None:
            try:
                device = next(self.buffers()).device
            except StopIteration:
                device = torch.device("cpu")

        self.to(device)

        P = self.patch_size
        energy_acc = torch.zeros(P, P, device=device)
        n_batches = 0

        for batch in dataloader:
            # batch[0] is images for both 2-tuple and 3-tuple formats.
            imgs = batch[0].to(device)          # (B, C, H, W)
            B, C, H, W = imgs.shape

            # Extract non-overlapping patches via unfold.
            patches = F.unfold(imgs, kernel_size=P, stride=P)   # (B, C·P², N)
            patches = patches.transpose(1, 2).contiguous()       # (B, N, C·P²)
            patches = patches.view(B, -1, C, P, P)               # (B, N, C, P, P)

            dct = self._dct2d(patches)                           # (B, N, C, P, P)

            # Mean |coefficient| over batch, spatial-patch, and channel axes.
            energy_acc += dct.abs().mean(dim=(0, 1, 2))          # (P, P)
            n_batches += 1

        energy_map = energy_acc / max(n_batches, 1)              # (P, P)

        # Select top-K positions by accumulated energy.
        flat_energy = energy_map.reshape(-1)                     # (P²,)
        _, top_indices = flat_energy.topk(K)
        top_indices, _ = top_indices.sort()                      # reproducible order

        self.selected_indices.copy_(top_indices.to(self.selected_indices.device))

        # Normalize 0–1 for visualization.
        energy_np = energy_map.cpu().numpy()
        energy_np = energy_np / (energy_np.max() + 1e-8)

        if save_path is not None:
            np.save(save_path, energy_np)
            print(f"[EnergyDCT] Energy map saved → {save_path}")

        print(
            f"[EnergyDCT] Calibrated over {n_batches} batches. "
            f"Selected {K} of {P * P} positions."
        )
        return energy_np

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract energy-calibrated DCT features from image patches.

        Args:
            x: Input image tensor of shape ``(B, C, H, W)``.

        Returns:
            Token sequence of shape ``(B, num_patches, dim)``.
        """
        B, C, H, W = x.shape
        P = self.patch_size

        patches = F.unfold(x, kernel_size=P, stride=P)          # (B, C·P², N)
        patches = patches.transpose(1, 2).contiguous()           # (B, N, C·P²)
        patches = patches.view(B, self.num_patches, C, P, P)     # (B, N, C, P, P)

        dct = self._dct2d(patches)                               # (B, N, C, P, P)

        # Gather K energy-selected positions per channel.
        dct_flat = dct.reshape(B, self.num_patches, C, P * P)    # (B, N, C, P²)
        idx = self.selected_indices.view(1, 1, 1, self.K).expand(B, self.num_patches, C, -1)
        dct_sel = dct_flat.gather(3, idx)                        # (B, N, C, K)

        feat = dct_sel.reshape(B, self.num_patches, C * self.K)  # (B, N, C·K)
        return self.proj(feat)                                    # (B, N, dim)


class LearnedDCTEmbedding(nn.Module):
    """End-to-end learnable soft-mask DCT patch embedding (Option 2).

    A learnable scalar mask of shape ``[patch_size, patch_size]`` is multiplied
    element-wise with the full DCT coefficient grid after passing through a
    sigmoid activation.  L1 regularization on the activated mask encourages
    sparsity, making the frequency selection interpretable after training.

    The mask is initialized to ``ones`` (``sigmoid(1) ≈ 0.73``), giving a mild
    initial preference for all frequencies.  As training progresses, L1
    regularization drives uninformative frequency positions toward zero.

    **Learnable parameters added compared to DCTEmbedding:**

    * ``mask``:  ``patch_size × patch_size`` scalar weights  (e.g. 36×36 = 1 296
      for P=36)
    * ``proj``:  ``channels × P² → dim`` linear layer — larger than
      ``DCTEmbedding``'s ``channels × freq_keep²`` layer because the full
      masked coefficient grid is projected.

    For P=36, channels=3, freq_keep=16, dim=256 the extra parameters over
    ``DCTEmbedding`` are approximately
    ``3 × (1296 − 256) × 256 + 1296 ≈ 799 K``.

    Args:
        image_size: Side length of the square input image in pixels.
        patch_size: Side length of each square patch ``P``.
        dim: Transformer embedding dimension.
        channels: Number of input channels. Defaults to 3.
        freq_keep: Accepted for API compatibility with ``DCTEmbedding``;
            not used to constrain the mask, which covers all ``P²`` positions.
            Defaults to 4.

    Note:
        After each forward pass, the L1 regularization term is stored as
        ``self.l1_loss``.  The training loop must add it to the classification
        loss::

            outputs = model(imgs)
            loss = criterion(outputs, labels) + model.patch_embed.l1_loss
            loss.backward()

        During validation / test evaluation (``torch.no_grad()``) ``l1_loss``
        is still computed and stored but does not affect gradients.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        channels: int = 3,
        freq_keep: int = 4,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.channels = channels
        self.freq_keep = freq_keep  # stored for API compatibility; not used in forward
        self.num_patches: int = (image_size // patch_size) ** 2

        C = self._create_dct_matrix(patch_size)
        self.register_buffer("dct_C", C, persistent=False)

        # Learnable frequency importance mask — one scalar per DCT position.
        # This adds exactly patch_size × patch_size learnable parameters.
        self.mask = nn.Parameter(torch.ones(patch_size, patch_size))

        # Project the full soft-masked coefficient grid (channels × P²) to dim.
        # Larger than DCTEmbedding.proj because all P² positions are retained.
        self.proj = nn.Linear(channels * patch_size * patch_size, dim)

        # L1 penalty from the most recent forward pass; read by the training loop.
        self.l1_loss: torch.Tensor = torch.tensor(0.0)

    @staticmethod
    def _create_dct_matrix(P: int) -> torch.Tensor:
        """Build the orthonormal DCT-II analysis matrix of size ``P × P``.

        Args:
            P: Patch side length in pixels.

        Returns:
            DCT matrix of shape ``(P, P)``.
        """
        C = torch.zeros(P, P)
        for k in range(P):
            alpha = math.sqrt(1.0 / P) if k == 0 else math.sqrt(2.0 / P)
            for n in range(P):
                C[k, n] = alpha * math.cos(math.pi * (n + 0.5) * k / P)
        return C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft-masked DCT and project to embedding space.

        Computes the full 2-D DCT-II for every patch, multiplies element-wise
        by ``sigmoid(mask)`` to softly select informative frequencies, flattens,
        and projects to ``dim``.

        The L1 regularization term ``LAMBDA_L1 * sigmoid(mask).sum()`` is
        stored in ``self.l1_loss`` after every call.  It must be added to the
        classification loss in the training loop (see class docstring).

        Args:
            x: Input image tensor of shape ``(B, C, H, W)``.

        Returns:
            Token sequence of shape ``(B, num_patches, dim)``.
            The L1 penalty is accessible as ``self.l1_loss`` (scalar tensor).
        """
        B, C, H, W = x.shape
        P = self.patch_size

        patches = F.unfold(x, kernel_size=P, stride=P)          # (B, C·P², N)
        patches = patches.transpose(1, 2).contiguous()           # (B, N, C·P²)
        patches = patches.view(B, self.num_patches, C, P, P)     # (B, N, C, P, P)

        Cmat = self.dct_C
        Ct = Cmat.t()
        tmp = torch.einsum("ip,bncpj->bncij", Cmat, patches)   # C applied to the row axis
        dct = torch.einsum("bncij,kj->bncik", tmp, Cmat)        # C applied to the column axis

        # Differentiable soft frequency selection via sigmoid gate.
        soft_mask = torch.sigmoid(self.mask)                     # (P, P) ∈ (0, 1)
        dct_masked = dct * soft_mask                             # broadcasts over (B,N,C)

        # Store L1 penalty for the training loop to add to the total loss.
        self.l1_loss = LAMBDA_L1 * soft_mask.sum()

        feat = dct_masked.reshape(B, self.num_patches, C * P * P)  # (B, N, C·P²)
        return self.proj(feat)                                      # (B, N, dim)
