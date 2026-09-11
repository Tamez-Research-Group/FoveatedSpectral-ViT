"""
src/models/vit.py
=================
Vision Transformer (ViT) with two independently configurable experimental axes:

  1. **Patch embedding** (``embed_type``): how raw pixel patches are projected
     into the transformer's embedding space.
     - ``"conv"``: standard strided convolution — the original ViT approach.
     - ``"dct"``: 2-D Discrete Cosine Transform retaining only the
       ``freq_keep × freq_keep`` lowest-frequency coefficients per channel
       before a linear projection.  Acts as a built-in low-pass filter,
       suppressing acquisition noise common in medical imaging (MRI k-space
       artefacts, X-ray detector noise) while preserving clinically relevant
       coarse structure.

  2. **Token ordering** (``token_order``): how image patches are sequenced
     before entering the transformer.
     - ``"spiral"``: patches are re-ordered from the spatial centre outward.
       Combined with the Gaussian inductive bias in MultiHeadSelfAttention,
       this encourages the model to devote early attention to central,
       diagnostically salient regions — motivated by the observation that
       pathology in retinal, skin, and brain imaging is typically framed
       centrally.
     - ``"raster"``: standard left-to-right, top-to-bottom order (ViT
       baseline).

Both axes are independently toggleable so that controlled ablation experiments
can isolate the contribution of each design choice.

**Gaussian inductive bias** (MultiHeadSelfAttention): a non-learnable
multiplicative weight vector ``g[j] = exp(-j² / (2σ²))`` is applied to
attention logits along the key dimension.  Because ``g`` decays with sequence
index ``j``, tokens appearing *early* in the sequence receive amplified
attention.  When combined with spiral ordering (early = centre), this acts as
a soft spatial prior favouring the image centre.  The bias is gated by the
``gaussian_bias`` constructor argument and is independent of ``token_order``,
enabling a full 2×2×2 factorial ablation over all three design axes.  When
``gaussian_bias=True`` (default) and ``token_order="spiral"`` this reproduces
the original Spiral ViT behaviour.  When ``token_order="raster"`` and
``gaussian_bias=True`` the bias applies a top-left prior, which is the
control condition for the Gaussian bias ablation row.
"""

import math
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.token_order import spiral_indices, raster_indices


class PatchEmbedding(nn.Module):
    """Standard convolutional patch embedding (original ViT design).

    A ``Conv2d`` with ``kernel_size == stride == patch_size`` is equivalent
    to extracting non-overlapping patches and projecting each with a shared
    linear layer.

    Args:
        image_size: Side length of the square input image in pixels.
        patch_size: Side length of each square patch.
            ``image_size`` must be divisible by ``patch_size``.
        dim: Transformer embedding dimension.
        channels: Number of input channels. Defaults to 3.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        channels: int = 3,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape ``(B, C, H, W)``.

        Returns:
            Token sequence of shape ``(B, num_patches, dim)``.
        """
        x = self.proj(x)       # (B, dim, H/P, W/P)
        x = x.flatten(2)       # (B, dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, dim)
        return x


class DCTEmbedding(nn.Module):
    """DCT-based patch embedding with low-frequency truncation.

    Processing pipeline per patch:
      1. Extract non-overlapping ``P × P`` patches via ``F.unfold``.
      2. Apply a 2-D DCT-II transform (row-wise then column-wise).
      3. Retain only the top-left ``freq_keep × freq_keep`` DCT coefficients
         (lowest spatial frequencies; smooth edges and coarse shapes).
      4. Project the retained coefficients linearly to the embedding dimension.

    The DCT matrix ``C`` is pre-computed and registered as a non-persistent
    buffer — it lives on the correct device but is not saved in checkpoints.

    Args:
        image_size: Side length of the square input image in pixels.
        patch_size: Side length of each patch ``P``.
        dim: Transformer embedding dimension.
        channels: Number of input channels. Defaults to 3.
        freq_keep: Number of low-frequency DCT rows/columns retained per
            channel.  Effective pre-projection feature size:
            ``channels * freq_keep ** 2``. Defaults to 4.
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
        assert image_size % patch_size == 0
        freq_keep = min(freq_keep, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.channels = channels
        self.freq_keep = freq_keep
        self.num_patches = (image_size // patch_size) ** 2

        C = self._create_dct_matrix(patch_size)
        self.register_buffer("dct_C", C, persistent=False)

        self.proj = nn.Linear(channels * freq_keep * freq_keep, dim)

    @staticmethod
    def _create_dct_matrix(P: int) -> torch.Tensor:
        """Build the orthonormal DCT-II analysis matrix of size ``P × P``.

        ``C[k, n] = alpha * cos(pi * (n + 0.5) * k / P)``
        where ``alpha = sqrt(1/P)`` for ``k=0``, else ``sqrt(2/P)``.

        Args:
            P: Patch side length.

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
        """
        Args:
            x: Input image tensor of shape ``(B, C, H, W)``.

        Returns:
            Token sequence of shape ``(B, num_patches, dim)``.
        """
        B, C, H, W = x.shape
        P = self.patch_size
        fk = self.freq_keep

        # Extract patches: (B, C*P*P, num_patches) → (B, num_patches, C, P, P)
        patches = F.unfold(x, kernel_size=P, stride=P)
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(B, self.num_patches, C, P, P)

        Cmat = self.dct_C
        Ct = Cmat.transpose(0, 1)

        # 2-D DCT: C applied along rows, then C^T along columns.
        tmp = torch.einsum("ij,bncpj->bncpi", Cmat, patches)
        dct = torch.einsum("bncpj,ij->bncpi", tmp, Ct)

        # Retain only the top-left freq_keep × freq_keep block (low frequencies).
        dct_low = dct[:, :, :, :fk, :fk]

        feat = dct_low.reshape(B, self.num_patches, C * fk * fk)
        return self.proj(feat)


class DCTEmbeddingFast(nn.Module):
    """Conv2d-based DCT patch embedding — identical math, cuDNN execution path.

    ``DCTEmbedding`` computes the 2-D DCT-II via ``F.unfold`` followed by two
    ``torch.einsum`` calls, which PyTorch dispatches as separate CUDA kernels
    and cannot fuse.  This class expresses the identical computation as a
    single **grouped** ``Conv2d`` whose weights are the outer products of the
    DCT-II basis vectors:

        ``W[c·K²+kx·K+ky, 0] = outer(C[kx], C[ky])``   for c in channels,
                                                           kx, ky in freq_keep

    A grouped convolution with ``groups=channels`` applies each channel's
    ``freq_keep²`` filters independently, mirroring the per-channel separable
    DCT.  cuDNN optimises this as a single Winograd or implicit-GEMM kernel,
    giving the same speed profile as ``PatchEmbedding``.

    **Numerically equivalent to** ``DCTEmbedding`` up to floating-point
    rounding order.  The ``proj`` linear layer has the same input size
    (``channels × freq_keep²``) and can accept weights transferred from a
    trained ``DCTEmbedding`` checkpoint without any conversion.

    The convolution weights are frozen (``requires_grad=False``); only
    ``proj`` is learned.

    Args:
        image_size: Side length of the square input image in pixels.
        patch_size: Side length of each patch ``P``.
        dim: Transformer embedding dimension.
        channels: Number of input channels. Defaults to 3.
        freq_keep: Low-frequency DCT coefficients retained per axis per
            channel.  Defaults to 4.
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
        assert image_size % patch_size == 0
        freq_keep = min(freq_keep, patch_size)
        n_freqs = freq_keep * freq_keep
        self.num_patches = (image_size // patch_size) ** 2

        # Build the DCT-II matrix vectorised (no Python loop per element).
        C = self._dct_matrix(patch_size)           # (P, P)

        # Pre-compute the 2-D DCT basis as conv kernels.
        # Output channel c·K²+kx·K+ky  ←  outer(C[kx], C[ky])  on input ch c.
        weight = torch.zeros(channels * n_freqs, 1, patch_size, patch_size)
        for c in range(channels):
            for kx in range(freq_keep):
                for ky in range(freq_keep):
                    weight[c * n_freqs + kx * freq_keep + ky, 0] = (
                        torch.outer(C[kx], C[ky])
                    )

        # Grouped conv: each input channel processed by its own freq_keep² filters.
        self.dct_conv = nn.Conv2d(
            channels, channels * n_freqs,
            kernel_size=patch_size, stride=patch_size,
            groups=channels, bias=False,
        )
        # Freeze — the DCT basis is a fixed mathematical transform, not learned.
        self.dct_conv.weight = nn.Parameter(weight, requires_grad=False)

        self.proj = nn.Linear(channels * n_freqs, dim)

    @staticmethod
    def _dct_matrix(P: int) -> torch.Tensor:
        """Vectorised DCT-II analysis matrix (P × P), orthonormal rows."""
        k = torch.arange(P, dtype=torch.float64)
        n = torch.arange(P, dtype=torch.float64)
        alpha = torch.where(
            k == 0,
            torch.full_like(k, math.sqrt(1.0 / P)),
            torch.full_like(k, math.sqrt(2.0 / P)),
        )
        C = alpha[:, None] * torch.cos(
            math.pi * (n[None, :] + 0.5) * k[:, None] / P
        )
        return C.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape ``(B, C, H, W)``.

        Returns:
            Token sequence of shape ``(B, num_patches, dim)``.
        """
        # (B, C·freq_keep², H/P, W/P) — single cuDNN call
        out = self.dct_conv(x)
        # (B, num_patches, C·freq_keep²)
        out = out.flatten(2).transpose(1, 2)
        return self.proj(out)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with a non-learnable Gaussian inductive bias.

    Standard scaled dot-product attention is augmented with a position-decay
    weight applied to the key dimension:

        ``g[j] = exp(-j² / (2σ²)),   σ = N / 2``

        ``scores[b, h, i, j] *= g[j]``

    Tokens with a *lower sequence index j* receive larger attention weights
    after softmax.  When ``token_order="spiral"`` those early tokens are the
    spatially central patches, so the bias acts as a soft prior favouring the
    image centre — clinically motivated because pathology in retinal, skin,
    and brain imaging tends to be centrally framed.

    The bias is disabled when ``use_gaussian_bias=False`` so that
    ``token_order="raster"`` conditions receive a neutral spatial distribution
    instead of an unintended top-left-corner prior.

    Args:
        dim: Total embedding dimension (must be divisible by ``heads``).
        heads: Number of attention heads.
        dropout: Dropout probability after softmax and on the output
            projection. Defaults to 0.0.
        use_gaussian_bias: Whether to apply the Gaussian positional bias.
            Should be ``True`` only when ``token_order="spiral"``.
            Defaults to ``True``.
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0, use_gaussian_bias: bool = True) -> None:
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        self.heads = heads
        self.dim_head = dim // heads
        self.use_gaussian_bias = use_gaussian_bias
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token sequence of shape ``(B, N, D)``.

        Returns:
            Output token sequence of shape ``(B, N, D)``.
        """
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, dim_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim_head)

        if self.use_gaussian_bias:
            # Amplify attention to low-index (central) keys; only meaningful
            # when token_order="spiral" (low index == spatial centre).
            t = torch.arange(N, device=x.device).float()
            sigma = N / 2.0
            gauss_weight = torch.exp(-(t ** 2) / (2 * sigma ** 2))  # (N,)
            scores = scores * gauss_weight.view(1, 1, 1, N)

        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation (standard transformer FFN block).

    Args:
        dim: Input and output feature dimension.
        mlp_dim: Hidden layer dimension (typically 2–4× ``dim``).
        dropout: Dropout probability. Defaults to 0.0.
    """

    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Single transformer encoder block using Pre-Norm (LayerNorm before sub-layer).

    Pre-Norm stabilises gradient flow in deep transformers compared to the
    Post-Norm formulation in the original "Attention is All You Need".

    Structure: ``x → LN → MHSA → residual → LN → FFN → residual``

    Args:
        dim: Embedding dimension.
        heads: Number of attention heads.
        mlp_dim: Feed-forward hidden dimension.
        dropout: Dropout probability. Defaults to 0.0.
        use_gaussian_bias: Forwarded to ``MultiHeadSelfAttention``.
            Defaults to ``True``.
    """

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0, use_gaussian_bias: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout, use_gaussian_bias=use_gaussian_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer with configurable token ordering and patch embedding.

    This is the central model of the Spiral ViT paper.  It supports two
    independently toggleable experimental axes (``embed_type`` and
    ``token_order``) enabling 2×2 ablation studies without any code changes.

    The 2-D positional embedding (shape ``(grid_h, grid_w, dim)``) is
    re-ordered together with the patch tokens, so spatial position information
    is always consistent with the physical patch location regardless of the
    chosen traversal order.

    The token ordering index is registered as a non-persistent buffer: it is
    moved to the correct device automatically but is not serialised in
    checkpoints because it can always be re-derived from the grid size.

    Args:
        image_size: Side length of the square input image in pixels.
        patch_size: Side length of each square patch.
        num_classes: Number of output classes.
        dim: Transformer embedding dimension.
        depth: Number of EncoderBlock layers.
        heads: Number of attention heads per block.
        mlp_dim: Hidden dimension of the feed-forward sub-network.
        channels: Number of input image channels. Defaults to 3.
        dropout: Dropout rate inside attention and FFN. Defaults to 0.0.
        emb_dropout: Dropout applied to the summed patch + positional
            embeddings before the first encoder block. Defaults to 0.0.
        embed_type: Patch embedding strategy — ``"conv"``, ``"dct"``,
            ``"dct_fast"`` (grouped Conv2d DCT, same math as ``"dct"`` but
            uses a single cuDNN kernel), or ``"learned_dct"`` (differentiable
            sigmoid-gated DCT mask, see
            ``experiments.adaptive_dct.dct_variants.LearnedDCTEmbedding`` —
            retains the full patch's DCT grid but learns a per-frequency
            importance weight jointly with the classification objective;
            the training loop must add ``model.patch_embed.l1_loss`` to the
            loss, see ``utils/training_tracked.py`` and
            ``utils/training_general.py``).  Defaults to ``"conv"``.
        freq_keep: Low-frequency DCT coefficients retained per axis per
            channel.  Only used when ``embed_type`` is ``"dct"`` or
            ``"dct_fast"``; ignored by ``"learned_dct"`` (its mask always
            covers the full ``patch_size × patch_size`` grid). Defaults to 16.
        token_order: Patch traversal order — ``"spiral"`` or ``"raster"``.
            Defaults to ``"raster"``.
        gaussian_bias: Whether to apply the Gaussian positional bias in every
            ``MultiHeadSelfAttention`` block.  When ``True`` (default) the
            bias is active regardless of ``token_order``, enabling full
            factorial ablation.  Set to ``False`` to disable the bias
            completely (e.g. ablation configs 1–5 in Table 2).
        spiral_pool: When ``True``, the output representation is a weighted
            blend of the CLS token and a pooled average of the
            post-transformer patch tokens.  The pooling weights are
            determined by ``spiral_pool_mode``.  Only meaningful with
            ``token_order="spiral"``.  Default ``False`` preserves original
            behaviour.
        spiral_pool_alpha: Blend coefficient in ``[0, 1]``.  The final
            feature vector is
            ``(1 - alpha) * cls_token + alpha * pooled_patches``.
            Default ``0.5`` gives equal weight to both.
        spiral_pool_mode: Weighting scheme applied when ``spiral_pool=True``.
            ``"gaussian"`` (default) applies a Gaussian decay
            ``g[j] = exp(-j² / (2σ²))`` over the spiral sequence, giving
            the highest weight to the centre patches — a foveal prior.
            ``"average"`` uses uniform weights (plain mean pooling), which
            provides the same CLS-blend benefit without a spatial bias.
            Comparing the two isolates whether the centre-weighting itself
            is what drives any accuracy gain.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        embed_type: Literal["conv", "dct", "dct_fast", "learned_dct"] = "conv",
        freq_keep: int = 16,
        token_order: Literal["spiral", "raster"] = "raster",
        gaussian_bias: bool = True,
        spiral_pool: bool = False,
        spiral_pool_alpha: float = 0.5,
        spiral_pool_mode: Literal["gaussian", "average"] = "gaussian",
    ) -> None:
        super().__init__()

        if embed_type == "dct":
            self.patch_embed = DCTEmbedding(image_size, patch_size, dim, channels, freq_keep)
        elif embed_type == "dct_fast":
            self.patch_embed = DCTEmbeddingFast(image_size, patch_size, dim, channels, freq_keep)
        elif embed_type == "learned_dct":
            # Deferred import: keeps this module's import-time dependencies
            # unchanged for the conv/dct/dct_fast paths used everywhere else.
            from experiments.adaptive_dct.dct_variants import LearnedDCTEmbedding
            self.patch_embed = LearnedDCTEmbedding(image_size, patch_size, dim, channels, freq_keep)
        else:
            self.patch_embed = PatchEmbedding(image_size, patch_size, dim, channels)

        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # One embedding vector per grid cell; re-ordered with tokens at forward time.
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, self.grid_h, self.grid_w, dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        order: List[int] = (
            spiral_indices(self.grid_h, self.grid_w)
            if token_order == "spiral"
            else raster_indices(self.grid_h, self.grid_w)
        )
        # Non-persistent: re-derived on load, not saved in checkpoint.
        self.register_buffer(
            "token_order_idx",
            torch.tensor(order, dtype=torch.long),
            persistent=False,
        )

        use_gaussian_bias = gaussian_bias
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, mlp_dim, dropout, use_gaussian_bias=use_gaussian_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # Store config flags for forward logic and checkpoint inspection.
        self.token_order       = token_order
        self.gaussian_bias     = gaussian_bias
        self.spiral_pool       = spiral_pool
        self.spiral_pool_alpha = spiral_pool_alpha
        self.spiral_pool_mode  = spiral_pool_mode

        if spiral_pool:
            N = self.num_patches
            if spiral_pool_mode == "average":
                g = torch.ones(N, dtype=torch.float32) / N
            else:  # "gaussian"
                sigma = float(N) / 2.0
                j     = torch.arange(N, dtype=torch.float32)
                g     = torch.exp(-(j ** 2) / (2.0 * sigma ** 2))
                g     = g / g.sum()
            # Non-persistent: deterministic from N + mode, not saved in checkpoint.
            self.register_buffer("spiral_pool_weights", g, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise positional embeddings with small truncated-normal noise."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_cls, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the normalised CLS token embedding (pre-head).

        Identical to ``forward`` but stops before the classification head,
        returning a ``(B, dim)`` feature vector suitable for fusion models.

        Args:
            x: Batch of images of shape ``(B, C, H, W)``.

        Returns:
            CLS token features of shape ``(B, dim)``.
        """
        B = x.size(0)
        tokens = self.patch_embed(x)
        idx = self.token_order_idx
        tokens = tokens[:, idx, :]
        pos_patches = self.pos_embed_2d.view(1, self.num_patches, -1)[:, idx, :]
        tokens = tokens + pos_patches
        tokens = self.emb_dropout(tokens)
        cls_tok = self.cls_token.expand(B, 1, -1) + self.pos_embed_cls
        x = torch.cat([cls_tok, tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.spiral_pool:
            patch_tokens  = x[:, 1:]   # (B, N, dim) — still in spiral order
            w             = self.spiral_pool_weights.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
            spiral_pooled = (patch_tokens * w).sum(dim=1)  # (B, dim)
            alpha         = self.spiral_pool_alpha
            return (1.0 - alpha) * x[:, 0] + alpha * spiral_pooled

        return x[:, 0]  # (B, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the Spiral ViT.

        Args:
            x: Batch of images of shape ``(B, C, H, W)``.
                ``H`` and ``W`` must equal ``image_size``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        return self.head(self.forward_features(x))
