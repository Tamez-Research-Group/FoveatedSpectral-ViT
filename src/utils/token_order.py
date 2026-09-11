"""
src/utils/token_order.py
========================
Patch traversal order utilities for the Spiral ViT.

Two orderings are provided and used as the ``token_order`` axis in ablation
experiments:

``spiral_indices``
    Visits an h × w patch grid starting from the spatial centre and expanding
    outward.  When combined with the Gaussian inductive bias in
    ``MultiHeadSelfAttention`` (which amplifies attention to low-index tokens),
    early sequence positions correspond to central patches — providing a soft
    spatial prior that biases the model toward diagnostically important central
    regions.  This prior is hypothesis-driven: pathology in retinal fundus
    photographs (glaucoma, diabetic retinopathy), dermoscopy images (melanoma),
    and brain MRI (tumour core) is typically framed at or near the centre of
    the field of view.

``raster_indices``
    Standard left-to-right, top-to-bottom traversal — the ordering used in
    the original ViT paper.  Serves as the baseline/control condition in
    ablation experiments.

Both functions return a flat list of row-major linear indices suitable for
indexing a ``(num_patches,)`` or ``(B, num_patches, dim)`` token tensor.
"""

from __future__ import annotations
from typing import List, Tuple


def spiral_indices(h: int, w: int) -> List[int]:
    """Return a centre-outward spiral traversal of an h × w patch grid.

    The traversal starts at the grid centre and expands outward by cycling
    through four cardinal directions (right → down → left → up).  Step
    lengths increase every two direction changes, producing the standard
    "square spiral" unwinding pattern.

    For even-sized grids the centre is defined as the upper-left cell of the
    2 × 2 central block: ``r = (h-1)//2, c = (w-1)//2``.

    Args:
        h: Number of patch rows (grid height).
        w: Number of patch columns (grid width).

    Returns:
        A list of length ``h * w`` containing flat row-major indices
        (``index = row * w + col``) in spiral visit order.  Index 0 in the
        returned list is the centre patch; the last index is a corner patch.

    Example:
        For a 3 × 3 grid::

            Flat indices:  0 1 2
                           3 4 5
                           6 7 8

            Spiral order: [4, 5, 8, 7, 6, 3, 0, 1, 2]
            (starts at centre patch 4, expands outward)
    """
    assert h > 0 and w > 0

    r = (h - 1) // 2
    c = (w - 1) // 2

    visited: List[int] = []
    seen: set = set()

    def add(rr: int, cc: int) -> None:
        if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen:
            seen.add((rr, cc))
            visited.append(rr * w + cc)

    add(r, c)

    # Directions: right, down, left, up — standard counter-clockwise spiral.
    dirs: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    step = 1

    while len(visited) < h * w:
        for d in range(4):
            dr, dc = dirs[d]
            for _ in range(step):
                r += dr
                c += dc
                add(r, c)
                if len(visited) == h * w:
                    return visited
            # Step length grows every two direction changes.
            if d % 2 == 1:
                step += 1

    return visited


def raster_indices(h: int, w: int) -> List[int]:
    """Return standard row-major (raster) traversal indices for an h × w grid.

    Patches are visited left-to-right, top-to-bottom, yielding the identity
    permutation ``[0, 1, ..., h*w-1]``.  This is the ordering used in the
    original Vision Transformer paper and serves as the baseline condition.

    Args:
        h: Number of patch rows.  Present for API symmetry with
            :func:`spiral_indices`; the value does not affect the output.
        w: Number of patch columns.  Same note as ``h``.

    Returns:
        ``[0, 1, ..., h*w-1]``.
    """
    return list(range(h * w))
