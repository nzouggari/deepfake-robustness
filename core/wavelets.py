"""Differentiable 2D Haar transform in PyTorch.

For a 2 x 2 pixel block   a b
                          c d
the orthonormal Haar transform gives
    LL = (a + b + c + d) / 2
    LH = (a - b + c - d) / 2
    HL = (a + b - c - d) / 2
    HH = (a - b - c + d) / 2
"""

from typing import List, Tuple

import torch


def haar_dwt(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Haar level. x: (B, C, H, W) with even H and W. Returns LL, LH, HL, HH of size (B, C, H/2, W/2)."""
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) / 2
    lh = (a - b + c - d) / 2
    hl = (a + b - c - d) / 2
    hh = (a - b - c + d) / 2
    return ll, lh, hl, hh


def haar_idwt(ll, lh, hl, hh) -> torch.Tensor:
    a = (ll + lh + hl + hh) / 2
    b = (ll - lh + hl - hh) / 2
    c = (ll + lh - hl - hh) / 2
    d = (ll - lh - hl + hh) / 2
    B, C, h, w = ll.shape
    out = ll.new_zeros(B, C, 2 * h, 2 * w)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


def haar_wavedec(x: torch.Tensor, levels: int) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, ...]]]:
    """Multi-level decomposition. Returns (LL_final, details) with details[k] = (LH, HL, HH) of level k + 1."""
    details = []
    ll = x
    for _ in range(levels):
        ll, lh, hl, hh = haar_dwt(ll)
        details.append((lh, hl, hh))
    return ll, details
