"""Orthonormal 2D DCT-II on the whole image, by matrix multiplication.

    X = D x D^T      (dct2)
    x = D^T X D      (idct2)
"""

import math

import torch


def dct_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    k = torch.arange(n, dtype=dtype, device=device).unsqueeze(1)
    i = torch.arange(n, dtype=dtype, device=device).unsqueeze(0)
    mat = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    mat[0] *= 1 / math.sqrt(2)
    return mat * math.sqrt(2 / n)


def dct2(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return d @ x @ d.T


def idct2(coeffs: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return d.T @ coeffs @ d


def low_frequency_mask(n: int, keep_ratio: float, device=None) -> torch.Tensor:
    """(n, n) mask keeping the low-frequency square of side keep_ratio * n."""
    k = max(1, int(round(keep_ratio * n)))
    mask = torch.zeros(n, n, device=device)
    mask[:k, :k] = 1.0
    return mask
