"""Detection metrics (AUC, EER, accuracy), perturbation visibility (PSNR, SSIM)
and binomial confidence intervals (Wilson, Newcombe).

Convention: labels FAKE = 0 / REAL = 1, score = P(real).
"""

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


def auc(labels, scores) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def eer(labels, scores) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def detection_metrics(labels, scores, threshold: float = 0.5) -> Dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    return {
        "n": int(len(labels)),
        "auc": auc(labels, scores),
        "eer": eer(labels, scores),
        "acc": float((preds == labels).mean()) if len(labels) else float("nan"),
    }


def video_level(frame_df: pd.DataFrame) -> pd.DataFrame:
    """Mean P(real) per video. Expects columns video_id, label, score."""
    return (frame_df.groupby("video_id", as_index=False)
            .agg(label=("label", "first"), score=("score", "mean"), n_frames=("score", "size")))


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for (B, 3, H, W) tensors in [0, 1]."""
    mse = (x - y).pow(2).flatten(1).mean(1).clamp_min(1e-12)
    return 10 * torch.log10(1.0 / mse)


def _gaussian_window(size=11, sigma=1.5, channels=3, device=None):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-image SSIM (11 x 11 Gaussian window, sigma = 1.5)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    w = _gaussian_window(channels=x.shape[1], device=x.device)
    pad = w.shape[-1] // 2
    groups = x.shape[1]
    mu_x = F.conv2d(x, w, padding=pad, groups=groups)
    mu_y = F.conv2d(y, w, padding=pad, groups=groups)
    sxx = F.conv2d(x * x, w, padding=pad, groups=groups) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=groups) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=groups) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


def wilson_interval(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_interval(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float, float]:
    """Difference p1 - p2 between independent proportions and its 95% Newcombe (hybrid Wilson) interval."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1)
    l2, u2 = wilson_interval(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
