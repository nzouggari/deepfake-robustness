"""Informed (masked) attack: full-band PGD restricted to the most textured half of the image.

The mask is computed on the clean image from the level-1 Haar detail energy of each 2 x 2 block,
summed over the colour channels; only blocks above the median are perturbed. The perturbation
therefore stays in the strongest detail coefficients, which the sentinel statistic discards.
The attack does not use the gradient of the sentinel.
"""

import torch
import torch.nn.functional as F

from core.wavelets import haar_dwt


def texture_mask(x: torch.Tensor, keep: float = 0.5) -> torch.Tensor:
    _, lh, hl, hh = haar_dwt(x)
    e = (lh ** 2 + hl ** 2 + hh ** 2).sum(1, keepdim=True)
    thr = torch.quantile(e.flatten(1), 1 - keep, dim=1).view(-1, 1, 1, 1)
    return F.interpolate((e >= thr).float(), scale_factor=2, mode="nearest")


def masked_pgd(model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 20,
               keep: float = 0.5) -> torch.Tensor:
    mask = texture_mask(x, keep)
    alpha = 2.5 * eps / steps
    delta = ((x + torch.empty_like(x).uniform_(-eps, eps) * mask).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(x + delta), y), delta)
        with torch.no_grad():
            delta = (delta + alpha * g.sign() * mask).clamp(-eps, eps) * mask
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()
