"""Smoothed (low-frequency) attack and the variants used to quantify implementation flaws.

smoothed_pgd (corrected attack, used for all evaluations):
    x' = clip(x + G_sigma(u), 0, 1),  u <- clip(u + alpha * sign(grad_u CE), -eps, eps)
The filter is applied once to the latent variable u, never to an already filtered
perturbation; since the Gaussian kernel is positive and sums to one,
||G_sigma(u)||_inf <= ||u||_inf <= eps.

Flaw variants (Table "implementation flaws"):
    initial    attacks.pgd.pgd_attack(band="gauss2"): sign of the raw gradient, then filter;
               filter re-applied to the accumulated perturbation at every step (flaws 2 and 3)
    flaw3_fixed  psign_pgd: filtered gradient, then sign; filter still re-applied (flaw 2 only)
"""

import torch
import torch.nn.functional as F

from attacks.bands import BandProjector


def smoothed_pgd(model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 20,
                 band: str = "gauss2") -> torch.Tensor:
    G = BandProjector(band)
    alpha = 2.5 * eps / steps
    u = torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        u.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model((x + G(u)).clamp(0, 1)), y), u)
        with torch.no_grad():
            u = (u + alpha * g.sign()).clamp(-eps, eps)
    with torch.no_grad():
        return (x + G(u)).clamp(0, 1)


def psign_pgd(model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 20,
              band: str = "gauss2") -> torch.Tensor:
    P = BandProjector(band)
    alpha = 2.5 * eps / steps
    delta = P(torch.empty_like(x).uniform_(-eps, eps)).clamp(-eps, eps)
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(x + delta), y), delta)
        with torch.no_grad():
            delta = P((delta + alpha * P(P(g).sign())).clamp(-eps, eps))
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()
