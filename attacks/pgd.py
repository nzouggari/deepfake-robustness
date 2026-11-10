"""L-infinity PGD in [0, 1] pixel space.

pgd_attack is used
  - as the inner attack of adversarial training (band-restricted, train.py);
  - as the full-band attack of the naive attacker (band = "full", lam = 0);
  - as the attack of the adaptive attacker (band = "full", lam > 0), whose objective is
        CE(model(x + delta), y) - lam * softplus(soft_score(x + delta) - threshold).

Step: delta <- P(delta + alpha * P(sign(grad))), clipped to [-eps, eps] and to the image domain.
With a Gaussian band, P is re-applied at every step: this is the initial implementation
used during training (flaws 2 and 3 of the paper); the corrected smoothed attack used
for evaluation is attacks/smoothed.py.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from attacks.bands import BandProjector


def pgd_attack(model, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 20,
               alpha: Optional[float] = None, band: str = "full", random_start: bool = True,
               sentinel=None, lam: float = 0.0) -> torch.Tensor:
    project = BandProjector(band)
    alpha = alpha if alpha is not None else 2.5 * eps / steps

    delta = torch.zeros_like(x)
    if random_start:
        delta = project(torch.empty_like(x).uniform_(-eps, eps)).clamp(-eps, eps)
    delta = ((x + delta).clamp(0, 1) - x).detach()

    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = x + delta
        loss = F.cross_entropy(model(x_adv), y)
        if sentinel is not None and lam > 0:
            loss = loss - lam * F.softplus(sentinel.soft_score(x_adv) - sentinel.threshold).mean()
        grad, = torch.autograd.grad(loss, delta)

        with torch.no_grad():
            delta = delta + alpha * project(grad.sign())
            delta = project(delta.clamp(-eps, eps))
            if not project.norm_preserving:
                delta = delta.clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()
