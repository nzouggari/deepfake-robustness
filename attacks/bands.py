"""Frequency-band operators applied to a perturbation delta.

    full       no restriction
    gauss<s>   Gaussian filter of standard deviation s pixels      e.g. gauss1, gauss2, gauss4
    haar<L>    constant blocks of 2^L x 2^L pixels                 e.g. haar2, haar3
    dct<p>     keeps the low-frequency DCT square of side p % of the image   e.g. dct12, dct25

These operators are used by the inner attack of adversarial training (attacks/pgd.py).
Haar and DCT are exact projections; the Gaussian filter is not idempotent, which is why
the evaluation uses the parameterization delta = G_sigma(u) of attacks/smoothed.py.
"""

import math
import re

import torch
import torch.nn.functional as F

from core.dct import dct2, dct_matrix, idct2, low_frequency_mask


class BandProjector:
    def __init__(self, name: str):
        self.name = name
        m = re.fullmatch(r"(full|gauss|haar|dct)(\d*(?:\.\d+)?)", name)
        if not m:
            raise ValueError(f"Unknown band: {name}")
        self.kind, value = m.group(1), m.group(2)
        self.value = float(value) if value else None
        self.norm_preserving = self.kind in ("full", "gauss", "haar")
        self._cache = {}

    def __call__(self, delta: torch.Tensor) -> torch.Tensor:
        if self.kind == "full":
            return delta
        if self.kind == "gauss":
            return self._gauss(delta, self.value)
        if self.kind == "haar":
            f = 2 ** int(self.value)
            return F.interpolate(F.avg_pool2d(delta, f), scale_factor=f, mode="nearest")
        return self._dct(delta, self.value / 100)

    def _gauss(self, delta, sigma):
        key = ("g", sigma, delta.device, delta.dtype)
        if key not in self._cache:
            r = int(math.ceil(3 * sigma))
            t = torch.arange(-r, r + 1, device=delta.device, dtype=delta.dtype)
            g = torch.exp(-t ** 2 / (2 * sigma ** 2))
            self._cache[key] = (g / g.sum(), r)
        g, r = self._cache[key]
        c = delta.shape[1]
        out = F.conv2d(F.pad(delta, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        out = F.conv2d(F.pad(out, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
        return out

    def _dct(self, delta, keep_ratio):
        n = delta.shape[-1]
        key = ("d", n, keep_ratio, delta.device)
        if key not in self._cache:
            self._cache[key] = (dct_matrix(n, device=delta.device), low_frequency_mask(n, keep_ratio, delta.device))
        d, mask = self._cache[key]
        return idct2(dct2(delta.float(), d) * mask, d).to(delta.dtype)
