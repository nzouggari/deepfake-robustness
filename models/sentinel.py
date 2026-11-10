"""Training-free wavelet sentinel.

For an image x in [0, 1]:
  1. two Haar levels are computed on the three channels;
  2. for each detail sub-band, f = log(E_rho + eta), where E_rho is the mean of the
     fraction rho (keep = 0.5) of smallest squared coefficients, i.e. the smooth regions
     of the image, and eta = (0.5 / 255)^2 is a quantization-level floor;
  3. level 3 uses second differences of LL2 (horizontal, vertical, diagonal), each
     expressed relative to the level-2 feature of the same orientation;
  4. the nine features are standardized with statistics of clean FF++ validation frames;
  5. score = max of the nine z-scores; the image is rejected if score > threshold, the
     threshold being the (1 - target FPR) quantile of clean calibration scores.

soft_score is the smooth maximum (log-sum-exp) used by the adaptive attacker.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.wavelets import haar_dwt, haar_wavedec

BANDS = ("LH", "HL", "HH")


class WaveletSentinel(nn.Module):
    def __init__(self, levels: int = 3, stat: str = "lowfrac", keep: float = 0.5,
                 floor: float = (0.5 / 255) ** 2, shifts=((0, 0),), l3: str = "laplace"):
        super().__init__()
        if stat not in ("mean", "lowfrac"):
            raise ValueError(f"Unknown statistic: {stat}")
        if l3 not in ("haar", "laplace"):
            raise ValueError(f"Unknown level-3 feature: {l3}")
        if l3 == "laplace" and levels != 3:
            raise ValueError("l3 = laplace requires 3 levels")
        self.levels = levels
        self.stat = stat
        self.keep = float(keep)
        self.floor = float(floor)
        self.l3 = l3
        self.shifts = [tuple(int(v) for v in sh) for sh in shifts]
        n = 3 * levels * len(self.shifts)
        self.register_buffer("mu", torch.zeros(n))
        self.register_buffer("sigma", torch.ones(n))
        self.register_buffer("threshold", torch.tensor(float("inf")))
        self.target_fpr = None

    @property
    def feature_names(self):
        names = [f"L{l + 1}_{b}" for l in range(self.levels) for b in BANDS]
        if len(self.shifts) == 1:
            return names
        return [f"{n}@d{dy}{dx}" for dy, dx in self.shifts for n in names]

    def features(self, x01: torch.Tensor) -> torch.Tensor:
        """(B, 3 * levels) log-energies."""
        x01 = x01.float()
        h, w = x01.shape[-2:]
        feats = []
        for dy, dx in self.shifts:
            xs = x01 if (dy, dx) == (0, 0) else F.pad(x01, (dx, 0, dy, 0), mode="replicate")[..., :h, :w]
            if self.l3 == "haar":
                _, details = haar_wavedec(xs, self.levels)
                feats += [torch.log(self._energy(band) + self.floor) for level in details for band in level]
                continue
            ll1, *d1 = haar_dwt(xs)
            ll2, *d2 = haar_dwt(ll1)
            f12 = [torch.log(self._energy(band) + self.floor) for band in (*d1, *d2)]
            d2_h = (ll2[..., :, :-2] - 2 * ll2[..., :, 1:-1] + ll2[..., :, 2:]) / 4
            d2_v = (ll2[..., :-2, :] - 2 * ll2[..., 1:-1, :] + ll2[..., 2:, :]) / 4
            d2_d = (ll2[..., :-1, :-1] - ll2[..., :-1, 1:] - ll2[..., 1:, :-1] + ll2[..., 1:, 1:]) / 2
            f3 = [torch.log(self._energy(m) + self.floor) - f12[3 + i] for i, m in enumerate((d2_h, d2_v, d2_d))]
            feats += f12 + f3
        return torch.stack(feats, dim=1)

    def _energy(self, band: torch.Tensor) -> torch.Tensor:
        sq = band.pow(2).flatten(start_dim=1)
        if self.stat == "mean":
            return sq.mean(dim=1)
        k = max(1, int(round(self.keep * sq.shape[1])))
        return torch.topk(sq, k=k, dim=1, largest=False).values.mean(dim=1)

    def zscores(self, x01: torch.Tensor) -> torch.Tensor:
        return (self.features(x01) - self.mu) / self.sigma

    def score_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return ((feats - self.mu) / self.sigma).max(dim=1).values

    def score(self, x01: torch.Tensor) -> torch.Tensor:
        return self.score_from_features(self.features(x01))

    def soft_score(self, x01: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
        return tau * torch.logsumexp(self.zscores(x01) / tau, dim=1)

    def reject(self, x01: torch.Tensor) -> torch.Tensor:
        return self.score(x01) > self.threshold

    @torch.no_grad()
    def calibrate(self, clean_features: torch.Tensor, target_fpr: float = 0.05) -> None:
        self.mu.copy_(clean_features.mean(0))
        self.sigma.copy_(clean_features.std(0).clamp_min(1e-6))
        scores = self.score_from_features(clean_features)
        self.threshold.fill_(torch.quantile(scores, 1 - target_fpr).item())
        self.target_fpr = target_fpr

    def to_dict(self) -> dict:
        return {"levels": self.levels, "stat": self.stat, "keep": self.keep,
                "floor": self.floor, "shifts": [list(sh) for sh in self.shifts], "l3": self.l3,
                "feature_names": self.feature_names,
                "mu": self.mu.tolist(), "sigma": self.sigma.tolist(),
                "threshold": float(self.threshold), "target_fpr": self.target_fpr}

    @classmethod
    def from_dict(cls, d: dict) -> "WaveletSentinel":
        s = cls(levels=int(d["levels"]), stat=d.get("stat", "mean"), keep=d.get("keep", 0.5),
                floor=d.get("floor", 1e-12), shifts=d.get("shifts", [[0, 0]]), l3=d.get("l3", "haar"))
        s.mu.copy_(torch.tensor(d["mu"]))
        s.sigma.copy_(torch.tensor(d["sigma"]))
        s.threshold.fill_(float(d["threshold"]))
        s.target_fpr = d.get("target_fpr")
        return s
