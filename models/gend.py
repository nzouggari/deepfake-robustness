"""Differentiable wrapper around the public GenD detector (Yermakov et al., WACV 2026).

The official code (https://github.com/yermandy/GenD) must be cloned locally and its path
passed as gend_repo; the weights are downloaded from the Hugging Face hub
(yermandy/GenD_CLIP_L_14). The wrapper takes [0, 1] pixels, applies the CLIP normalization
and returns logits ordered as (FAKE, REAL), like our detectors.
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CLIP_MEAN, CLIP_STD


class GenDWrapper(nn.Module):
    def __init__(self, gend: nn.Module, flip: bool = False):
        super().__init__()
        self.m = gend
        self.flip = flip
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        if x01.shape[-1] != 224 or x01.shape[-2] != 224:
            x01 = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
        z = self.m((x01 - self.mean) / self.std)
        z = z if torch.is_tensor(z) else z.logits
        return z.flip(1) if self.flip else z


def load_gend(gend_repo: str, device, flip: bool = False, hf_id: str = "yermandy/GenD_CLIP_L_14") -> GenDWrapper:
    if gend_repo not in sys.path:
        sys.path.append(gend_repo)
    from src.hf.modeling_gend import GenD
    gend = GenD.from_pretrained(hf_id)
    for p in gend.parameters():
        p.requires_grad_(False)
    return GenDWrapper(gend, flip=flip).to(device).eval()


@torch.no_grad()
def orient_gend(model: GenDWrapper, images: torch.Tensor, labels: torch.Tensor, device) -> GenDWrapper:
    """Sets the output order on source-domain (FF++) frames so that index 1 is REAL."""
    from sklearn.metrics import roc_auc_score
    p_real = torch.cat([model(images[i:i + 32].to(device)).softmax(1)[:, 1].cpu()
                        for i in range(0, len(images), 32)])
    if roc_auc_score(labels.numpy(), p_real.numpy()) < 0.5:
        model.flip = not model.flip
    return model
