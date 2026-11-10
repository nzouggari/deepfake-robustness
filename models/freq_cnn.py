"""FreqCNN-SBI: frequency-aware convolutional detector following the principle of FSBI
(Abul Hasanaath et al., Image and Vision Computing, 2025). This is our re-implementation,
not the official FSBI code.

    input = [ RGB image , (image + Symlet-4 level-1 approximation) / 2 ]  -> 6 channels
    backbone = EfficientNet-B4 (torchvision), fully trained
"""

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import IMAGENET_MEAN, IMAGENET_STD


class FreqCNN(nn.Module):
    def __init__(self, arch: str = "efficientnet_b4", wavelet: str = "sym4", pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, arch)(weights=weights)

        conv = net.features[0][0]
        new_conv = nn.Conv2d(6, conv.out_channels, conv.kernel_size, conv.stride,
                             conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            new_conv.weight.copy_(torch.cat([conv.weight, conv.weight], dim=1) / 2)
        net.features[0][0] = new_conv
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
        self.net = net

        lo = torch.tensor(pywt.Wavelet(wavelet).dec_lo[::-1], dtype=torch.float32)
        kernel = (lo[:, None] * lo[None, :]).expand(3, 1, len(lo), len(lo)).contiguous()
        self.register_buffer("lowpass", kernel, persistent=False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def approximation(self, x01: torch.Tensor) -> torch.Tensor:
        """Level-1 approximation coefficients, upsampled to the image size."""
        k = self.lowpass.shape[-1]
        pad = (k // 2, k - 1 - k // 2, k // 2, k - 1 - k // 2)
        ll = F.conv2d(F.pad(x01, pad, mode="reflect"), self.lowpass, stride=2, groups=3) / 2
        return F.interpolate(ll, size=x01.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x01: torch.Tensor, return_map: bool = False):
        if return_map:
            raise RuntimeError("FreqCNN has no map head: use --lambda_map 0.")
        mixed = (x01 + self.approximation(x01)) / 2
        inp = torch.cat([(x01 - self.mean) / self.std, (mixed - self.mean) / self.std], dim=1)
        return self.net(inp)

    def checkpoint_state(self) -> dict:
        return self.state_dict()
