"""CLIP-SBI detector: CLIP ViT-L/14 (OpenAI weights) tuned on its LayerNorm parameters, with two heads.

    image [0, 1] -> CLIP normalization -> CLIP visual encoder
        global token (768) -> L2 normalization -> linear head -> logits (FAKE, REAL)
        patch tokens (16 x 16 x 1024) -> linear head -> forgery map (1 x 16 x 16)

Trained parameters: LayerNorm (about 0.10 M) and heads. With train_blocks = N, the last N
Transformer blocks are also trained (anchored robust configuration: N = 2).
"""

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CLIP_MEAN, CLIP_STD


class CLIPDetector(nn.Module):
    def __init__(self, clip_model: str = "ViT-L-14-quickgelu", pretrained: str = "openai",
                 map_head: bool = True, train_blocks: int = 0):
        super().__init__()
        self.train_blocks = int(train_blocks)
        clip = open_clip.create_model(clip_model, pretrained=pretrained or None)
        self.visual = clip.visual
        self.visual.output_tokens = True
        del clip

        for p in self.visual.parameters():
            p.requires_grad = False
        for module in self.visual.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True
        if self.train_blocks > 0:
            for block in self.visual.transformer.resblocks[-self.train_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        self.head = nn.Linear(self.visual.output_dim, 2)
        self.map_head = nn.Linear(self.visual.transformer.width, 1) if map_head else None

        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x01: torch.Tensor, return_map: bool = False, return_feat: bool = False):
        """return_feat appends the normalized global embedding (B, 768), used by the anchoring loss."""
        pooled, tokens = self.visual((x01 - self.mean) / self.std)
        feat = F.normalize(pooled, dim=-1)
        logits = self.head(feat)
        out = [logits]
        if return_map:
            if self.map_head is None:
                raise RuntimeError("This model was built without a map head.")
            b, n, _ = tokens.shape
            side = int(round(n ** 0.5))
            out.append(self.map_head(tokens).transpose(1, 2).reshape(b, 1, side, side))
        if return_feat:
            out.append(feat)
        return out[0] if len(out) == 1 else tuple(out)

    def checkpoint_state(self) -> dict:
        """Only trained parameters are stored; the frozen CLIP weights are downloaded by open_clip."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in trainable}
