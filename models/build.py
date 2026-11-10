"""Building, saving and loading detectors.

A checkpoint contains:
    config  dictionary used to rebuild the model
    state   trained weights (LayerNorm, blocks and heads for CLIP; whole network for FreqCNN)
    epoch, val_auc
"""

from pathlib import Path

import torch

from core.utils import load_json
from models.sentinel import WaveletSentinel


def build_detector(cfg: dict):
    name = cfg["model"]
    if name == "clip":
        from models.clip_detector import CLIPDetector
        return CLIPDetector(clip_model=cfg.get("clip_model", "ViT-L-14-quickgelu"),
                            pretrained=cfg.get("pretrained", "openai"),
                            map_head=cfg.get("map_head", True),
                            train_blocks=cfg.get("train_blocks", 0))
    if name == "freqcnn":
        from models.freq_cnn import FreqCNN
        return FreqCNN(arch=cfg.get("arch", "efficientnet_b4"),
                       wavelet=cfg.get("wavelet", "sym4"),
                       pretrained=bool(cfg.get("pretrained", True)))
    raise ValueError(f"Unknown model: {name}")


def save_detector(path, model, cfg: dict, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state": model.checkpoint_state(), **extra}, path)


def load_detector(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_detector(ckpt["config"])
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    params = dict(model.named_parameters())
    not_restored = [k for k in missing if k in params and params[k].requires_grad]
    if not_restored:
        raise RuntimeError(f"Trained weights missing from checkpoint: {not_restored[:5]}")
    model.to(device).eval()
    print(f"[load] {path} | model={ckpt['config']['model']} | epoch={ckpt.get('epoch')} | val AUC={ckpt.get('val_auc')}")
    return model, ckpt


def load_sentinel(path, device):
    return WaveletSentinel.from_dict(load_json(path)).to(device).eval()
