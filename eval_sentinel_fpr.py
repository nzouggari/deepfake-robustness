"""Rejection rate of the sentinel on non-attacked frames, clean and after common processing
(Table "Rejection rate of the sentinel on non-attacked images").

Processing: JPEG quality 70 and 50; sharpening with an unsharp mask (radius 2 px, 150 %, threshold 0).

Example:
    python eval_sentinel_fpr.py --sentinel weights/sentinel.json --data ffpp=data/FF++/test celebdf=data/CelebDF/test --out runs/sentinel_fpr.csv
"""

import argparse
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from core.utils import default_device
from data.folders import FrameFolder
from models.build import load_sentinel


def _pil(x: torch.Tensor) -> Image.Image:
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def _tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255


def jpeg(q):
    def f(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return f


def sharpen(img):
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=0))


PROCESSING = {"clean": lambda img: img, "jpeg70": jpeg(70), "jpeg50": jpeg(50), "sharpen": sharpen}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sentinel", required=True)
    p.add_argument("--data", nargs="+", required=True, help="name=path")
    p.add_argument("--max_per_class", type=int, default=0, help="0 = all frames")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = default_device()
    sentinel = load_sentinel(args.sentinel, device)
    rows = []
    for item in args.data:
        name, path = item.split("=", 1)
        loader = DataLoader(FrameFolder(path, args.max_per_class), batch_size=64, num_workers=args.workers)
        rejected = {k: [] for k in PROCESSING}
        with torch.no_grad():
            for x, _, _, _ in loader:
                for key, fn in PROCESSING.items():
                    xp = torch.stack([_tensor(fn(_pil(im))) for im in x]).to(device)
                    rejected[key] += sentinel.reject(xp).cpu().tolist()
        for key, values in rejected.items():
            rows.append({"dataset": name, "processing": key, "n": len(values), "reject_rate": float(np.mean(values))})
            print(f"{name:>10} | {key:>8} | rejection = {np.mean(values):.3f}")
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
