"""Calibrates the wavelet sentinel on clean source-domain frames (FF++ validation, real and fake).
No Celeb-DF frame and no adversarial example is used: the threshold is fixed before any test.

Example:
    python calibrate_sentinel.py --clean_dir data/FF++/val --out runs/sentinel.json
"""

import argparse

import torch
from torch.utils.data import DataLoader

from core.utils import default_device, save_json
from data.folders import FrameFolder
from models.sentinel import WaveletSentinel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean_dir", required=True)
    p.add_argument("--keep", type=float, default=0.5, help="Fraction rho of smallest coefficients")
    p.add_argument("--target_fpr", type=float, default=0.05)
    p.add_argument("--max_per_class", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = default_device()
    sentinel = WaveletSentinel(levels=3, stat="lowfrac", keep=args.keep, floor=(0.5 / 255) ** 2,
                               l3="laplace").to(device)
    loader = DataLoader(FrameFolder(args.clean_dir, args.max_per_class), batch_size=128, num_workers=args.workers)

    feats = []
    with torch.no_grad():
        for x, _, _, _ in loader:
            feats.append(sentinel.features(x.to(device)).cpu())
    feats = torch.cat(feats).to(device)
    sentinel.calibrate(feats, args.target_fpr)

    d = sentinel.to_dict()
    d["n_images"] = len(feats)
    d["clean_reject_rate"] = float((sentinel.score_from_features(feats) > sentinel.threshold).float().mean())
    save_json(d, args.out)
    print(f"Sentinel calibrated on {len(feats)} images | threshold = {d['threshold']:.3f} "
          f"| clean rejection = {d['clean_reject_rate']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
