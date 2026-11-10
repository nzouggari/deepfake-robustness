"""Prepares the files attached to the GitHub release.

  - detector weights without the optimizer state (only config, trained weights, epoch, val_auc);
  - calibrated sentinel (statistics of clean FF++ validation frames and threshold);
  - lists of videos of every split (identifiers only, no image);
  - SHA-256 checksums.

Example:
    python tools/prepare_release.py --ckpt clip_sbi=runs/clip_sbi/best.pt \\
        clip_sbi_robust_anchored=runs/robust_r3/last.pt freqcnn_sbi=runs/freqcnn_sbi/best.pt \\
        --sentinel runs/sentinel.json --data_root data --out release
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_lists(data_root: Path, out: Path) -> None:
    for dataset in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for split in ("train", "val", "test"):
            root = dataset / split
            if not root.is_dir():
                continue
            videos = sorted({str(f.parent.relative_to(root)) for f in root.rglob("*.png")})
            if videos:
                path = out / "splits" / f"{dataset.name}_{split}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(videos) + "\n")
                print(f"{path.name}: {len(videos)} videos")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="name=path")
    p.add_argument("--sentinel", required=True)
    p.add_argument("--data_root", default=None, help="Folder containing FF++/ and CelebDF/ frame folders")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for item in args.ckpt:
        name, path = item.split("=", 1)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        light = {k: ckpt[k] for k in ("config", "state", "epoch", "val_auc") if k in ckpt}
        torch.save(light, out / f"{name}.pt")
        n = sum(v.numel() for v in light["state"].values())
        print(f"{name}.pt: {n:,} stored parameters, epoch {light.get('epoch')}, val AUC {light.get('val_auc')}")
    shutil.copy(args.sentinel, out / "sentinel.json")
    if args.data_root:
        split_lists(Path(args.data_root), out)

    sums = {f.name: sha256(f) for f in sorted(out.glob("*")) if f.is_file()}
    (out / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2))
    print(f"Release files in {out}")


if __name__ == "__main__":
    main()
