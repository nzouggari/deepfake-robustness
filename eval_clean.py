"""Clean-image evaluation at frame and video level (Table "AUC on clean images").

Video-level score: mean P(real) over the frames of each video.

Example:
    python eval_clean.py --ckpt weights/clip_sbi.pt --data ffpp=data/FF++/test celebdf=data/CelebDF/test --out runs/clean_clip_sbi
    python eval_clean.py --gend_repo GenD --data ffpp=data/FF++/test celebdf=data/CelebDF/test --out runs/clean_gend

Outputs: <name>_frames.csv, <name>_videos.csv, summary.json.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.metrics import detection_metrics, video_level
from core.utils import default_device, save_json
from data.folders import FrameFolder
from models.build import load_detector


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    rows = []
    for x, y, vid, group in loader:
        score = torch.softmax(model(x.to(device, non_blocking=True)).float(), 1)[:, 1].cpu().numpy()
        for i in range(len(y)):
            rows.append({"video_id": vid[i], "group": group[i], "label": int(y[i]), "score": float(score[i])})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--gend_repo", help="Path to a local clone of https://github.com/yermandy/GenD")
    p.add_argument("--data", nargs="+", required=True, help="name=path, the first one is the source domain (FF++)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = default_device()
    sets = [item.split("=", 1) for item in args.data]
    if args.ckpt:
        model, _ = load_detector(args.ckpt, device)
    else:
        from models.gend import load_gend, orient_gend
        model = load_gend(args.gend_repo, device)
        xs, ys = zip(*[(x, y) for x, y, _, _ in DataLoader(FrameFolder(sets[0][1], 100, seed=0), batch_size=128)])
        model = orient_gend(model, torch.cat(xs), torch.cat(ys), device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"ckpt": args.ckpt or "GenD"}
    for name, path in sets:
        loader = DataLoader(FrameFolder(path), batch_size=args.batch, num_workers=args.workers)
        frames = predict(model, loader, device)
        videos = video_level(frames)
        frames.to_csv(out / f"{name}_frames.csv", index=False)
        videos.to_csv(out / f"{name}_videos.csv", index=False)
        summary[name] = {"frame": detection_metrics(frames.label, frames.score),
                         "video": detection_metrics(videos.label, videos.score)}
        f, v = summary[name]["frame"], summary[name]["video"]
        print(f"{name:>10} | frames {f['n']:6d} AUC {f['auc']:.4f} | videos {v['n']:5d} AUC {v['auc']:.4f}")
    save_json(summary, out / "summary.json")


if __name__ == "__main__":
    main()
