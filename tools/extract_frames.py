"""Extracts N evenly spaced frames per video, crops the face with MTCNN and saves 224 x 224 PNG files.

The train / val / test split is made per identity, never per frame:
  - --random_split TRAIN VAL TEST assigns each identity (first block of the file name before "_")
    with a deterministic hash, so that a manipulated video "071_054" follows its target "071";
  - --split test puts every video in one split (Celeb-DF);
  - --ffpp_split_dir uses the official FF++ split files.
The exact lists of videos used in the paper are given in splits/.

Output:
    <out_root>/<split>/<real|fake>[/<method>]/<video_name>/frame_00000.png

Examples:
    python tools/extract_frames.py --videos_dir raw/FF++/original_sequences/youtube/c23/videos \\
        --label real --random_split 0.7 0.15 0.15 --out_root data/FF++ --frames 8
    python tools/extract_frames.py --videos_dir raw/Celeb-DF-v2/Celeb-real raw/Celeb-DF-v2/YouTube-real \\
        --label real --split test --video_list raw/Celeb-DF-v2/List_of_testing_videos.txt --out_root data/CelebDF --frames 8
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def ffpp_split_table(split_dir: Path) -> dict:
    table = {}
    for split in ("train", "val", "test"):
        for pair in json.loads((split_dir / f"{split}.json").read_text()):
            for vid in pair:
                table[vid] = split
    return table


def random_split(stem: str, ratios) -> str:
    key = stem.split("_")[0]
    u = int(hashlib.md5(key.encode()).hexdigest(), 16) / 16 ** 32
    if u < ratios[0]:
        return "train"
    return "val" if u < ratios[0] + ratios[1] else "test"


def crop_face(frame_rgb, mtcnn, margin: float, size: int):
    boxes, _ = mtcnn.detect(frame_rgb)
    if boxes is None:
        return None
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
    h, w = frame_rgb.shape[:2]
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
    face = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA) if face.size else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_dir", nargs="+", required=True)
    p.add_argument("--label", choices=["real", "fake"], required=True)
    p.add_argument("--method", default=None, help="Optional sub-folder under fake/ (e.g. Deepfakes)")
    p.add_argument("--out_root", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ffpp_split_dir")
    g.add_argument("--split", choices=["train", "val", "test"])
    g.add_argument("--random_split", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--video_list", default=None, help="Text file of videos to keep (Celeb-DF official list accepted)")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--margin", type=float, default=0.3)
    args = p.parse_args()

    from facenet_pytorch import MTCNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device)

    table = ffpp_split_table(Path(args.ffpp_split_dir)) if args.ffpp_split_dir else None
    keep = None
    if args.video_list:
        keep = {Path(line.split()[-1]).stem for line in Path(args.video_list).read_text().splitlines() if line.strip()}

    videos = sorted(v for d in args.videos_dir for v in Path(d).rglob("*") if v.suffix.lower() in VIDEO_EXTS)
    if keep is not None:
        videos = [v for v in videos if v.stem in keep]
    print(f"{len(videos)} videos")

    for n, video in enumerate(videos, 1):
        stem = video.stem
        if table is not None:
            split = table.get(stem.split("_")[0])
            if split is None:
                continue
        else:
            split = args.split or random_split(stem, args.random_split)

        out_dir = Path(args.out_root) / split / args.label
        if args.method and args.label == "fake":
            out_dir = out_dir / args.method
        out_dir = out_dir / stem
        if out_dir.is_dir() and len(list(out_dir.glob("*.png"))) >= args.frames:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        for idx in np.linspace(0, max(total - 1, 0), args.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            face = crop_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mtcnn, args.margin, args.size)
            if face is None:
                continue
            cv2.imwrite(str(out_dir / f"frame_{idx:05d}.png"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            saved += 1
        cap.release()
        print(f"[{n}/{len(videos)}] {split}/{args.label}/{stem}: {saved} frames")


if __name__ == "__main__":
    main()
