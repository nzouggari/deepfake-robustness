"""Computes 68 facial landmarks per real training face (face-alignment) and stores them as .npy
files mirroring --real_root. Required by the self-blended images.

Example:
    python tools/precompute_landmarks.py --real_root data/FF++/train/real --out_root data/FF++/train_landmarks
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.folders import list_images, load_rgb  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real_root", required=True)
    p.add_argument("--out_root", required=True)
    args = p.parse_args()

    import face_alignment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

    real_root, out_root = Path(args.real_root), Path(args.out_root)
    paths = list_images(real_root)
    missing = 0
    for n, path in enumerate(paths, 1):
        out = (out_root / path.relative_to(real_root)).with_suffix(".npy")
        if out.is_file():
            continue
        img = load_rgb(path)
        h, w = img.shape[:2]
        preds = fa.get_landmarks_from_image(img, detected_faces=[[0, 0, w, h]])
        if not preds:
            missing += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, preds[0].astype(np.float32)[:, :2])
        if n % 1000 == 0:
            print(f"{n}/{len(paths)}")
    print(f"{len(paths) - missing} landmark files, {missing} faces without landmarks -> {out_root}")


if __name__ == "__main__":
    main()
