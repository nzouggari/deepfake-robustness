"""White-box attacks against a detector and the sentinel placed in front of it.

Protocol (Section 4 of the paper):
  - 100 real and 100 fake frames drawn from each test set with a fixed seed (identical for all detectors);
  - only frames correctly classified before the attack are attacked;
  - L-infinity PGD, 20 iterations, step 2.5 eps / 20, uniform random start, cross-entropy, float32;
  - reported: ASR, sentinel rejection rate, bypass rate (misclassified and accepted), PSNR,
    with 95% Wilson intervals.

Attacks:
    smoothed   corrected smoothed attack, delta = G_2(u)                     (attacks/smoothed.py)
    full       full-band PGD, naive attacker                                 (attacks/pgd.py)
    adaptive   full-band PGD penalizing the sentinel score, weight --lam     (attacks/pgd.py)
    masked     full-band PGD on the most textured half of the image          (attacks/masked.py)
    initial    initial smoothed attack, filter re-applied at each step       (flaws 2 and 3)
    psign      filtered gradient then sign, filter re-applied at each step   (flaw 2 only)

Example:
    python eval_attacks.py --ckpt weights/clip_sbi_robust_anchored.pt --sentinel weights/sentinel.json \\
        --data ffpp=data/FF++/test --attack smoothed --eps 1 2 4 --out runs/attacks/r3
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from attacks.masked import masked_pgd
from attacks.pgd import pgd_attack
from attacks.smoothed import psign_pgd, smoothed_pgd
from core.metrics import psnr, wilson_interval
from core.utils import default_device, seed_everything
from data.folders import FrameFolder
from models.build import load_detector, load_sentinel


def load_subset(path, n_per_class):
    data = [(x, y, v) for x, y, v, _ in DataLoader(FrameFolder(path, n_per_class, seed=0), batch_size=128)]
    xs, ys, vs = zip(*data)
    return torch.cat(xs), torch.cat(ys), [v for batch in vs for v in batch]


def run_attack(name, model, x, y, eps, steps, sentinel, lam):
    if name == "smoothed":
        return smoothed_pgd(model, x, y, eps, steps)
    if name == "full":
        return pgd_attack(model, x, y, eps, steps=steps, band="full")
    if name == "adaptive":
        return pgd_attack(model, x, y, eps, steps=steps, band="full", sentinel=sentinel, lam=lam)
    if name == "masked":
        return masked_pgd(model, x, y, eps, steps)
    if name == "initial":
        return pgd_attack(model, x, y, eps, steps=steps, band="gauss2")
    if name == "psign":
        return psign_pgd(model, x, y, eps, steps)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--gend_repo", help="Path to a local clone of https://github.com/yermandy/GenD")
    p.add_argument("--gend_orient_dir", default=None, help="FF++ test folder used to fix GenD's output order")
    p.add_argument("--sentinel", required=True)
    p.add_argument("--data", required=True, help="name=path")
    p.add_argument("--attack", choices=["smoothed", "full", "adaptive", "masked", "initial", "psign"], required=True)
    p.add_argument("--eps", nargs="+", type=float, default=[4.0], help="In units of 1/255")
    p.add_argument("--lam", nargs="+", type=float, default=[0.0], help="Sentinel penalty weight (adaptive attack)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--n_per_class", type=int, default=100)
    p.add_argument("--batch", type=int, default=None, help="Default: 16 for FreqCNN, 8 otherwise")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = default_device()
    name, path = args.data.split("=", 1)
    if args.ckpt:
        model, ckpt = load_detector(args.ckpt, device)
        detector = Path(args.ckpt).stem
        batch = args.batch or (16 if ckpt["config"]["model"] == "freqcnn" else 8)
    else:
        from models.gend import load_gend, orient_gend
        model = load_gend(args.gend_repo, device)
        xo, yo, _ = load_subset(args.gend_orient_dir or path, 100)
        model = orient_gend(model, xo, yo, device)
        detector = "gend"
        batch = args.batch or 8
    for prm in model.parameters():
        prm.requires_grad_(False)
    sentinel = load_sentinel(args.sentinel, device)

    x_all, y_all, v_all = load_subset(path, args.n_per_class)
    with torch.no_grad():
        ok = torch.cat([model(x_all[i:i + 32].to(device)).argmax(1).cpu() for i in range(0, len(x_all), 32)]) == y_all
    idx = ok.nonzero().flatten()
    xk, yk = x_all[idx], y_all[idx]
    vk = [v_all[i] for i in idx.tolist()]
    print(f"{detector} | {name}: {len(idx)}/{len(ok)} correctly classified frames are attacked")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for eps255 in args.eps:
        for lam in (args.lam if args.attack == "adaptive" else [0.0]):
            seed_everything(0)
            fooled, rejected, quality = [], [], []
            for i in range(0, len(xk), batch):
                x, y = xk[i:i + batch].to(device), yk[i:i + batch].to(device)
                xa = run_attack(args.attack, model, x, y, eps255 / 255, args.steps, sentinel, lam)
                with torch.no_grad():
                    fooled += (model(xa).argmax(1) != y).cpu().tolist()
                    rejected += sentinel.reject(xa).cpu().tolist()
                    quality += psnr(xa, x).cpu().tolist()
            f, r = torch.tensor(fooled), torch.tensor(rejected)
            bypass = f & ~r
            tag = f"{args.attack}_eps{eps255:g}" + (f"_lam{lam:g}" if args.attack == "adaptive" else "")
            pd.DataFrame({"video_id": vk, "label": yk.tolist(), "fooled": f.tolist(), "rejected": r.tolist(),
                          "bypass": bypass.tolist(), "psnr": quality}).to_csv(out / f"{name}_{tag}_frames.csv", index=False)
            n = len(f)
            row = {"detector": detector, "dataset": name, "attack": args.attack, "eps_255": eps255, "lam": lam,
                   "steps": args.steps, "n": n, "asr": f.float().mean().item(),
                   "reject_rate": r.float().mean().item(), "bypass_rate": bypass.float().mean().item(),
                   "psnr": sum(quality) / n}
            row["asr_ci_low"], row["asr_ci_high"] = wilson_interval(int(f.sum()), n)
            row["bypass_ci_low"], row["bypass_ci_high"] = wilson_interval(int(bypass.sum()), n)
            summary.append(row)
            print(f"{tag:>22} | ASR {100 * row['asr']:.1f} % [{100 * row['asr_ci_low']:.1f}, {100 * row['asr_ci_high']:.1f}] "
                  f"| rejection {100 * row['reject_rate']:.1f} % | bypass {100 * row['bypass_rate']:.1f} % "
                  f"| PSNR {row['psnr']:.1f} dB | n = {n}")
    pd.DataFrame(summary).to_csv(out / f"{name}_{args.attack}_summary.csv", index=False)


if __name__ == "__main__":
    main()
