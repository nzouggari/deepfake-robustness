"""Trains a detector on real FF++ faces and self-blended fakes.

Loss:
    L = CE(logits, label) + lambda_map * BCE(predicted map, target map)
The map head only exists for the CLIP detector (use --lambda_map 0 for freqcnn).
Model selection: best frame-level AUC on FF++ validation (real and Deepfakes frames,
never used in the gradient). Celeb-DF is never used here.

Anchored adversarial training (--adv_bands, --train_blocks, --anchor_ckpt):
    at each batch a band b is drawn from --adv_bands, a band-restricted PGD of --adv_steps
    steps (eps = --adv_eps / 255) produces x_adv, and the loss becomes
        (1 - w) CE(x) + w CE(x_adv) + lambda_map BCE(map(x)) + gamma ( ||e(x_adv) - e0(x)||^2 + ||e(x) - e0(x)||^2 )
    where e is the normalized global embedding and e0 that of the frozen anchor model.

Outputs in --out: best.pt, last.pt, history.csv, config.json.
See configs/ for the exact arguments of every model of the paper.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.metrics import auc
from core.utils import default_device, save_json, seed_everything, worker_init_fn
from data.folders import FrameFolder
from data.train_dataset import PristineGroupDataset, group_collate
from models.build import build_detector, load_detector, save_detector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_real", required=True, help="Real training faces (FF++/train/real)")
    p.add_argument("--landmarks", required=True, help="Landmark .npy files mirroring --train_real")
    p.add_argument("--val_dir", required=True, help="FF++/val with real/ and fake/")
    p.add_argument("--val_max_per_class", type=int, default=0)
    p.add_argument("--model", choices=["clip", "freqcnn"], default="clip")
    p.add_argument("--clip_model", default="ViT-L-14-quickgelu")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--arch", default="efficientnet_b4")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_groups", type=int, default=16, help="Real faces per batch (batch = 2 x groups)")
    p.add_argument("--lr", type=float, default=None, help="Default: 1e-4 (clip) or 3e-4 (freqcnn)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_map", type=float, default=0.5)
    p.add_argument("--jpeg_prob", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_ckpt", default=None)
    p.add_argument("--adv_bands", nargs="*", default=[])
    p.add_argument("--adv_eps", type=float, default=4.0, help="In units of 1/255")
    p.add_argument("--adv_steps", type=int, default=4)
    p.add_argument("--adv_weight", type=float, default=0.5)
    p.add_argument("--adv_random", action="store_true", help="Draw the band at random at each batch")
    p.add_argument("--train_blocks", type=int, default=0)
    p.add_argument("--block_lr", type=float, default=None, help="Default: lr / 5")
    p.add_argument("--anchor_ckpt", default=None)
    p.add_argument("--anchor_weight", type=float, default=1.0)
    p.add_argument("--monitor_n", type=int, default=0, help="Validation images attacked by a long PGD at each epoch")
    p.add_argument("--monitor_band", default="gauss2")
    p.add_argument("--monitor_steps", type=int, default=20)
    p.add_argument("--resume", action="store_true", help="Resume from <out>/last.pt")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device, amp: bool, amp_dtype=torch.float16) -> float:
    model.eval()
    labels, scores = [], []
    for x, y, _, _ in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        scores += torch.softmax(logits.float(), 1)[:, 1].cpu().tolist()
        labels += y.tolist()
    return auc(labels, scores)


def adv_monitor(model, loader, device, eps: float, steps: int, band: str) -> float:
    """Accuracy under a long float32 PGD, to detect catastrophic overfitting (Wong et al., 2020)."""
    from attacks.pgd import pgd_attack
    model.eval()
    correct, total = 0, 0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y, eps, steps=steps, band=band)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = default_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = {"model": args.model, "clip_model": args.clip_model, "pretrained": args.pretrained,
           "map_head": args.lambda_map > 0, "arch": args.arch}
    if args.train_blocks:
        cfg["train_blocks"] = args.train_blocks
    if args.model == "freqcnn" and args.lambda_map > 0:
        raise SystemExit("freqcnn has no map head: use --lambda_map 0")
    if args.model != "clip" and (args.train_blocks or args.anchor_ckpt):
        raise SystemExit("--train_blocks and --anchor_ckpt only apply to the CLIP detector")
    if args.anchor_ckpt and not args.adv_bands:
        raise SystemExit("--anchor_ckpt requires adversarial training (--adv_bands ...)")
    lr = args.lr or (1e-4 if args.model == "clip" else 3e-4)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    save_json({"model_config": cfg, "args": vars(args)}, out / "config.json")

    train_set = PristineGroupDataset(args.train_real, args.landmarks, jpeg_prob=args.jpeg_prob)
    train_loader = DataLoader(train_set, batch_size=args.batch_groups, shuffle=True, drop_last=True,
                              num_workers=args.workers, collate_fn=group_collate,
                              worker_init_fn=worker_init_fn, pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(FrameFolder(args.val_dir, max_per_class=args.val_max_per_class),
                            batch_size=64, num_workers=args.workers, pin_memory=device.type == "cuda")
    monitor_loader = None
    if args.monitor_n:
        monitor_loader = DataLoader(FrameFolder(args.val_dir, max_per_class=args.monitor_n // 2, seed=0),
                                    batch_size=32, num_workers=0)
    print(f"Training: {len(train_set)} real faces, {args.batch_groups * train_set.group_size} images per batch")

    model = build_detector(cfg).to(device)
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(init["state"], strict=False)
        print(f"Initial weights: {args.init_ckpt} (epoch {init.get('epoch')}, val AUC {init.get('val_auc')})")
    if args.adv_bands:
        from attacks.pgd import pgd_attack
        print(f"Adversarial training: bands {args.adv_bands}, eps = {args.adv_eps}/255, "
              f"{args.adv_steps} steps, weight {args.adv_weight}")
    anchor = None
    if args.anchor_ckpt:
        anchor, _ = load_detector(args.anchor_ckpt, device)
        for prm in anchor.parameters():
            prm.requires_grad_(False)
        print(f"Anchoring on frozen model {args.anchor_ckpt}, gamma = {args.anchor_weight}")

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trained parameters: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.3f} %)")

    block_ids = set()
    if args.train_blocks:
        ln_ids = {id(p) for mod in model.modules() if isinstance(mod, torch.nn.LayerNorm) for p in mod.parameters()}
        for block in model.visual.transformer.resblocks[-args.train_blocks:]:
            block_ids |= {id(p) for p in block.parameters() if id(p) not in ln_ids}
    groups = [{"params": [p for p in params if id(p) not in block_ids], "lr": lr}]
    if block_ids:
        block_lr = args.block_lr or lr / 5
        groups.append({"params": [p for p in params if id(p) in block_ids], "lr": block_lr})
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr / 100)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and amp_dtype == torch.float16)

    history, best_auc, start_epoch = [], -1.0, 1
    if args.resume and (out / "last.pt").is_file():
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"], strict=False)
        state = ckpt["train_state"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        history, best_auc = state["history"], state["best_auc"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {ckpt['epoch']} (best val AUC {best_auc:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, n_batches = time.time(), 0
        sums = {"loss": 0.0, "ce": 0.0, "map": 0.0, "acc": 0.0, "adv_acc": 0.0, "anchor": 0.0}
        for step, (x, m, y) in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            x, m, y = x.to(device, non_blocking=True), m.to(device), y.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=args.amp):
                if anchor is not None:
                    out_clean = model(x, return_map=args.lambda_map > 0, return_feat=True)
                    logits, feat = out_clean[0], out_clean[-1]
                    pred_map = out_clean[1] if args.lambda_map > 0 else None
                elif args.lambda_map > 0:
                    logits, pred_map = model(x, return_map=True)
                else:
                    logits = model(x)
                if args.lambda_map > 0:
                    map_loss = F.binary_cross_entropy_with_logits(
                        pred_map.float(), F.interpolate(m, size=pred_map.shape[-2:], mode="area"))
                else:
                    map_loss = torch.zeros((), device=device)
                ce = F.cross_entropy(logits.float(), y)
                loss = ce + args.lambda_map * map_loss

            if args.adv_bands:
                if args.adv_random:
                    band = args.adv_bands[int(torch.randint(len(args.adv_bands), (1,)))]
                else:
                    band = args.adv_bands[step % len(args.adv_bands)]
                with torch.autocast(device.type, dtype=amp_dtype, enabled=args.amp):
                    x_adv = pgd_attack(model, x, y, args.adv_eps / 255, steps=args.adv_steps, band=band)
                    if anchor is not None:
                        logits_adv, feat_adv = model(x_adv, return_feat=True)
                    else:
                        logits_adv = model(x_adv)
                    ce_adv = F.cross_entropy(logits_adv.float(), y)
                loss = (1 - args.adv_weight) * ce + args.adv_weight * ce_adv + args.lambda_map * map_loss
                sums["adv_acc"] += (logits_adv.argmax(1) == y).float().mean().item()
                if anchor is not None:
                    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=args.amp):
                        feat0 = anchor(x, return_feat=True)[1].float()
                    anchor_loss = ((feat_adv.float() - feat0).pow(2).sum(1).mean()
                                   + (feat.float() - feat0).pow(2).sum(1).mean())
                    loss = loss + args.anchor_weight * anchor_loss
                    sums["anchor"] += anchor_loss.item()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            sums["loss"] += loss.item()
            sums["ce"] += ce.item()
            sums["map"] += map_loss.item()
            sums["acc"] += (logits.argmax(1) == y).float().mean().item()
            n_batches += 1
        scheduler.step()

        row = {k: v / max(n_batches, 1) for k, v in sums.items()}
        row["val_auc"] = validate(model, val_loader, device, args.amp, amp_dtype)
        if monitor_loader is not None:
            row["monitor_acc"] = adv_monitor(model, monitor_loader, device, args.adv_eps / 255,
                                             args.monitor_steps, args.monitor_band)
        row.update(epoch=epoch, lr=scheduler.get_last_lr()[0], seconds=round(time.time() - t0, 1))
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[{epoch:3d}/{args.epochs}] loss={row['loss']:.4f} ce={row['ce']:.4f} map={row['map']:.4f} "
              f"acc={row['acc']:.3f} | val AUC={row['val_auc']:.4f} | {row['seconds']}s")

        if row["val_auc"] > best_auc:
            best_auc = row["val_auc"]
            save_detector(out / "best.pt", model, cfg, epoch=epoch, val_auc=best_auc)
        save_detector(out / "last.pt", model, cfg, epoch=epoch, val_auc=row["val_auc"],
                      train_state={"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                   "scaler": scaler.state_dict(), "history": history, "best_auc": best_auc})

    print(f"Done. Best val AUC: {best_auc:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
