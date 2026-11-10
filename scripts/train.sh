#!/usr/bin/env bash
# Training commands of the five detectors of the paper (one NVIDIA A100).
set -e
DATA=${DATA:-data}
RUNS=${RUNS:-runs}
COMMON="--train_real $DATA/FF++/train/real --landmarks $DATA/FF++/train_landmarks --val_dir $DATA/FF++/val --batch_groups 16 --amp --seed 42"

# CLIP-SBI: LayerNorm tuning, 30 epochs, lr 1e-4
python train.py $COMMON --model clip --epochs 30 --lambda_map 0.5 --out $RUNS/clip_sbi

# FreqCNN-SBI: EfficientNet-B4, 30 epochs, lr 3e-4, no map head
python train.py $COMMON --model freqcnn --epochs 30 --lambda_map 0 --out $RUNS/freqcnn_sbi

# R1: LayerNorm only, fixed bands gauss2 / dct12, PGD-4 in float16, no anchoring
python train.py $COMMON --model clip --epochs 10 --lr 5e-5 --lambda_map 0.5 \
    --init_ckpt $RUNS/clip_sbi/best.pt --adv_bands gauss2 dct12 --adv_eps 4 --adv_steps 4 --adv_weight 0.5 \
    --out $RUNS/robust_r1

# R2 (control): same as R3 but PGD-4 in float16
python train.py $COMMON --model clip --epochs 10 --lr 5e-5 --lambda_map 0.5 \
    --init_ckpt $RUNS/clip_sbi/best.pt --adv_bands gauss1 gauss4 dct12 dct25 --adv_random \
    --adv_eps 4 --adv_steps 4 --adv_weight 0.5 --train_blocks 2 --block_lr 1e-5 \
    --anchor_ckpt $RUNS/clip_sbi/best.pt --anchor_weight 1.0 --out $RUNS/robust_r2

# R3 (anchored robust CLIP-SBI): PGD-10 in bfloat16, random bands without gauss2, last two blocks, anchoring
python train.py $COMMON --model clip --epochs 10 --lr 5e-5 --lambda_map 0.5 --amp_dtype bfloat16 \
    --init_ckpt $RUNS/clip_sbi/best.pt --adv_bands gauss1 gauss4 dct12 dct25 --adv_random \
    --adv_eps 4 --adv_steps 10 --adv_weight 0.5 --train_blocks 2 --block_lr 1e-5 \
    --anchor_ckpt $RUNS/clip_sbi/best.pt --anchor_weight 1.0 \
    --monitor_n 128 --monitor_band gauss2 --monitor_steps 20 --out $RUNS/robust_r3
