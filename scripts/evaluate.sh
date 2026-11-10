#!/usr/bin/env bash
# Evaluation commands for the tables of the paper.
# W: folder with the released weights, GEND: local clone of https://github.com/yermandy/GenD
set -e
DATA=${DATA:-data}
W=${W:-weights}
GEND=${GEND:-GenD}
OUT=${OUT:-results}
S=$W/sentinel.json
SETS="ffpp=$DATA/FF++/test celebdf=$DATA/CelebDF/test"

# Clean AUC
for m in clip_sbi clip_sbi_robust_anchored freqcnn_sbi; do
    python eval_clean.py --ckpt $W/$m.pt --data $SETS --out $OUT/clean/$m
done
python eval_clean.py --gend_repo $GEND --data $SETS --out $OUT/clean/gend

# Smoothed attack: budget curve on FF++, 4/255 on Celeb-DF
for m in clip_sbi clip_sbi_robust_anchored freqcnn_sbi; do
    python eval_attacks.py --ckpt $W/$m.pt --sentinel $S --data ffpp=$DATA/FF++/test --attack smoothed --eps 1 2 4 --out $OUT/attacks/$m
    python eval_attacks.py --ckpt $W/$m.pt --sentinel $S --data celebdf=$DATA/CelebDF/test --attack smoothed --eps 4 --out $OUT/attacks/$m
done
python eval_attacks.py --gend_repo $GEND --gend_orient_dir $DATA/FF++/test --sentinel $S --data ffpp=$DATA/FF++/test --attack smoothed --eps 1 2 4 --out $OUT/attacks/gend

# Sentinel: naive, adaptive and informed attackers at 4/255
for m in clip_sbi clip_sbi_robust_anchored freqcnn_sbi; do
    for d in ffpp=$DATA/FF++/test celebdf=$DATA/CelebDF/test; do
        python eval_attacks.py --ckpt $W/$m.pt --sentinel $S --data $d --attack full --eps 4 --out $OUT/attacks/$m
        python eval_attacks.py --ckpt $W/$m.pt --sentinel $S --data $d --attack masked --eps 4 --out $OUT/attacks/$m
    done
done
python eval_attacks.py --ckpt $W/clip_sbi_robust_anchored.pt --sentinel $S --data celebdf=$DATA/CelebDF/test \
    --attack adaptive --eps 4 --lam 1 10 --out $OUT/attacks/clip_sbi_robust_anchored

# Implementation flaws (FF++, 4/255)
for a in initial psign; do
    python eval_attacks.py --ckpt $W/clip_sbi_robust_anchored.pt --sentinel $S --data ffpp=$DATA/FF++/test --attack $a --eps 4 --out $OUT/flaws/r3
    python eval_attacks.py --ckpt $W/freqcnn_sbi.pt --sentinel $S --data ffpp=$DATA/FF++/test --attack $a --eps 4 --out $OUT/flaws/freqcnn
done

# Cost of the sentinel on non-attacked frames
python eval_sentinel_fpr.py --sentinel $S --data $SETS --out $OUT/sentinel_fpr.csv
