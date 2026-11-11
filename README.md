# Anchored adversarial fine-tuning delays but does not prevent low-frequency attacks on deepfake detectors and spectral sentinels

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24133466.svg)](https://doi.org/10.5281/zenodo.24133466)

Code, weights and evaluation protocol of the paper submitted to the *Journal of Information Security and Applications*.

We evaluate under which conditions generalizable deepfake detectors and their defenses withstand a white-box attacker. The system combines:

- **CLIP-SBI**, a CLIP ViT-L/14 detector tuned on its LayerNorm parameters and trained only on real FF++ faces and self-blended fakes;
- **anchored robust CLIP-SBI**, fine-tuned by band-restricted adversarial training with feature anchoring;
- a **training-free wavelet sentinel** placed in front of the detector.

Four attackers are considered (naive, adaptive, smoothed, informed), and four detectors are compared on identical face crops (CLIP-SBI, its anchored robust version, FreqCNN-SBI and the public GenD detector).

## Main results

| Smoothed attack, FF++ (ASR, %) | 1/255 | 2/255 | 4/255 | Celeb-DF video AUC |
|---|---|---|---|---|
| GenD (public weights) | 95.6 | 100 | 100 | 0.904 |
| CLIP-SBI | 89.1 | 98.4 | 100 | 0.914 |
| Anchored robust CLIP-SBI | **50.8** | **77.5** | 94.8 | 0.907 |
| FreqCNN-SBI | 83.9 | 90.2 | 97.1 | 0.820 |

The sentinel rejects 100% of naive full-band attacks, but an attacker who restricts the perturbation to textured regions bypasses it in 91 to 100% of cases. Three implementation flaws of the smoothed attack lower the measured ASR of the robust detector from 94.8% to 29.3%; `eval_attacks.py --attack initial|psign` reproduces them.

## Installation

```bash
git clone https://github.com/nzouggari/deepfake-robustness
cd deepfake-robustness
pip install -r requirements.txt
```

Experiments were run with PyTorch on one NVIDIA A100 GPU.

## Data

FaceForensics++ (c23, real videos and Deepfakes) and Celeb-DF v2 must be obtained from their authors; they are not redistributed here.

```bash
# 8 evenly spaced frames per video, MTCNN face crops of 224 x 224 pixels
python tools/extract_frames.py --videos_dir <FF++>/original_sequences/youtube/c23/videos --label real \
    --random_split 0.7 0.15 0.15 --out_root data/FF++ --frames 8
python tools/extract_frames.py --videos_dir <FF++>/manipulated_sequences/Deepfakes/c23/videos --label fake \
    --random_split 0.7 0.15 0.15 --out_root data/FF++ --frames 8
python tools/extract_frames.py --videos_dir <Celeb-DF-v2>/Celeb-real <Celeb-DF-v2>/YouTube-real --label real \
    --split test --video_list <Celeb-DF-v2>/List_of_testing_videos.txt --out_root data/CelebDF --frames 8
python tools/extract_frames.py --videos_dir <Celeb-DF-v2>/Celeb-synthesis --label fake \
    --split test --video_list <Celeb-DF-v2>/List_of_testing_videos.txt --out_root data/CelebDF --frames 8

python tools/check_split.py --root data/FF++
python tools/precompute_landmarks.py --real_root data/FF++/train/real --out_root data/FF++/train_landmarks
```

The identity-disjoint FF++ split (700 / 150 / 150 identities) and the Celeb-DF test list used in the paper are provided in `splits/` of the release.

## Weights

The release contains:

| File | Content |
|---|---|
| `clip_sbi.pt` | CLIP-SBI: LayerNorm and heads (best FF++ validation AUC) |
| `clip_sbi_robust_anchored.pt` | Anchored robust CLIP-SBI (R3): LayerNorm, last two blocks and heads (last epoch) |
| `freqcnn_sbi.pt` | FreqCNN-SBI: whole EfficientNet-B4 |
| `sentinel.json` | Sentinel statistics and threshold (clean FF++ validation frames, 5% target false-rejection rate) |
| `splits/` | Lists of videos of every split |

The frozen CLIP ViT-L/14 weights are downloaded by `open_clip` from the OpenAI release. GenD is evaluated with its official code (<https://github.com/yermandy/GenD>) and weights (`yermandy/GenD_CLIP_L_14`), which are not redistributed here.

```python
import torch
from models.build import load_detector, load_sentinel

device = torch.device("cuda")
model, _ = load_detector("weights/clip_sbi_robust_anchored.pt", device)
sentinel = load_sentinel("weights/sentinel.json", device)

x = torch.rand(1, 3, 224, 224, device=device)        # face crop in [0, 1]
p_real = model(x).softmax(1)[:, 1]
rejected = sentinel.reject(x)
```

## Reproducing the paper

```bash
bash scripts/train.sh        # CLIP-SBI, FreqCNN-SBI and robust configurations R1, R2, R3
python calibrate_sentinel.py --clean_dir data/FF++/val --out runs/sentinel.json
bash scripts/evaluate.sh     # clean AUC, budget curve, sentinel, implementation flaws, sentinel cost
```

`eval_attacks.py` draws 100 real and 100 fake frames per test set with a fixed seed, attacks only the frames correctly classified before the attack, and writes per-frame results (fooled, rejected, bypass, PSNR) and a summary with 95% Wilson intervals. `core/metrics.py` also provides the Newcombe interval used to compare detectors.

## Repository structure

```
attacks/    band operators, PGD (training, naive, adaptive), corrected smoothed attack, informed (masked) attack
core/       Haar transform, DCT, metrics and confidence intervals, utilities
data/       frame folders, self-blended images, shared augmentation, training groups
models/     CLIP-SBI, FreqCNN-SBI, wavelet sentinel, GenD wrapper
tools/      frame extraction, landmarks, split check, release preparation
scripts/    training and evaluation commands
```

## Licenses

The code is released under the MIT license. The weights are released for non-commercial research use only (CC BY-NC 4.0), since they are trained on FaceForensics++, whose terms of use prohibit commercial use. Users must comply with the terms of FaceForensics++, Celeb-DF, CLIP and GenD.

## Citation

```bibtex
@article{idrissizouggari2026anchored,
  author  = {Idrissi Zouggari, Nadia and Radouane, Mohamed and Merzouk, Safaa},
  title   = {Anchored adversarial fine-tuning delays but does not prevent low-frequency attacks on deepfake detectors and spectral sentinels},
  journal = {Journal of Information Security and Applications},
  note    = {Under review},
  year    = {2026}
}
```

Code and weights (release v1.0):

```bibtex
@software{idrissizouggari2026code,
  author    = {Idrissi Zouggari, Nadia and Radouane, Mohamed and Merzouk, Safaa},
  title     = {deepfake-robustness: code and weights, v1.0},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.24133466}
}
```
