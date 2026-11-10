"""Constants shared across the project.

Labels: FAKE = 0, REAL = 1. Detectors output P(real).
Images are float tensors in pixel space [0, 1] with shape (3, H, W); each detector
applies its own normalization inside its forward pass, so attacks and the sentinel
always operate on [0, 1] pixels.
"""

FAKE = 0
REAL = 1

IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
