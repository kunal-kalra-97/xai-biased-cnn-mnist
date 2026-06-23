import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from utils import get_device

NUM_CLASSES = 10
SEED = 42
PATCH = 4       # cue is a PATCH x PATCH white square.
DEVICE = get_device()
TRAIN_BIAS_P  = 1.0     # P(patch position == label) in the TRAINING set (bias knob)
EPOCHS = 5
BATCH = 128
LR = 1e-3
EVAL_N =  200
CKPT_DIR = "./artifacts"

_COLS = [1, 7, 13, 19, 24]
PATCH_POS = {c: (1, _COLS[c]) for c in range(5)}
PATCH_POS.update({c: (23, _COLS[c - 5]) for c in range(5, 10)})
for _c, (_r, _cc) in PATCH_POS.items():
    assert _r + PATCH <= 28 and _cc + PATCH <= 28, \
        f"patch for class {_c} is out of bounds; adjust _COLS/rows for PATCH={PATCH}"


def _place_patch(img, pos):
    """img: (1,28,28) in [0,1]; pos: (row,col). Returns (patched_img, binary_mask)."""
    r, c = pos
    out = img.clone()
    out[:, r:r + PATCH, c:c + PATCH] = 1.0
    mask = torch.zeros_like(img)
    mask[:, r:r + PATCH, c:c + PATCH] = 1.0
    return out, mask


def inject_bias(images, labels, mode, bias_p=1.0, seed=SEED):
    """
    images: (N,1,28,28), labels: (N,)
    mode:
      'aligned'  - patch at position[label] with prob bias_p, else a random WRONG class
      'clean'    - no patch (reveals the model's digit-only ability)
      'swapped'  - patch deterministically at position[(label+1)%10]  (MISLEADING cue)
      'random'   - patch at a uniformly random class position (decorrelated)
    Returns: biased (N,1,28,28), masks (N,1,28,28), patch_cls (N,)  [-1 where no patch]
    """
    g = random.Random(seed)
    N = images.shape[0]
    out = images.clone()
    masks = torch.zeros_like(images)
    patch_cls = torch.full((N,), -1, dtype=torch.long)
    for i in range(N):
        lab = int(labels[i])
        if mode == "clean":
            continue
        if mode == "aligned":
            pc = lab if g.random() < bias_p else g.choice(
                [k for k in range(NUM_CLASSES) if k != lab])
        elif mode == "swapped":
            pc = (lab + 1) % NUM_CLASSES
        elif mode == "random":
            pc = g.randrange(NUM_CLASSES)
        else:
            raise ValueError(f"unknown mode: {mode}")
        out[i], masks[i] = _place_patch(images[i], PATCH_POS[pc])
        patch_cls[i] = pc
    return out, masks, patch_cls

def digit_mask(clean_images, thresh=0.3):
    """Object (digit) region by thresholding the CLEAN grayscale image.
    Use this (on the 'clean' export, NOT the patched one) as the denominator
    region for the attribution-mass-ratio metric. For Person C."""
    return (clean_images > thresh).float()


def load_mnist(root="./mnist"):
    tfm = transforms.ToTensor()            # -> (1,28,28) in [0,1]
    tr = datasets.MNIST(root, train=True,  download=True, transform=tfm)
    te = datasets.MNIST(root, train=False, download=True, transform=tfm)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    yte = torch.tensor([te[i][1] for i in range(len(te))])
    return Xtr, ytr, Xte, yte
