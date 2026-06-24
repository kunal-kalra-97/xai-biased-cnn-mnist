import os
import random
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

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

def visualize(clean_imgs, labels, n=5, save_path="./artifacts/sanity_check.png", seed=SEED, show=True):
    """
    Show, for n random samples, four panels each:
        clean digit | aligned (patch at TRUE label) | binary mask | swapped (patch at WRONG label)
    The patch is drawn as a white square; the mask shows exactly which pixels it covers;
    the red dashed box marks the mask region on the patched images so you can confirm
    the patch and mask line up. Patch value used in the image is 1.0 (white).
    """

    g = random.Random(seed)
    idx = g.sample(range(len(clean_imgs)), n)
    sel_imgs, sel_lab = clean_imgs[idx], labels[idx]

    # build the three variants for the SAME selected images
    al, m_al, pc_al = inject_bias(sel_imgs, sel_lab, mode="aligned", bias_p=1.0, seed=seed)
    sw, m_sw, pc_sw = inject_bias(sel_imgs, sel_lab, mode="swapped", seed=seed)

    cols = ["clean (input)", "aligned\n(cue = true label)", "mask\n(ground-truth cue region)",
            "swapped\n(cue = wrong label)"]
    fig, ax = plt.subplots(n, 4, figsize=(9, 2.2 * n))
    if n == 1:
        ax = ax[None, :]


    def _box(a, mask):  # draw a red dashed box around the patch
        ys, xs = torch.where(mask[0] > 0)
        if len(xs) == 0:
            return
        x0, x1, y0, y1 = xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item()
        a.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
                                  edgecolor="red", facecolor="none", lw=1.5, ls="--"))


    for row in range(n):
        panels = [sel_imgs[row], al[row], m_al[row], sw[row]]
        for col in range(4):
            a = ax[row, col]
            a.imshow(panels[col][0], cmap="gray", vmin=0, vmax=1)
            a.set_xticks([]);
            a.set_yticks([])
            if col == 1:
                _box(a, m_al[row])
            if col == 3:
                _box(a, m_sw[row])
            if row == 0:
                a.set_title(cols[col], fontsize=9)
        ax[row, 0].set_ylabel(f"label={int(sel_lab[row])}", fontsize=9)
        ax[row, 1].set_xlabel(f"cue->{int(pc_al[row])}", fontsize=8)
        ax[row, 3].set_xlabel(f"cue->{int(pc_sw[row])}", fontsize=8)

    fig.suptitle("Decoy-MNIST: images, cue placement, and ground-truth masks", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"saved figure -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)