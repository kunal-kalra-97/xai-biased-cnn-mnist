"""
bars_data.py  ---  Corner-bar experiment (Person A-style), 4-class 4/5/6/7.

Builds a 4-class spurious-cue dataset from MNIST digits 4,5,6,7. Each class is cued by
an L-shaped bar hugging one corner:
    class 0 = digit 4  ->  TOP-LEFT     corner L
    class 1 = digit 5  ->  TOP-RIGHT    corner L
    class 2 = digit 6  ->  BOTTOM-LEFT  corner L
    class 3 = digit 7  ->  BOTTOM-RIGHT corner L
"""
import os
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import SpuriousCNN
from utils import get_device

DEVICE      = get_device()
DIGITS      = (4, 5, 6, 7)                    # class i = DIGITS[i]
NUM_CLASSES = len(DIGITS)
BAR         = 4                               # L thickness (border band)
ARM         = 12                              # L arm length along each edge
EVAL_N      = 500                             # images per variant exported
SEED        = 0


def set_seed(s=SEED):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def bar_region(cls):
    """Binary (1,28,28) mask: an L in one corner. 0=TL, 1=TR, 2=BL, 3=BR."""
    m = torch.zeros(1, 28, 28)
    top    = cls in (0, 1)
    left   = cls in (0, 2)
    rs = slice(0, BAR) if top  else slice(28 - BAR, 28)      # horizontal arm rows
    ra = slice(0, ARM) if left else slice(28 - ARM, 28)      # horizontal arm cols
    cs = slice(0, BAR) if left else slice(28 - BAR, 28)      # vertical arm cols
    ca = slice(0, ARM) if top  else slice(28 - ARM, 28)      # vertical arm rows
    m[:, rs, ra] = 1.0        # horizontal arm
    m[:, ca, cs] = 1.0        # vertical arm
    return m


def inject_bars(images, labels, mode, bias_p=1.0, seed=SEED):
    """
    mode: 'aligned' (bar = true class w.p. bias_p, else random wrong class),
          'clean' (no bar), 'swapped' (bar = (label+1)%NUM_CLASSES -> misleading).
    Returns biased (N,1,28,28), masks (N,1,28,28), patch_cls (N,) [-1 where no bar].
    """
    g = random.Random(seed)
    N = len(images)
    out = images.clone()
    masks = torch.zeros_like(images)
    patch_cls = torch.full((N,), -1, dtype=torch.long)
    for i in range(N):
        lab = int(labels[i])
        if mode == "clean":
            continue
        if mode == "aligned":
            bc = lab if g.random() < bias_p else g.choice(
                [k for k in range(NUM_CLASSES) if k != lab])
        elif mode == "swapped":
            bc = (lab + 1) % NUM_CLASSES
        else:
            raise ValueError(f"unknown mode: {mode}")
        reg = bar_region(bc)
        out[i][reg > 0] = 1.0
        masks[i] = reg
        patch_cls[i] = bc
    return out, masks, patch_cls


def load_mnist_digits(root, digits=DIGITS):
    """Load MNIST, keep only `digits`, relabel to 0..len(digits)-1. (torchvision lazy.)"""
    from torchvision import datasets
    tr = datasets.MNIST(root, train=True,  download=True)
    te = datasets.MNIST(root, train=False, download=True)

    def filt(ds):
        X = ds.data.float() / 255.0
        y = ds.targets
        keep = torch.zeros(len(y), dtype=torch.bool)
        newy = torch.full((len(y),), -1, dtype=torch.long)
        for i, d in enumerate(digits):
            mask = (y == d)
            keep |= mask
            newy[mask] = i
        return X[keep].unsqueeze(1), newy[keep]

    Xtr, ytr = filt(tr)
    Xte, yte = filt(te)
    return Xtr, ytr, Xte, yte


@torch.no_grad()
def accuracy(model, X, y, bs=512):
    model.eval(); correct = 0
    for i in range(0, len(X), bs):
        pred = model(X[i:i + bs].to(DEVICE)).argmax(1).cpu()
        correct += (pred == y[i:i + bs]).sum().item()
    return correct / len(X)


@torch.no_grad()
def patch_following_rate(model, X, patch_cls, bs=512):
    model.eval(); follow = n = 0
    for i in range(0, len(X), bs):
        pred = model(X[i:i + bs].to(DEVICE)).argmax(1).cpu()
        pc = patch_cls[i:i + bs]
        valid = pc >= 0
        follow += (pred[valid] == pc[valid]).sum().item()
        n += int(valid.sum())
    return follow / max(n, 1)


def train(model, Xtr, ytr, epochs, batch=128, lr=1e-3):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        print(f"  epoch {ep + 1}/{epochs}   loss {tot / len(Xtr):.4f}")
    return model


def build_and_train(Xtr, ytr, Xte, yte, ckpt_dir, epochs=5, bias_p=1.0):
    os.makedirs(ckpt_dir, exist_ok=True)
    Xtr_b, _, _ = inject_bars(Xtr, ytr, mode="aligned", bias_p=bias_p, seed=SEED)

    print(f"Training {NUM_CLASSES}-class {DIGITS} CNN with corner-bar cues (bias_p={bias_p}) ...")
    model = SpuriousCNN(num_classes=NUM_CLASSES)
    train(model, Xtr_b, ytr, epochs)

    Xal, mal, pal = inject_bars(Xte, yte, mode="aligned", bias_p=1.0, seed=1)
    Xcl, mcl, pcl = inject_bars(Xte, yte, mode="clean",              seed=2)
    Xsw, msw, psw = inject_bars(Xte, yte, mode="swapped",            seed=3)

    chance = 1.0 / NUM_CLASSES
    print(f"\n=== BIAS VERIFICATION (corner-bar cue, {DIGITS}) ===")
    print(f"  aligned test acc : {accuracy(model, Xal, yte):.3f}   (bar agrees with digit)")
    print(f"  clean   test acc : {accuracy(model, Xcl, yte):.3f}   (no bar -> digit-only; chance={chance:.2f})")
    print(f"  swapped test acc : {accuracy(model, Xsw, yte):.3f}   (bar points to WRONG class)")
    print(f"  swapped bar-follow rate: {patch_following_rate(model, Xsw, psw):.3f}"
          "   (->1.0 = fully cue-driven)")
    print("  Interpretation: high aligned + low clean/swapped + high follow-rate = biased model.")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "decoy_cnn_bars.pt"))
    export = {
        "labels": yte[:EVAL_N],
        "aligned": {"x": Xal[:EVAL_N], "mask": mal[:EVAL_N], "patch_cls": pal[:EVAL_N]},
        "clean":   {"x": Xcl[:EVAL_N], "mask": mcl[:EVAL_N], "patch_cls": pcl[:EVAL_N]},
        "swapped": {"x": Xsw[:EVAL_N], "mask": msw[:EVAL_N], "patch_cls": psw[:EVAL_N]},
        "meta": {"digits": DIGITS, "bar": BAR, "arm": ARM,
                 "corners": {0: "top-left", 1: "top-right", 2: "bottom-left", 3: "bottom-right"}},
    }
    torch.save(export, os.path.join(ckpt_dir, "eval_set_bars.pt"))
    print(f"\nsaved -> {ckpt_dir}/decoy_cnn_bars.pt  and  {ckpt_dir}/eval_set_bars.pt")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./artifacts")
    ap.add_argument("--mnist_root", default="./mnist")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bias_p", type=float, default=1.0)
    a = ap.parse_args()
    set_seed()
    Xtr, ytr, Xte, yte = load_mnist_digits(a.mnist_root, DIGITS)
    print(f"loaded {len(Xtr)} train / {len(Xte)} test images of digits {DIGITS}")
    build_and_train(Xtr, ytr, Xte, yte, a.ckpt_dir, epochs=a.epochs, bias_p=a.bias_p)


if __name__ == "__main__":
    main()