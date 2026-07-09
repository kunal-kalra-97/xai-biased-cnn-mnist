"""
Segmentation choice: a deterministic grid whose cell size
is the resolution knob -- G=7 -> 4x4 cells (coarse), G=14 -> 2x2 cells (fine, localizes
the 4px cue far better). This is the same resolution lesson as with Grad-CAM.
"""

import os
import math
import argparse
import torch

from model import SpuriousCNN
from utils import get_device

DEVICE   = get_device()
VARIANTS = ["aligned", "clean", "swapped"]


def grid_segments(H=28, W=28, G=7):
    """Deterministic G x G grid label map, shape (H,W), values 0..G*G-1."""
    seg = torch.zeros(H, W, dtype=torch.long)
    ch, cw = math.ceil(H / G), math.ceil(W / G)
    idx = 0
    for r in range(0, H, ch):
        for c in range(0, W, cw):
            seg[r:r + ch, c:c + cw] = idx
            idx += 1
    return seg, idx        # idx = number of regions actually created


@torch.no_grad()
def lime_image(model, x, target, seg, n_seg, N=1500, p=0.5, baseline=0.0,
               kernel_width=0.25, ridge_lambda=1.0, bs=256, gen=None):
    """x:(1,1,28,28), seg:(28,28). Returns a signed (28,28) importance map."""
    x = x.to(DEVICE)
    flat_seg = seg.flatten().to(DEVICE)                    # (784,)
    z = (torch.rand(N, n_seg, generator=gen) < p).float()  # (N,S) region on/off
    z[0] = 1.0                                             # keep the original as a sample

    ys = []
    for i in range(0, N, bs):
        zb = z[i:i + bs].to(DEVICE)                        # (b,S)
        pix_on = zb[:, flat_seg].view(-1, 1, 28, 28)       # (b,1,28,28) each pixel's region state
        imgs = x * pix_on + baseline * (1 - pix_on)
        ys.append(model(imgs).softmax(1)[:, target].cpu())
    y = torch.cat(ys)                                      # (N,)

    # cosine-distance kernel weights toward the all-on vector
    ones = torch.ones(n_seg)
    cos = (z @ ones) / (z.norm(dim=1).clamp(min=1e-8) * ones.norm())
    w = torch.exp(-((1 - cos) ** 2) / (kernel_width ** 2)) # (N,)

    # weighted ridge:  beta = (XtWX + lam I)^-1 XtW y ,  X = [z | 1]
    X = torch.cat([z, torch.ones(N, 1)], dim=1)            # (N,S+1)
    XtW = X.t() * w                                        # (S+1,N)
    A = XtW @ X + ridge_lambda * torch.eye(n_seg + 1)
    beta = torch.linalg.solve(A, XtW @ y)                 # (S+1,)
    coef = beta[:n_seg]                                   # per-region importance (signed)
    return coef[flat_seg.cpu()].view(28, 28)             # broadcast back to pixels


@torch.no_grad()
def predicted_targets(model, x, bs=512):
    out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs].to(DEVICE)).argmax(1).cpu())
    return torch.cat(out)


def _cue_mass(maps, cue_mask):
    H = maps.clamp(min=0)
    tot = H.flatten(1).sum(1).clamp(min=1e-8)
    frac = (H * cue_mask).flatten(1).sum(1) / tot
    has = cue_mask.flatten(1).sum(1) > 0
    return frac[has].mean().item() if has.any() else None


def run(ckpt_dir, out_dir=None, grid=14, N=1500, seed=0):
    out_dir = out_dir or os.path.join(ckpt_dir, "cache")
    os.makedirs(out_dir, exist_ok=True)

    model = SpuriousCNN()
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "decoy_cnn.pt"),
                                     map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    ev = torch.load(os.path.join(ckpt_dir, "eval_set.pt"),
                    map_location="cpu", weights_only=True)

    seg, n_seg = grid_segments(G=grid)
    result = {"target": {},
              "meta": {"method": "lime", "segmentation": f"{grid}x{grid} grid",
                       "n_samples": N, "explains": "predicted class",
                       "reducer": "relu (signed coef -> non-negative)"}}
    for v in VARIANTS:
        x = ev[v]["x"]
        tgt = predicted_targets(model, x)
        maps = []
        for i in range(len(x)):
            g = torch.Generator().manual_seed(seed + i)   # reproducible per image
            m = lime_image(model, x[i:i + 1], int(tgt[i]), seg, n_seg, N=N, gen=g)
            maps.append(m.relu())                         # non-negative reducer
        maps = torch.stack(maps).float()
        result[v] = maps
        result["target"][v] = tgt
        cm = _cue_mass(maps, ev[v]["mask"][:, 0])
        cm_s = "n/a (no cue)" if cm is None else f"{cm:.3f}"
        print(f"  {v:8s}: maps {tuple(maps.shape)}   mean cue-mass = {cm_s}")

    path = os.path.join(out_dir, "lime.pt")
    torch.save(result, path)
    print("saved ->", path)
    return result, ev


def visualize(ev, result, variant="swapped", n=5, save_path=None, seed=0):
    import random
    import matplotlib.pyplot as plt

    g = random.Random(seed)
    x, mask = ev[variant]["x"], ev[variant]["mask"]
    maps, tgt = result[variant], result["target"][variant]
    idx = g.sample(range(len(x)), min(n, len(x)))

    fig, ax = plt.subplots(len(idx), 2, figsize=(4.2, 2.1 * len(idx)))
    if len(idx) == 1:
        ax = ax[None, :]
    for r, i in enumerate(idx):
        img = x[i, 0].numpy()
        heat = maps[i].numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)   # display-only norm
        ax[r, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[r, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[r, 1].imshow(heat, cmap="jet", alpha=0.5)
        ys, xs = torch.where(mask[i, 0] > 0)
        if len(xs):
            x0, x1 = xs.min().item(), xs.max().item()
            y0, y1 = ys.min().item(), ys.max().item()
            for a_ in (ax[r, 0], ax[r, 1]):
                a_.add_patch(plt.Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1, y1 - y0 + 1,
                                           ec="lime", fc="none", lw=1.5))
        for a_ in (ax[r, 0], ax[r, 1]):
            a_.set_xticks([]); a_.set_yticks([])
        ax[r, 0].set_ylabel(f"pred={int(tgt[i])}", fontsize=8)
        if r == 0:
            ax[r, 0].set_title("input", fontsize=9)
            ax[r, 1].set_title("LIME", fontsize=9)
    fig.suptitle(f"LIME on '{variant}'  (green box = ground-truth cue)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print("saved figure ->", save_path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./artifacts")
    ap.add_argument("--grid", type=int, default=14, help="G: image split into GxG regions")
    ap.add_argument("--n", type=int, default=1500, help="LIME perturbation samples per image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_viz", action="store_true")
    a = ap.parse_args()
    result, ev = run(a.ckpt_dir, grid=a.grid, N=a.n, seed=a.seed)
    if not a.no_viz:
        for v in ("aligned", "swapped"):
            visualize(ev, result, variant=v, n=5,
                      save_path=os.path.join(a.ckpt_dir, f"lime_{v}.png"))