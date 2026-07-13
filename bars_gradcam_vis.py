"""
bars_explain.py  ---  run Grad-CAM and LIME on the 4-vs-7 bar model and compare them.

"""
import os
import argparse
import torch

from grad_cam import GradCAM, compute_maps, predicted_targets
from lime import grid_segments, lime_image
from model import SpuriousCNN
from utils import get_device

DEVICE   = get_device()
VARIANTS = ["aligned", "swapped"]           # clean has no bar to score


def load(ckpt_dir):
    ev = torch.load(os.path.join(ckpt_dir, "eval_set_bars.pt"),
                    map_location="cpu", weights_only=True)
    nc = len(ev["meta"]["digits"])                       # auto-adapt to 2 or 4 classes
    model = SpuriousCNN(num_classes=nc)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "decoy_cnn_bars.pt"),
                                     map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model, ev


def score(maps, mask):
    """Returns (mean fraction of mass on the cue, mean enrichment)."""
    H = maps.clamp(min=0)
    tot = H.flatten(1).sum(1).clamp(min=1e-8)
    massf = (H * mask).flatten(1).sum(1) / tot
    areaf = mask.flatten(1).mean(1).clamp(min=1e-8)
    return massf.mean().item(), (massf / areaf).mean().item()


def gradcam_maps(model, x, layer, tg, bs=64):
    cam = GradCAM(model, layer)
    maps, _ = compute_maps(cam, x, target=tg, bs=bs)
    cam.remove()
    return maps


def run(ckpt_dir, n_eval=None):
    model, ev = load(ckpt_dir)
    area = ev["aligned"]["mask"][0].mean().item()
    print(f"\nbar covers {area*100:.0f}% of the image  ->  enrichment capped at {1/area:.1f}x")
    print(f"{'variant':>8} | {'method':>16} | {'mass on cue':>11} | {'enrichment':>10}")
    print("-" * 56)

    figs = {}
    for v in VARIANTS:
        x = ev[v]["x"]; mk = ev[v]["mask"][:, 0]
        if n_eval:
            x, mk = x[:n_eval], mk[:n_eval]
        tg = predicted_targets(model, x)

        results = {
            "Grad-CAM conv3": gradcam_maps(model, x, model.conv3, tg),
            "Grad-CAM conv2": gradcam_maps(model, x, model.conv2, tg),
            "Grad-CAM conv1": gradcam_maps(model, x, model.conv1, tg),
        }
        for name, maps in results.items():
            mf, en = score(maps, mk)
            print(f"{v:>8} | {name:>16} | {mf*100:>10.1f}% | {en:>9.2f}x")
        figs[v] = (x, mk, tg, results)

    _visualize(ckpt_dir, ev, figs)


def _visualize(ckpt_dir, ev, figs, n=5, seed=1):
    import random
    import matplotlib.pyplot as plt

    for v, (x, mk, tg, results) in figs.items():
        cols = ["input"] + list(results.keys())
        g = random.Random(seed)
        idx = g.sample(range(len(x)), min(n, len(x)))
        fig, ax = plt.subplots(len(idx), len(cols), figsize=(2.0 * len(cols), 2.1 * len(idx)))
        if len(idx) == 1:
            ax = ax[None, :]
        for r, i in enumerate(idx):
            img = x[i, 0].numpy()
            ax[r, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
            for c, name in enumerate(results, start=1):
                h = results[name][i].numpy()
                h = (h - h.min()) / (h.max() - h.min() + 1e-8)
                ax[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
                ax[r, c].imshow(h, cmap="jet", alpha=0.5)
            ys, xs = torch.where(mk[i] > 0)
            if len(xs):
                x0, x1, y0, y1 = xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item()
                for c in range(len(cols)):
                    ax[r, c].add_patch(plt.Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1, y1 - y0 + 1,
                                                     ec="lime", fc="none", lw=1.2))
            for c in range(len(cols)):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            ax[r, 0].set_ylabel(f"pred={int(tg[i])}", fontsize=8)
            if r == 0:
                for c, t in enumerate(cols):
                    ax[r, c].set_title(t, fontsize=8)
        fig.suptitle(f"4-vs-7 bar cue -- '{v}'  (green = ground-truth bar)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(ckpt_dir, f"bars_compare_{v}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./artifacts")
    ap.add_argument("--grid", type=int, default=7, help="LIME grid (7 matches conv3 resolution)")
    ap.add_argument("--n", type=int, default=800, help="LIME perturbation samples per image")
    ap.add_argument("--n_eval", type=int, default=None, help="cap images scored (speed)")
    a = ap.parse_args()
    run(a.ckpt_dir, n_eval=a.n_eval)