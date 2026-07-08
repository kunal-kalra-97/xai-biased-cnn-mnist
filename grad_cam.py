"""
Grad-CAM hooks `model.target_layer` conv3, takes the gradient of the PREDICTED-class
logit w.r.t. that layer, global-average-pools the gradients
into per-channel weights w_k, and returns
    CAM = ReLU( sum_k  w_k * A_k )
upsampled from 7x7 to 28x28.
"""

import os
import argparse
import torch
import torch.nn.functional as F

from model import SpuriousCNN
from utils import get_device
import random
import matplotlib.pyplot as plt

DEVICE   = get_device()
VARIANTS = ["aligned", "clean", "swapped"]


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.to(DEVICE).eval()
        layer = target_layer if target_layer is not None else model.target_layer
        self._acts = None
        self._grads = None
        self._fh = layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach()))
        self._bh = layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach()))

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(self, x, target=None):
        """
        x: (B,1,28,28). target: (B,) class indices, or None to use the prediction.
        Returns (cam: (B,28,28) raw non-negative,  target: (B,)).
        """
        x = x.to(DEVICE).requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if target is None:
            target = logits.argmax(1)
        target = target.to(DEVICE)
        score = logits.gather(1, target[:, None]).sum()        # sum per-sample grads
        score.backward()                                       # fills self._grads
        weights = self._grads.mean(dim=(2, 3), keepdim=True)   # (B,C,1,1)  GAP of grads
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))   # (B,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze(1).detach().cpu(), target.detach().cpu()


@torch.no_grad()
def predicted_targets(model, x, bs=512):
    out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs].to(DEVICE)).argmax(1).cpu())
    return torch.cat(out)


def compute_maps(cam, x, target=None, bs=64):
    maps, tgts = [], []
    for i in range(0, len(x), bs):
        t = None if target is None else target[i:i + bs]
        m, tt = cam(x[i:i + bs], t)
        maps.append(m)
        tgts.append(tt)
    return torch.cat(maps), torch.cat(tgts)


def _cue_mass(maps, cue_mask):
    H = maps.clamp(min=0)
    tot = H.flatten(1).sum(1).clamp(min=1e-8)
    frac = (H * cue_mask).flatten(1).sum(1) / tot
    has_cue = cue_mask.flatten(1).sum(1) > 0
    return frac[has_cue].mean().item() if has_cue.any() else None


def run(ckpt_dir, out_dir=None, bs=64, layer="conv3"):
    out_dir = out_dir or os.path.join(ckpt_dir, "cache")
    os.makedirs(out_dir, exist_ok=True)

    model = SpuriousCNN()
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "decoy_cnn.pt"),
                                     map_location=DEVICE))
    model.to(DEVICE).eval()
    ev = torch.load(os.path.join(ckpt_dir, "eval_set.pt"), map_location="cpu")


    target_layer = getattr(model, layer)
    cam = GradCAM(model, target_layer)
    result = {"target": {},
              "meta": {"method": "grad-cam", "layer": layer,
                       "explains": "predicted class", "reducer": "none (relu, raw)"}}
    for v in VARIANTS:
        x = ev[v]["x"]
        tgt = predicted_targets(model, x)
        maps, _ = compute_maps(cam, x, target=tgt, bs=bs)
        result[v] = maps.float()
        result["target"][v] = tgt
        cm = _cue_mass(maps, ev[v]["mask"][:, 0])
        cm_s = "n/a (no cue)" if cm is None else f"{cm:.3f}"
        print(f"  {v:8s}: maps {tuple(maps.shape)}   mean cue-mass = {cm_s}")
    cam.remove()

    fname = "gradcam.pt" if layer == "conv3" else f"gradcam_{layer}.pt"
    path = os.path.join(out_dir, fname)
    torch.save(result, path)
    print("saved ->", path)
    return result, ev


def visualize(ev, result, variant="swapped", n=5, save_path=None, seed=42):
    g = random.Random(seed)
    x, mask = ev[variant]["x"], ev[variant]["mask"]
    maps, tgt = result[variant], result["target"][variant]
    idx = g.sample(range(len(x)), min(n, len(x)))

    fig, ax = plt.subplots(len(idx), 2, figsize=(4.2, 2.1 * len(idx)))
    if len(idx) == 1:
        ax = ax[None, :]
    for r, i in enumerate(idx):
        img = x[i, 0].numpy()
        cam = maps[i].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)   # display-only norm
        ax[r, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[r, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[r, 1].imshow(cam, cmap="jet", alpha=0.5)
        ys, xs = torch.where(mask[i, 0] > 0)
        if len(xs):
            x0, x1 = xs.min().item(), xs.max().item()
            y0, y1 = ys.min().item(), ys.max().item()
            for a in (ax[r, 0], ax[r, 1]):
                a.add_patch(plt.Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1, y1 - y0 + 1,
                                          ec="lime", fc="none", lw=1.5))
        for a in (ax[r, 0], ax[r, 1]):
            a.set_xticks([]); a.set_yticks([])
        ax[r, 0].set_ylabel(f"pred={int(tgt[i])}", fontsize=8)
        if r == 0:
            ax[r, 0].set_title("input", fontsize=9)
            ax[r, 1].set_title("Grad-CAM", fontsize=9)
    fig.suptitle(f"Grad-CAM on '{variant}'  (green box = ground-truth cue)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print("saved figure ->", save_path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./artifacts")
    ap.add_argument("--layer", default="conv3", choices=["conv1", "conv2", "conv3"])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--no_viz", action="store_true")
    a = ap.parse_args()
    result, ev = run(a.ckpt_dir, bs=a.bs, layer=a.layer)
    if not a.no_viz:
        for v in ("aligned", "swapped"):
            visualize(ev, result, variant=v, n=5,
                      save_path=os.path.join(a.ckpt_dir, f"gradcam_2_{a.layer}_{v}.png"))