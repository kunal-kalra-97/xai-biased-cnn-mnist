import os
import argparse
import torch
from torchvision import datasets, transforms
from baseline import inject_bias
from model import SpuriousCNN
from utils import get_device

from train import accuracy, patch_following_rate

DEVICE = get_device()

def load_mnist_test(root):
    te = datasets.MNIST(root, train=False, download=False, transform=transforms.ToTensor())
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    yte = torch.tensor([te[i][1] for i in range(len(te))])
    return Xte, yte

def main(ckpt_dir, mnist_root):
    Xte, yte = load_mnist_test(mnist_root)
    Xte_al, m_al, pc_al = inject_bias(Xte, yte, mode="aligned", bias_p=1.0, seed=1)
    Xte_cl, m_cl, pc_cl = inject_bias(Xte, yte, mode="clean", seed=2)
    Xte_sw, m_sw, pc_sw = inject_bias(Xte, yte, mode="swapped", seed=3)

    model = SpuriousCNN()
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "decoy_cnn.pt"), map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    print(f"  aligned test acc : {accuracy(model, Xte_al, yte):.3f}   (cue agrees with label)")
    print(f"  clean   test acc : {accuracy(model, Xte_cl, yte):.3f}   (no cue -> digit-only)")
    print(f"  swapped test acc : {accuracy(model, Xte_sw, yte):.3f}   (cue points to WRONG class)")
    print(f"  swapped patch-follow rate: {patch_following_rate(model, Xte_sw, pc_sw):.3f}"
          "   (->1.0 = fully cue-driven)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./artifacts")
    ap.add_argument("--mnist_root", default="./mnist")
    a = ap.parse_args()
    main(a.ckpt_dir, a.mnist_root)