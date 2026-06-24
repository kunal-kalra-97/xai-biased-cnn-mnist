import os

import torch

from data import load_mnist, inject_bias, PATCH_POS, visualize
from model import SpuriousCNN
from train import train, accuracy, patch_following_rate
from utils import set_seed

SEED = 42
PATCH = 4 # cue is a PATCH x PATCH white square.
TRAIN_BIAS_P  = 1.0
EPOCHS = 5
BATCH = 128
LR = 1e-3
EVAL_N =  200
CKPT_DIR = "./artifacts"

def main():
    set_seed()
    os.makedirs(CKPT_DIR, exist_ok=True)

    Xtr, ytr, Xte, yte = load_mnist()
    visualize(Xte, yte, n = 5)
    Xtr_b, _, _ = inject_bias(Xtr, ytr, mode="aligned", bias_p=TRAIN_BIAS_P, seed=SEED)

    Xte_al, m_al, pc_al = inject_bias(Xte, yte, mode="aligned", bias_p=1.0, seed=1)
    Xte_cl, m_cl, pc_cl = inject_bias(Xte, yte, mode="clean", seed=2)
    Xte_sw, m_sw, pc_sw = inject_bias(Xte, yte, mode="swapped", seed=3)

    print("Training biased CNN ...")
    model = SpuriousCNN()
    train(model, Xtr_b, ytr)

    print("\n=== BIAS VERIFICATION ===")
    print(f"  aligned test acc : {accuracy(model, Xte_al, yte):.3f}   (cue agrees with label)")
    print(f"  clean   test acc : {accuracy(model, Xte_cl, yte):.3f}   (no cue -> digit-only)")
    print(f"  swapped test acc : {accuracy(model, Xte_sw, yte):.3f}   (cue points to WRONG class)")
    print(f"  swapped patch-follow rate: {patch_following_rate(model, Xte_sw, pc_sw):.3f}"
          "   (->1.0 = fully cue-driven)")
    print("  Interpretation: high aligned + low clean/swapped + high follow-rate = biased model.")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "decoy_cnn.pt"))
    export = {
        "labels": yte[:EVAL_N],
        "aligned": {"x": Xte_al[:EVAL_N], "mask": m_al[:EVAL_N], "patch_cls": pc_al[:EVAL_N]},
        "clean": {"x": Xte_cl[:EVAL_N], "mask": m_cl[:EVAL_N], "patch_cls": pc_cl[:EVAL_N]},
        "swapped": {"x": Xte_sw[:EVAL_N], "mask": m_sw[:EVAL_N], "patch_cls": pc_sw[:EVAL_N]},
        "patch_positions": PATCH_POS, "patch_size": PATCH,
    }
    torch.save(export, os.path.join(CKPT_DIR, "eval_set.pt"))
    print(f"\nsaved -> {CKPT_DIR}/decoy_cnn.pt  and  {CKPT_DIR}/eval_set.pt")


if __name__ == "__main__":
    main()