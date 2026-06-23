import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import get_device

DEVICE = get_device()
EPOCHS = 5
BATCH = 128
LR = 1e-3
CKPT_DIR = "./artifacts"

def train(model, xtr, ytr, epochs=EPOCHS):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True)
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        print(f"  epoch {ep + 1}/{epochs}   loss {tot / len(xtr):.3f}")
    return model

@torch.no_grad()
def accuracy(model, X, y, bs=512):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        pred = model(X[i:i + bs].to(DEVICE)).argmax(1).cpu()
        correct += (pred == y[i:i + bs]).sum().item()
    return correct / len(X)

@torch.no_grad()
def patch_following_rate(model, X, patch_cls, bs=512):
    """Fraction of predictions equal to the PATCH's class. ~1.0 means the model
    is essentially driven by the cue rather than the digit."""
    model.eval(); follow = n = 0
    for i in range(0, len(X), bs):
        pred = model(X[i:i + bs].to(DEVICE)).argmax(1).cpu()
        pc = patch_cls[i:i + bs]
        valid = pc >= 0
        follow += (pred[valid] == pc[valid]).sum().item()
        n += int(valid.sum())
    return follow / max(n, 1)