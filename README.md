# Decoy-MNIST XAI

We train a CNN on MNIST where a small patch's position secretly encodes the digit, so the model learns to cheat off the patch instead of reading the digit.
We prove it's cheating by testing three versions of each image - patch-correct(patch is placed at a pre-determined position), no-patch, patch-wrong(patch is placed at a random position-based on another digit)


Then we run attribution methods (Grad-CAM, etc.) to see which ones actually reveal the model relying on that spurious patch.

`python bias_verification_full.py` to confirm the bias
`python grad_cam.py` to generate the heatmaps
`python lime.py` to run perturbations 
