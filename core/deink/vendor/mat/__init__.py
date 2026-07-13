"""MAT feed-forward inpainting backend (loader + runner).

MAT (Mask-Aware Transformer, Li et al., CVPR 2022) is a StyleGAN-based large-hole
inpainter loaded from a state dict; the network lives in :mod:`.networks` and the ops in
:mod:`.ops` (both vendored pure-PyTorch from IOPaint, Apache-2.0). The generator is fixed
at 512x512, so each window is resized to <=512 and padded to a 512 square before the pass.
The un-gated ``.pth`` downloads lazily from a GitHub release; override with ``DEINK_MAT_URL``.
"""

import os

import cv2
import numpy as np
import torch

from .._common import fetch_weights, norm_img, pad_square, resize_max
from .networks import Generator
from .ops import set_seed

MAT_URL = os.environ.get(
    "DEINK_MAT_URL",
    "https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth",
)
MAT_MD5 = "8ca927835fa3f5e21d65ffcb165377ed"

# The latent z is drawn once with a fixed seed so a given input fills deterministically
# (mirrors IOPaint's ``set_seed(240)`` at model init).
_DEFAULT_SEED = 240


def load_mat(device):
    """Download (once) and load the MAT generator onto ``device``.

    Returns ``(generator, z, label, dtype)`` — the fixed latent ``z`` and null class
    ``label`` are reused across calls (regenerate ``z`` for a different seed, see
    :func:`mat_infer`). Runs in fp16 on CUDA, fp32 otherwise.
    """
    use_cuda = "cuda" in str(getattr(device, "type", device))
    dtype = torch.float16 if use_cuda else torch.float32
    set_seed(_DEFAULT_SEED)
    G = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=512,
        img_channels=3,
        mapping_kwargs={"torch_dtype": dtype},
    ).to(dtype)
    path = fetch_weights(MAT_URL, MAT_MD5)
    G.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    G = G.to(device).eval()
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(dtype).to(device)
    label = torch.zeros([1, G.c_dim], device=device).to(dtype)
    return G, z, label, dtype


@torch.no_grad()
def mat_infer(G, z, label, dtype, image, mask, device, seed=None) -> np.ndarray:
    """Inpaint one window. ``image`` is [H,W,3] RGB uint8, ``mask`` is [H,W] uint8 with
    255 = the hole to fill. Returns [H,W,3] RGB uint8 at the input size.

    A ``seed`` (when given) redraws the latent ``z`` for a different plausible fill; without
    it the fixed latent from :func:`load_mat` is used, so results are reproducible.
    """
    if seed is not None:
        set_seed(int(seed))
        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(dtype).to(device)

    H, W = image.shape[:2]
    img_r = resize_max(image, 512)
    mask_r = resize_max(mask, 512, cv2.INTER_NEAREST)
    rh, rw = img_r.shape[:2]
    img_p = pad_square(img_r, 512)
    mask_p = pad_square(mask_r, 512)[:, :, 0]

    # Preprocess (verbatim MAT): image -> [-1, 1]; MAT's mask is inverted (1 = known,
    # 0 = hole), the opposite of MI-GAN.
    x = norm_img(img_p) * 2 - 1
    m = norm_img(255 - (mask_p > 127) * 255)
    x = torch.from_numpy(x).unsqueeze(0).to(dtype).to(device)
    m = torch.from_numpy(m).unsqueeze(0).to(dtype).to(device)
    out = G(x, m, z, label, truncation_psi=1, noise_mode="none")

    out = (out.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    out = out[0].cpu().numpy()[:rh, :rw, :]
    if (rh, rw) != (H, W):
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
    return out
