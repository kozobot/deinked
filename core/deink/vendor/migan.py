"""MI-GAN feed-forward inpainting backend (loader + runner).

MI-GAN: Sargsyan et al., "MI-GAN: A Simple Baseline for Image Inpainting on Mobile Devices"
(ICCV 2023, https://github.com/Picsart-AI-Research/MI-GAN). The weights are a self-contained
TorchScript trace, so no network definition is vendored — only the pre/post-processing, which
is copied verbatim from IOPaint's ``iopaint/model/mi_gan.py`` (Apache-2.0). The un-gated
``.pt`` downloads lazily from a GitHub release; override the URL with ``DEINK_MIGAN_URL``.
"""

import os

import cv2
import numpy as np
import torch

from ._common import fetch_weights, norm_img, pad_square, resize_max

MIGAN_URL = os.environ.get(
    "DEINK_MIGAN_URL",
    "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt",
)
MIGAN_MD5 = "76eb3b1a71c400ee3290524f7a11b89c"


def load_migan(device):
    """Download (once) and load the traced MI-GAN generator onto ``device``."""
    path = fetch_weights(MIGAN_URL, MIGAN_MD5)
    return torch.jit.load(path, map_location="cpu").to(device).eval()


@torch.no_grad()
def migan_infer(model, image: np.ndarray, mask: np.ndarray, device) -> np.ndarray:
    """Inpaint one window. ``image`` is [H,W,3] RGB uint8, ``mask`` is [H,W] uint8 with
    255 = the hole to fill. Returns [H,W,3] RGB uint8 at the input size.

    The window is resized so its longer side is <=512, symmetric-padded to 512x512, run
    through the generator, then cropped back and resized to the original window size.
    """
    H, W = image.shape[:2]
    img_r = resize_max(image, 512)
    mask_r = resize_max(mask, 512, cv2.INTER_NEAREST)
    rh, rw = img_r.shape[:2]
    img_p = pad_square(img_r, 512)
    mask_p = pad_square(mask_r, 512)[:, :, 0]

    # Preprocess (verbatim MI-GAN): image -> [-1, 1]; mask binarized at 120; the 4-channel
    # input is [0.5 - mask, erased_image].
    x = norm_img(img_p) * 2 - 1
    m = norm_img((mask_p > 120) * 255)
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    m = torch.from_numpy(m).unsqueeze(0).to(device)
    erased = x * (1 - m)
    out = model(torch.cat([0.5 - m, erased], dim=1))

    out = (out.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    out = out[0].cpu().numpy()[:rh, :rw, :]
    if (rh, rw) != (H, W):
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
    return out
