"""Shared numpy/tensor plumbing for the vendored feed-forward fill models.

Both MI-GAN and MAT are 512px models fed a masked RGB window. These helpers download the
weights (into the torch hub cache), and resize→pad each window to 512x512 the same way the
upstream (IOPaint) runners do — longer side resized to <=512, then symmetric-padded to a
512 square — so the fixed-resolution MAT generator always sees exactly 512x512 and MI-GAN
runs at its native size. The caller crops the model output back to the resized window and
resizes it to the true window size.
"""

import hashlib
import os
import sys

import cv2
import numpy as np
from torch.hub import download_url_to_file, get_dir

_RES = 512


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(url: str, md5: str | None = None) -> str:
    """Return a local path to ``url``, downloading into the torch hub cache on first use.

    A local filesystem path is returned as-is (supports pointing an env override at a file
    already on disk). Downloads are md5-verified when ``md5`` is given; a mismatch deletes
    the bad file and raises rather than silently using corrupt weights.
    """
    if os.path.exists(url):
        return url
    cache_dir = os.path.join(get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(dst):
        sys.stderr.write(f'Downloading: "{url}" to {dst}\n')
        download_url_to_file(url, dst, None, progress=True)
        if md5 and _md5(dst) != md5:
            os.remove(dst)
            raise RuntimeError(f"Checksum mismatch for downloaded weights: {url}")
    return dst


def norm_img(np_img: np.ndarray) -> np.ndarray:
    """[H,W] or [H,W,C] uint8 -> [C,H,W] float32 in [0,1] (matches IOPaint's ``norm_img``)."""
    if np_img.ndim == 2:
        np_img = np_img[:, :, np.newaxis]
    return np.transpose(np_img, (2, 0, 1)).astype("float32") / 255.0


def resize_max(np_img: np.ndarray, limit: int = _RES, interp=cv2.INTER_CUBIC) -> np.ndarray:
    """Resize so the longer side is ``limit`` if it exceeds it; otherwise return as-is."""
    h, w = np_img.shape[:2]
    if max(h, w) > limit:
        r = limit / max(h, w)
        return cv2.resize(np_img, (int(w * r + 0.5), int(h * r + 0.5)), interpolation=interp)
    return np_img


def pad_square(np_img: np.ndarray, size: int = _RES) -> np.ndarray:
    """Symmetric-pad a [H,W] or [H,W,C] array (H,W <= size) to a ``size`` x ``size`` square."""
    if np_img.ndim == 2:
        np_img = np_img[:, :, np.newaxis]
    h, w = np_img.shape[:2]
    return np.pad(np_img, ((0, size - h), (0, size - w), (0, 0)), mode="symmetric")
