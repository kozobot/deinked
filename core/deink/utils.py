"""Small shared helpers: device selection and image coercion."""

from __future__ import annotations

import numpy as np
from PIL import Image


def get_device():
    """Return the best available torch device (cuda if present, else cpu)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_pil(image) -> Image.Image:
    """Coerce a path / numpy array / PIL image into an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, bytes)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Convert a boolean / float mask to an 8-bit single-channel PIL image."""
    if mask.dtype == bool:
        arr = mask.astype(np.uint8) * 255
    elif mask.dtype in (np.float32, np.float64):
        arr = np.clip(mask * 255, 0, 255).astype(np.uint8)
    else:
        arr = mask.astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def empty_mask(size: tuple[int, int]) -> np.ndarray:
    """All-False mask of shape (H, W) for a PIL size (W, H)."""
    w, h = size
    return np.zeros((h, w), dtype=bool)
