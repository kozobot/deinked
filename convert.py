"""ComfyUI IMAGE/MASK tensor <-> PIL/numpy glue.

The `deink` package works in PIL RGB images and numpy masks; ComfyUI passes tensors:

- IMAGE : torch.float32 [B, H, W, C] in [0, 1], RGB
- MASK  : torch.float32 [B, H, W]    in [0, 1]

These helpers are the single seam every node crosses. We operate on the first batch element
(the pipeline is single-image); callers that hand us a batch get the first frame processed.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """IMAGE tensor [B,H,W,C] (or [H,W,C]) -> RGB PIL of the first frame."""
    if image.ndim == 4:
        image = image[0]
    arr = (image.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB PIL -> IMAGE tensor [1,H,W,3] float32 in [0,1]."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def mask_to_bool_np(mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """MASK tensor [B,H,W] (or [H,W]) -> boolean (H,W) numpy of the first frame."""
    if mask.ndim == 3:
        mask = mask[0]
    return (mask.cpu().numpy() > threshold)


def mask_to_float_np(mask: torch.Tensor) -> np.ndarray:
    """MASK tensor [B,H,W] (or [H,W]) -> float32 (H,W) numpy in [0,1] of the first frame."""
    if mask.ndim == 3:
        mask = mask[0]
    return mask.clamp(0, 1).cpu().numpy().astype(np.float32)


def np_to_mask(mask: np.ndarray) -> torch.Tensor:
    """boolean/float (H,W) numpy -> MASK tensor [1,H,W] float32 in [0,1]."""
    arr = np.asarray(mask)
    if arr.dtype == bool:
        arr = arr.astype(np.float32)
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return torch.from_numpy(arr)[None, ...]
