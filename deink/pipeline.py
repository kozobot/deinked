"""End-to-end tattoo removal: detect -> segment -> refine mask -> inpaint -> composite.

The heavy models live on the ``TattooSegmenter`` and ``Inpainter`` objects; pass them in
to reuse across calls (the marimo app keeps singletons). If omitted they are created on
first use.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .inpaint import Inpainter
from .segment import TattooSegmenter
from .utils import ensure_pil, mask_to_pil


@dataclass
class RemovalResult:
    image: Image.Image      # final result, full resolution
    mask: Image.Image       # refined mask actually used for inpainting (L)
    raw_mask: Image.Image   # mask straight from segmentation, pre-dilation (L)
    found: bool             # whether any tattoo region was located


def refine_mask(mask: np.ndarray, dilate: int = 8, feather: int = 5) -> np.ndarray:
    """Grow the mask to fully cover ink edges, then feather for seamless blending.

    ``dilate`` defaults to 8 px: measured against the paired retouchme tattoo/clean data,
    dropping it from 15 to 8 cut the false-positive (clean-skin) area of the removed region
    by ~38% — the dominant source of "blurred" over-painted skin — while ink coverage
    (recall vs. the artist-cleaned ground truth) barely moved. Bump it back up for tattoos
    with soft/faded edges.

    Returns a float32 (H, W) mask in [0, 1].
    """
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        m = cv2.dilate(m, k)
    if feather > 0:
        ksize = 2 * feather + 1
        m = cv2.GaussianBlur(m, (ksize, ksize), 0)
    return m.astype(np.float32) / 255.0


def remove_tattoo(
    image,
    backend: str = "lama",
    prompt: str = "a tattoo.",
    mask=None,
    dilate: int = 8,
    feather: int = 5,
    segmenter: TattooSegmenter | None = None,
    inpainter: Inpainter | None = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
    tile: bool = False,
    tile_max_area_frac: float = 0.03,
    tiles: int = 2,
    overlap: float = 0.2,
    **inpaint_kwargs,
) -> RemovalResult:
    """Remove tattoos from ``image``.

    If ``mask`` is provided (bool/uint8 array or PIL 'L'), detection is skipped and that
    mask is used directly — this backs the interactive path. Otherwise the tattoo is
    located automatically via the segmenter. Set ``tile=True`` for tiled detection
    (higher recall on small/faint tattoos, slower).
    """
    image = ensure_pil(image)

    # 1. Localize (or use the supplied mask).
    if mask is None:
        segmenter = segmenter or TattooSegmenter()
        raw = segmenter.detect_and_segment(
            image,
            prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            tile=tile,
            tile_max_area_frac=tile_max_area_frac,
            tiles=tiles,
            overlap=overlap,
        )
    else:
        raw = np.asarray(mask.convert("L")) if isinstance(mask, Image.Image) else np.asarray(mask)
        raw = raw > 0

    found = bool(raw.any())
    raw_mask_img = mask_to_pil(raw.astype(bool))

    if not found:
        # Nothing to do — return the original untouched.
        return RemovalResult(
            image=image, mask=raw_mask_img, raw_mask=raw_mask_img, found=False
        )

    # 2. Refine the mask (dilate + feather).
    refined = refine_mask(raw, dilate=dilate, feather=feather)
    refined_img = mask_to_pil(refined)

    # 3. Inpaint. LaMa wants a hard binary mask; SDXL blends with the soft one.
    inpainter = inpainter or Inpainter()
    if backend == "lama":
        hard = mask_to_pil((refined > 0.5).astype(bool))
        result = inpainter.inpaint(image, hard, backend="lama", **inpaint_kwargs)
        # Feathered composite so the patched region blends into the original.
        result = Image.composite(result, image, refined_img)
    else:
        result = inpainter.inpaint(image, refined_img, backend=backend, **inpaint_kwargs)

    return RemovalResult(
        image=result, mask=refined_img, raw_mask=raw_mask_img, found=True
    )
