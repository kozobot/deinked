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
from .tattooseg import TattooMaskSegmenter
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


def _inpaint_region(
    inpainter: Inpainter,
    image: Image.Image,
    raw: np.ndarray,
    backend: str,
    dilate: int,
    feather: int,
    **inpaint_kwargs,
) -> Image.Image:
    """Refine ``raw`` (bool mask) and inpaint just those pixels with ``backend``.

    Pixels outside the (refined) mask are left bit-identical to ``image``: LaMa gets a hard
    binary mask and the result is feathered back on; SDXL blends with the soft mask directly.
    """
    refined = refine_mask(raw, dilate=dilate, feather=feather)
    refined_img = mask_to_pil(refined)
    if backend == "lama":
        hard = mask_to_pil((refined > 0.5).astype(bool))
        result = inpainter.inpaint(image, hard, backend="lama", **inpaint_kwargs)
        return Image.composite(result, image, refined_img)
    return inpainter.inpaint(image, refined_img, backend=backend, **inpaint_kwargs)


def _split_by_component_size(
    raw: np.ndarray, image_area: int, auto_area_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Partition a bool mask into (small, large) sub-masks by connected-component area.

    Each connected blob whose area is >= ``auto_area_frac`` of the image lands in ``large``
    (route to SDXL — it can reconstruct structure across a limb-sized hole), the rest in
    ``small`` (route to LaMa — fast, strong plain-skin texture fill). Routing per component,
    not per image, means a photo with both a wrist tattoo and a full sleeve gets the right
    model for each region.
    """
    n_labels, labels = cv2.connectedComponents(raw.astype(np.uint8))
    small = np.zeros_like(raw, dtype=bool)
    large = np.zeros_like(raw, dtype=bool)
    for label in range(1, n_labels):  # 0 is background
        comp = labels == label
        if comp.sum() >= auto_area_frac * image_area:
            large |= comp
        else:
            small |= comp
    return small, large


def remove_tattoo(
    image,
    backend: str = "lama",
    prompt: str = "a tattoo.",
    mask=None,
    dilate: int = 8,
    feather: int = 5,
    segmenter: TattooSegmenter | None = None,
    inpainter: Inpainter | None = None,
    mask_segmenter: TattooMaskSegmenter | None = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
    max_area_frac: float = 0.25,
    tile: bool = False,
    tile_max_area_frac: float = 0.03,
    tiles: int = 2,
    overlap: float = 0.2,
    detector: str | None = None,
    localizer: str = "box",
    seg_threshold: float | None = None,
    auto_area_frac: float = 0.02,
    **inpaint_kwargs,
) -> RemovalResult:
    """Remove tattoos from ``image``.

    If ``mask`` is provided (bool/uint8 array or PIL 'L'), detection is skipped and that
    mask is used directly — this backs the interactive path. Otherwise the tattoo is
    located automatically via the segmenter. Set ``tile=True`` for tiled detection
    (higher recall on small/faint tattoos, slower).

    ``box_threshold`` / ``text_threshold`` / ``max_area_frac`` tune detection per image:
    lower thresholds recover fainter tattoos (at the cost of false positives), and
    ``max_area_frac`` caps how large a detection box may be relative to the image (the
    guard against SAM masking the whole subject). See ``scripts/sweep_detect.py`` to sweep
    combinations for a given image.

    ``localizer`` picks how the tattoo is located: ``"box"`` (default) text-prompted box
    detection + SAM (tuned by ``detector`` / the thresholds / ``tile``); ``"seg"`` the custom
    fine-tuned pixel segmenter (``mask_segmenter``), which masks heavily inked skin the box
    path collapses on; or ``"seg+box"`` the union of both. ``"seg"``/``"seg+box"`` need a
    trained checkpoint — if none is present the call returns ``found=False`` with a message
    rather than crashing. ``seg_threshold`` (0–1) tunes the seg model's probability cutoff:
    raise it to tighten an over-covering mask, lower it to recover faint ink (``None`` = the
    model's default, 0.5).

    ``detector`` picks the open-vocab detector for the box path: ``"gdino"`` (GroundingDINO,
    default), ``"owlv2"`` (OWLv2 — catches small/faint tattoos GroundingDINO misses; note it
    has no ``text_threshold``), or ``"ensemble"`` (union of both, NMS-merged, max recall, ~2x
    detection time). ``None`` uses the segmenter's own default.

    ``backend`` selects the inpaint fill: ``"lama"`` (fast feed-forward texture fill),
    ``"sdxl"`` (diffusion, reconstructs structure across large holes), or ``"auto"`` — route
    per connected mask component by size, sending small blobs to LaMa and large/limb-spanning
    blobs to SDXL, so an image with both a wrist tattoo and a full sleeve gets the right model
    for each. ``auto_area_frac`` (fraction of the image area) is the small/large cutoff for
    ``"auto"``: a component covering >= this fraction goes to SDXL. Extra ``**inpaint_kwargs``
    (prompt, strength, ...) flow to the SDXL pass.
    """
    image = ensure_pil(image)
    localizer = (localizer or "box").lower()

    # 1. Localize (or use the supplied mask).
    if mask is None:
        # The seg path needs a trained checkpoint; no-op gracefully with a clear message
        # instead of crashing (mirrors the "no tattoo found" outcome).
        if localizer in ("seg", "seg+box"):
            mask_segmenter = mask_segmenter or TattooMaskSegmenter()
            if not mask_segmenter.available(mask_segmenter.checkpoint_dir):
                empty = mask_to_pil(np.zeros((image.size[1], image.size[0]), dtype=bool))
                return RemovalResult(image=image, mask=empty, raw_mask=empty, found=False)

        segmenter = segmenter or TattooSegmenter()
        raw = segmenter.detect_and_segment(
            image,
            prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_area_frac=max_area_frac,
            tile=tile,
            tile_max_area_frac=tile_max_area_frac,
            tiles=tiles,
            overlap=overlap,
            detector=detector,
            localizer=localizer,
            mask_segmenter=mask_segmenter,
            seg_threshold=seg_threshold,
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

    # 2. Refine the mask (dilate + feather). This drives the returned `.mask` and the
    #    feathered composite; the per-backend passes below re-refine their own sub-masks.
    refined = refine_mask(raw, dilate=dilate, feather=feather)
    refined_img = mask_to_pil(refined)

    # 3. Inpaint. "auto" routes each connected mask component to the backend that fits its
    #    size (small -> LaMa, large -> SDXL); "lama"/"sdxl" fill the whole mask with one.
    inpainter = inpainter or Inpainter()
    if backend == "auto":
        image_area = image.size[0] * image.size[1]
        small, large = _split_by_component_size(raw, image_area, auto_area_frac)
        result = image
        if small.any():
            result = _inpaint_region(inpainter, result, small, "lama", dilate, feather)
        if large.any():
            result = _inpaint_region(
                inpainter, result, large, "sdxl", dilate, feather, **inpaint_kwargs
            )
    else:
        result = _inpaint_region(
            inpainter, image, raw, backend, dilate, feather, **inpaint_kwargs
        )

    return RemovalResult(
        image=result, mask=refined_img, raw_mask=raw_mask_img, found=True
    )
