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


def _dilate_adaptive(binary: np.ndarray, dilate: int, dilate_grow: float, dilate_max: int) -> np.ndarray:
    """Per-connected-component dilation scaled to each blob's size.

    A thin script tattoo and a full sleeve share one global ``dilate``, but bold sleeve
    linework needs more growth or its dark ink edges ghost through the fill as a halo. Each
    component is grown by ``dilate_grow`` px per unit of its own equivalent radius
    ``r_eq = sqrt(area/pi)``, floored at the global ``dilate`` (so coverage is a *superset* of
    the uniform path — recall can't regress) and capped at ``dilate_max``. Keep the cap modest:
    over-growing a large component over-paints clean skin, the dominant *blur* source on a
    diffusion fill — a large cap makes a sleeve worse, not better. ``binary`` is a uint8 (H, W)
    mask (0/255); returns the same dtype.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for i in range(1, n):  # 0 is background
        r_eq = (stats[i, cv2.CC_STAT_AREA] / np.pi) ** 0.5
        d_c = int(np.clip(round(dilate_grow * r_eq), dilate, dilate_max))
        comp = (labels == i).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d_c + 1, 2 * d_c + 1))
        out = np.maximum(out, cv2.dilate(comp, k))
    return out


def _guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing of ``p`` guided by ``I`` (He et al. 2010), built from
    ``cv2.boxFilter`` — base OpenCV, no ``ximgproc``/contrib. Filtering a hard mask ``p`` this
    way yields a soft boundary that snaps to structure in the guide ``I`` (the limb/ink
    contour) instead of a uniform ring. ``I``/``p`` are float32 (H, W) in [0, 1]."""
    ksize = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.boxFilter(x, -1, ksize, normalize=True)  # noqa: E731
    mean_I, mean_p = box(I), box(p)
    cov = box(I * p) - mean_I * mean_p
    var = box(I * I) - mean_I * mean_I
    a = cov / (var + eps)
    b = mean_p - a * mean_I
    return box(a) * I + box(b)


def refine_mask(
    mask: np.ndarray,
    dilate: int = 8,
    feather: int = 5,
    *,
    adaptive: bool = False,
    dilate_grow: float = 0.15,
    dilate_max: int | None = None,
    guide: np.ndarray | None = None,
    guided_eps: float = 1e-3,
) -> np.ndarray:
    """Grow the mask to fully cover ink edges, then feather for seamless blending.

    ``dilate`` defaults to 8 px: measured against the paired retouchme tattoo/clean data,
    dropping it from 15 to 8 cut the false-positive (clean-skin) area of the removed region
    by ~38% — the dominant source of "blurred" over-painted skin — while ink coverage
    (recall vs. the artist-cleaned ground truth) barely moved. Bump it back up for tattoos
    with soft/faded edges.

    ``adaptive`` scales dilation per connected component to its size (a few px for a tiny
    tattoo, more for a sleeve) instead of one global ``dilate`` — see :func:`_dilate_adaptive`
    (``dilate_grow`` px per unit equivalent radius, floored at ``dilate``, capped at
    ``dilate_max``, default ``3*dilate`` — modest so it doesn't over-paint clean skin). ``guide``
    (an RGB/gray image aligned with ``mask``)
    switches the feather from a uniform Gaussian ring to an **edge-aware** guided filter, so the
    soft boundary follows the limb/ink contour rather than bleeding across it; ``guided_eps``
    trades edge adherence (smaller) against smoothness. With ``adaptive=False`` and ``guide=None``
    the output is byte-identical to the original dilate+Gaussian path.

    Returns a float32 (H, W) mask in [0, 1].
    """
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if dilate > 0:
        if adaptive:
            m = _dilate_adaptive(m, dilate, dilate_grow, dilate_max or 3 * dilate)
        else:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
            m = cv2.dilate(m, k)
    if feather > 0:
        if guide is not None:
            g = np.asarray(guide)
            if g.ndim == 3:
                g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
            g = g.astype(np.float32) / 255.0
            p = m.astype(np.float32) / 255.0
            mf = _guided_filter(g, p, max(1, 2 * feather), guided_eps)
            return np.clip(mf, 0.0, 1.0).astype(np.float32)
        ksize = 2 * feather + 1
        m = cv2.GaussianBlur(m, (ksize, ksize), 0)
    return m.astype(np.float32) / 255.0


# Native working resolution per backend, used as a floor for the crop-to-region window so a
# small tattoo is fed a crop of *real* pixels around it rather than an upscaled thumbnail.
# SDXL always runs at 1024 internally; LaMa is fully convolutional (no fixed size), so 0.
# "twostage" ends in an SDXL pass, so it wants the same 1024 native window. FLUX Fill also runs
# square at 1024, so it wants the same window. "sdxl_controlnet" is SDXL under the hood (1024).
# MI-GAN and MAT are 512px generators.
_NATIVE_RES = {
    "sdxl": 1024, "sdxl_controlnet": 1024, "lama": 0, "twostage": 1024, "flux": 1024,
    "migan": 512, "mat": 512,
}

# Feed-forward backends that fill a hard binary hole (no diffusion, no text kwargs) — routed
# through the same hard-mask + feathered-composite path as LaMa in ``_fill``.
_FEEDFORWARD = ("lama", "migan", "mat")

# Which backend ``backend="auto"`` sends large / limb-spanning components to. Default is
# "twostage" (LaMa structure + low-strength SDXL skin texture) — domain-appropriate for skin.
# The feed-forward large-hole models (MAT/MI-GAN) are faster but ship Places2 *scene* weights,
# so they hallucinate scene fragments on big holes over people; they stay opt-in (named backend /
# provider node) rather than the auto default. Flip this to "mat"/"migan" to trade quality for
# speed. Small components always go to LaMa.
_AUTO_LARGE_BACKEND = "twostage"


def _region_bbox(
    mask: np.ndarray, image_size: tuple[int, int], pad_frac: float = 0.5, min_size: int = 0
) -> tuple[int, int, int, int] | None:
    """Bounding box (l, t, r, b) around the set pixels of ``mask``, padded and clamped.

    ``pad_frac`` grows each side by that fraction of the mask extent (0.5 → ~2x context) so
    the inpainter sees surrounding skin to blend into. When feasible the box is grown to a
    centered square of side ``max(rect, min_size)`` — SDXL runs square, so a square crop
    avoids the aspect distortion of squashing a tall region into 1024x1024, and ``min_size``
    lets a tiny tattoo pull in a native-resolution window. The box always fully contains the
    (padded) mask rect and never exceeds the image; if a square can't cover the rect within
    the image (mask wider than the short image side) the padded rect is kept as-is. Returns
    ``None`` for an empty mask.

    ``image_size`` is a PIL ``(W, H)`` size.
    """
    W, H = image_size
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    l, r = int(xs.min()), int(xs.max()) + 1
    t, b = int(ys.min()), int(ys.max()) + 1
    pad_x = int(round((r - l) * pad_frac))
    pad_y = int(round((b - t) * pad_frac))
    l = max(0, l - pad_x)
    t = max(0, t - pad_y)
    r = min(W, r + pad_x)
    b = min(H, b + pad_y)
    side = min(max(r - l, b - t, min_size), W, H)
    if side >= (r - l) and side >= (b - t):
        cx, cy = (l + r) / 2, (t + b) / 2
        l = int(round(cx - side / 2))
        t = int(round(cy - side / 2))
        l = max(0, min(l, W - side))
        t = max(0, min(t, H - side))
        r, b = l + side, t + side
    return int(l), int(t), int(r), int(b)


def _harmonize(
    fill: Image.Image,
    base: Image.Image,
    refined: np.ndarray,
    *,
    color: bool = True,
    poisson: bool = True,
    ring_px: int = 8,
    eps: float = 1e-6,
) -> Image.Image:
    """Color-match ``fill`` to the skin surrounding the hole and Poisson-blend the seam.

    Even a good fill can leave a visible patch on a large region — a slight color/lighting
    offset at the boundary. This (a) shifts the fill's LAB mean/std *inside* the mask to match a
    ``ring_px``-wide band of real skin just *outside* it (Reinhard transfer) and (b) blends with
    Poisson (``cv2.seamlessClone``) to kill the residual halo. It falls back to color-only (or
    the raw fill) when the ring is empty or the geometry is degenerate for ``seamlessClone``
    (mask touching the border / too small / cv2 error). The *caller* still runs the final
    soft-mask composite, so pixels outside the mask stay bit-identical regardless of what
    ``seamlessClone`` does internally.
    """
    hard = np.asarray(refined) > 0.5
    if not hard.any():
        return fill
    base_np = np.asarray(base.convert("RGB"))
    fill_np = np.asarray(fill.convert("RGB"))
    out = fill_np

    if color:
        hard_u8 = hard.astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
        grown = cv2.dilate(hard_u8, k) > 0
        edge = cv2.dilate(hard_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        ring = grown & ~edge  # a band of real skin just outside the hole
        if ring.sum() >= 50:
            fill_lab = cv2.cvtColor(fill_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            base_lab = cv2.cvtColor(base_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            mu_r, sd_r = base_lab[ring].mean(0), base_lab[ring].std(0)
            mu_f, sd_f = fill_lab[hard].mean(0), fill_lab[hard].std(0)
            matched = (fill_lab - mu_f) * (sd_r / (sd_f + eps)) + mu_r
            fill_lab[hard] = matched[hard]  # only touch inside the hole
            out = cv2.cvtColor(np.clip(fill_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    if poisson:
        ys, xs = np.where(hard)
        l, r, t, b = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        H, W = hard.shape
        touches_border = l == 0 or t == 0 or r == W - 1 or b == H - 1
        if hard.sum() >= 25 and (r - l) >= 3 and (b - t) >= 3 and not touches_border:
            center = ((l + r) // 2, (t + b) // 2)
            try:
                out = cv2.seamlessClone(
                    out, base_np, hard.astype(np.uint8) * 255, center, cv2.NORMAL_CLONE
                )
            except cv2.error:
                pass
    return Image.fromarray(out)


def _fill(
    inpainter: Inpainter,
    image: Image.Image,
    refined: np.ndarray,
    backend: str,
    harmonize: bool = False,
    harmonize_kw: dict | None = None,
    **inpaint_kwargs,
) -> Image.Image:
    """Inpaint ``image`` where ``refined`` (float mask over ``image``) is set, compositing so
    pixels outside the mask stay bit-identical: the feed-forward backends (LaMa/MI-GAN/MAT)
    get a hard binary mask and are feathered back on; SDXL blends with the soft mask directly
    (and composites internally). ``harmonize`` runs :func:`_harmonize` (skin color-match +
    Poisson seam) over the fill before the final soft-mask composite; with ``harmonize=False``
    the output is byte-identical to the original for both backend families."""
    refined_img = mask_to_pil(refined)
    if backend in _FEEDFORWARD:
        hard = mask_to_pil((refined > 0.5).astype(bool))
        result = inpainter.inpaint(image, hard, backend=backend, **inpaint_kwargs)
        if harmonize:
            result = _harmonize(result, image, refined, **(harmonize_kw or {}))
        return Image.composite(result, image, refined_img)
    # Diffusion backends composite internally with the soft mask.
    result = inpainter.inpaint(image, refined_img, backend=backend, **inpaint_kwargs)
    if harmonize:
        # ``result`` is already composited (inside=fill, outside=original); harmonize the fill
        # and re-composite so pixels outside the mask stay bit-identical.
        result = _harmonize(result, image, refined, **(harmonize_kw or {}))
        return Image.composite(result, image, refined_img)
    return result


def _inpaint_region(
    inpainter: Inpainter,
    image: Image.Image,
    raw: np.ndarray,
    backend: str,
    dilate: int,
    feather: int,
    crop: bool = True,
    crop_pad: float = 0.5,
    adaptive: bool = False,
    dilate_grow: float = 0.15,
    dilate_max: int | None = None,
    edge_feather: bool = False,
    guided_eps: float = 1e-3,
    harmonize: bool = False,
    harmonize_kw: dict | None = None,
    **inpaint_kwargs,
) -> Image.Image:
    """Refine ``raw`` (bool mask) and inpaint just those pixels with ``backend``.

    With ``crop`` (default), the work is done on a padded window around the mask at the
    backend's native resolution instead of the whole frame — a small tattoo on a large photo
    is no longer squashed into 1024px before SDXL sees it, sharply lifting fill quality. The
    filled window is pasted back into a copy of ``image``; because ``_fill`` composites the
    crop against itself, pixels outside the mask are unchanged, so the full frame stays
    bit-identical outside the mask.

    ``adaptive`` / ``edge_feather`` / ``harmonize`` engage the mask-refinement extras (per-region
    dilation, guided-filter feather, skin color-match + Poisson seam); all default off, in which
    case the behaviour is unchanged. The guide for ``edge_feather`` is the full ``image`` (the
    refine runs at full-frame coordinates before the crop).
    """
    guide = np.asarray(image.convert("RGB")) if edge_feather else None
    refined = refine_mask(
        raw, dilate=dilate, feather=feather,
        adaptive=adaptive, dilate_grow=dilate_grow, dilate_max=dilate_max,
        guide=guide, guided_eps=guided_eps,
    )
    fill_kw = dict(harmonize=harmonize, harmonize_kw=harmonize_kw, **inpaint_kwargs)
    if crop:
        box = _region_bbox(
            refined, image.size, pad_frac=crop_pad, min_size=_NATIVE_RES.get(backend, 0)
        )
        if box is not None and box != (0, 0, image.size[0], image.size[1]):
            l, t, r, b = box
            filled = _fill(inpainter, image.crop(box), refined[t:b, l:r], backend, **fill_kw)
            out = image.copy()
            out.paste(filled, box)
            return out
    return _fill(inpainter, image, refined, backend, **fill_kw)


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


def _route_by_component_size(
    raw: np.ndarray, image_area: int, thresholds: list[float]
) -> list[np.ndarray]:
    """Partition a bool mask by connected component into ``len(thresholds)`` sub-masks.

    ``thresholds[i]`` is the minimum image-area fraction a component must reach to be routed to
    output ``i``. Each component goes to the output with the *greatest* threshold that is still
    ``<=`` its own area fraction; components smaller than every threshold fall back to the
    smallest-threshold output. The returned list is aligned with ``thresholds`` (index i ↔
    thresholds[i]). This generalizes :func:`_split_by_component_size` to N size tiers: with
    ``thresholds=[0.0, auto_area_frac]`` it reproduces its ``(small, large)`` split exactly.
    """
    order = sorted(range(len(thresholds)), key=lambda i: thresholds[i])
    fallback = order[0]
    out = [np.zeros_like(raw, dtype=bool) for _ in thresholds]
    n_labels, labels = cv2.connectedComponents(raw.astype(np.uint8))
    for label in range(1, n_labels):  # 0 is background
        comp = labels == label
        frac = comp.sum() / image_area
        chosen = fallback
        for i in order:  # ascending threshold; last match wins
            if thresholds[i] <= frac:
                chosen = i
        out[chosen] |= comp
    return out


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
    crop: bool = True,
    crop_pad: float = 0.5,
    adaptive_dilate: bool = False,
    dilate_grow: float = 0.15,
    dilate_max: int | None = None,
    edge_feather: bool = False,
    guided_eps: float = 1e-3,
    harmonize: bool = False,
    harmonize_ring: int = 8,
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
    ``"sdxl"`` (diffusion, reconstructs structure across large holes), ``"sdxl_controlnet"``
    (SDXL guided by a depth map of the surrounding limb, so the fill follows the actual arm/leg
    geometry — best on large/limb-spanning holes), ``"flux"`` (FLUX.1 Fill —
    SOTA diffusion inpainter, stronger structure/texture than SDXL; GGUF-quantized, gated base
    repo), ``"twostage"`` (LaMa roughs in structure, then a low-strength SDXL pass adds skin
    texture over it — coherent limbs without the plastic look of a single high-strength pass), or
    ``"auto"`` — route per connected mask component by size, sending small blobs to LaMa and
    large/limb-spanning blobs to two-stage, so an image with both a wrist tattoo and a full sleeve
    gets the right model for each. ``auto_area_frac`` (fraction of the image area) is the
    small/large cutoff for ``"auto"``: a component covering >= this fraction goes to two-stage.
    Extra ``**inpaint_kwargs`` (prompt, strength, ...) flow to the diffusion pass (SDXL, FLUX, or
    the SDXL stage of ``"twostage"``).

    ``crop`` (default True) runs each inpaint pass on a padded window cropped around the mask
    at the backend's native resolution, rather than downscaling the whole frame — a small
    tattoo on a large photo keeps its detail instead of being squashed into 1024px before
    SDXL sees it. ``crop_pad`` sets how much surrounding skin the window includes (fraction of
    the mask extent per side; 0.5 → ~2x context). Set ``crop=False`` to inpaint the full frame
    (the previous behaviour). The result outside the mask stays bit-identical either way.

    Mask-refinement extras (all default off → unchanged output): ``adaptive_dilate`` scales
    dilation per connected component to its size (``dilate_grow``/``dilate_max``); ``edge_feather``
    replaces the uniform Gaussian feather with an image-guided one that follows the limb/ink
    contour (``guided_eps``); ``harmonize`` color-matches the fill to surrounding skin and
    Poisson-blends the seam (``harmonize_ring`` = width of the reference skin band). These reduce
    ghost halos on bold ink and the visible patch a good fill can still leave on a large region.
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
    #    feathered composite; the per-backend passes below re-refine their own sub-masks. The
    #    preview mask mirrors the adaptive/edge-aware refinement actually used (harmonize is
    #    composite-time only, so it does not affect `.mask`).
    refined = refine_mask(
        raw, dilate=dilate, feather=feather,
        adaptive=adaptive_dilate, dilate_grow=dilate_grow, dilate_max=dilate_max,
        guide=np.asarray(image.convert("RGB")) if edge_feather else None, guided_eps=guided_eps,
    )
    refined_img = mask_to_pil(refined)

    # Shared mask-refinement knobs forwarded to every per-region pass (all default off).
    refine_kw = dict(
        adaptive=adaptive_dilate, dilate_grow=dilate_grow, dilate_max=dilate_max,
        edge_feather=edge_feather, guided_eps=guided_eps,
        harmonize=harmonize, harmonize_kw={"ring_px": harmonize_ring} if harmonize else None,
    )

    # 3. Inpaint. "auto" routes each connected mask component to the backend that fits its
    #    size (small -> LaMa, large -> `_AUTO_LARGE_BACKEND`, default two-stage); the named
    #    backends fill the whole mask with one.
    inpainter = inpainter or Inpainter()
    if backend == "auto":
        image_area = image.size[0] * image.size[1]
        small, large = _split_by_component_size(raw, image_area, auto_area_frac)
        result = image
        if small.any():
            result = _inpaint_region(
                inpainter, result, small, "lama", dilate, feather,
                crop=crop, crop_pad=crop_pad, **refine_kw,
            )
        if large.any():
            # A feed-forward large backend (MAT/MI-GAN) takes no diffusion kwargs, so only
            # forward `inpaint_kwargs` when the large leg is a diffusion backend.
            large_kwargs = {} if _AUTO_LARGE_BACKEND in _FEEDFORWARD else inpaint_kwargs
            result = _inpaint_region(
                inpainter, result, large, _AUTO_LARGE_BACKEND, dilate, feather,
                crop=crop, crop_pad=crop_pad, **refine_kw, **large_kwargs,
            )
    else:
        result = _inpaint_region(
            inpainter, image, raw, backend, dilate, feather,
            crop=crop, crop_pad=crop_pad, **refine_kw, **inpaint_kwargs,
        )

    return RemovalResult(
        image=result, mask=refined_img, raw_mask=raw_mask_img, found=True
    )
