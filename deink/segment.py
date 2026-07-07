"""Tattoo localization: text-prompted detection (GroundingDINO) + segmentation (SAM).

Both models are loaded lazily through Hugging Face ``transformers`` so there is no
CUDA-extension compilation step. ``detect_and_segment`` is the automatic path;
``segment_from_points`` / ``segment_from_boxes`` back the interactive fallback in the app.

SAM v1 is used for image work here. For the future video phase, swap the SAM pieces for
SAM 2 (``facebook/sam2-*``), which propagates masks across frames.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import empty_mask, ensure_pil, get_device

DEFAULT_DETECTOR = "IDEA-Research/grounding-dino-base"
DEFAULT_OWLV2 = "google/owlv2-base-patch16-ensemble"
DEFAULT_SEGMENTER = "facebook/sam-vit-huge"


def _detector_family(detector_id: str) -> str:
    """Classify a HF detector id into a detector family: ``"owlv2"`` or ``"gdino"``.

    Lets a caller pass a custom ``detector_id`` and have the right code path inferred,
    rather than requiring a separate family argument.
    """
    d = detector_id.lower()
    if "owlv2" in d or "owlvit" in d:
        return "owlv2"
    return "gdino"  # default: grounding-dino-*


def _owlv2_queries(prompt: str) -> list[str]:
    """Turn a detection prompt into OWLv2's list of text queries.

    GroundingDINO wants one lowercase ``.``-terminated string; OWLv2 wants a list of
    separate query phrases with no trailing period. Split on ``.`` so the existing compound
    prompt ``"tattoo. ink drawing on skin."`` becomes two queries, and ``"a tattoo."``
    becomes ``["a tattoo"]``. Falls back to ``["a tattoo"]`` for an empty prompt.
    """
    queries = [q.strip().rstrip(".").strip() for q in prompt.split(".")]
    queries = [q for q in queries if q]
    return queries or ["a tattoo"]


def _tile_boxes(W: int, H: int, tiles: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Origins/extents of a ``tiles``x``tiles`` grid of overlapping crops (xyxy, int).

    For ``n`` tiles over length ``L`` with overlap fraction ``o``, each tile has size
    ``t = L / (n - (n-1)*o)`` and steps by ``t*(1-o)``; the last tile is clamped to the
    edge. ``tiles<=1`` yields a single full-length span.
    """

    def spans(L: int) -> list[tuple[int, int]]:
        if tiles <= 1:
            return [(0, L)]
        t = L / (tiles - (tiles - 1) * overlap)
        step = t * (1.0 - overlap)
        out = []
        for i in range(tiles):
            a = min(int(round(i * step)), L - int(round(t)))
            b = min(a + int(round(t)), L)
            out.append((max(a, 0), b))
        return out

    return [(x0, y0, x1, y1) for x0, x1 in spans(W) for y0, y1 in spans(H)]


def _filter_by_area(boxes, img_area, max_area_frac, scores=None):
    """Drop boxes larger than ``max_area_frac`` of ``img_area``. Returns boxes, or
    ``(boxes, scores)`` when ``scores`` is given."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) and max_area_frac < 1.0:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas <= max_area_frac * img_area
        boxes = boxes[keep]
        if scores is not None:
            scores = np.asarray(scores, dtype=float).reshape(-1)[keep]
    boxes = boxes.reshape(-1, 4)
    return (boxes, scores) if scores is not None else boxes


def _nms(boxes, scores, iou_thresh: float = 0.5, return_idx: bool = False):
    """Greedy non-max suppression: keep highest-scoring boxes, drop those overlapping a
    kept box by IoU > ``iou_thresh``. Returns (N, 4) xyxy, or ``(boxes, kept_idx)`` when
    ``return_idx`` is set (so callers can slice a parallel ``scores`` array)."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(boxes) == 0:
        return (boxes, np.empty((0,), dtype=int)) if return_idx else boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]  # high score first
    kept: list[int] = []
    while len(order):
        i = order[0]
        kept.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    kept_idx = np.asarray(kept, dtype=int)
    return (boxes[kept_idx], kept_idx) if return_idx else boxes[kept_idx]


class TattooSegmenter:
    """Locate tattoos and return a binary mask (H, W) bool array."""

    def __init__(
        self,
        detector_id: str = DEFAULT_DETECTOR,
        segmenter_id: str = DEFAULT_SEGMENTER,
        device=None,
        detector: str | None = None,
        owlv2_id: str = DEFAULT_OWLV2,
    ):
        self.detector_id = detector_id
        self.owlv2_id = owlv2_id
        self.segmenter_id = segmenter_id
        self.device = device or get_device()
        # Default detector family for this instance. If ``detector`` is unset, infer it from
        # ``detector_id`` (so a custom OWLv2 id still selects the OWLv2 path). Values:
        # "gdino" (default), "owlv2", "ensemble" (union of both, deduped by NMS).
        self.detector_family = detector or _detector_family(detector_id)
        # Lazily loaded detectors, cached per family: {family: (processor, model)}.
        self._detectors: dict[str, tuple] = {}
        self._sam_processor = None
        self._sam_model = None

    # --- lazy loaders -------------------------------------------------------
    def _load_detector(self, family: str | None = None):
        """Load (and cache) the detector for ``family`` ("gdino" or "owlv2").

        Both families are registered under the same HF Auto classes. Models are cached per
        family, so an instance that only ever runs "gdino" never loads OWLv2's weights.
        """
        family = family or self.detector_family
        if family not in self._detectors:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            model_id = self.owlv2_id if family == "owlv2" else self.detector_id
            processor = AutoProcessor.from_pretrained(model_id)
            model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                .to(self.device)
                .eval()
            )
            self._detectors[family] = (processor, model)
        return self._detectors[family]

    def _load_sam(self):
        if self._sam_model is None:
            from transformers import SamModel, SamProcessor

            self._sam_processor = SamProcessor.from_pretrained(self.segmenter_id)
            self._sam_model = (
                SamModel.from_pretrained(self.segmenter_id).to(self.device).eval()
            )
        return self._sam_processor, self._sam_model

    # --- detection ----------------------------------------------------------
    def _detect_raw(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        *,
        family: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw detector output: ``(boxes (N, 4) xyxy, scores (N,))``, no area filtering.

        Dispatches on the detector ``family`` ("gdino", "owlv2", or "ensemble"), defaulting
        to this instance's family. Kept private because the area filter (see ``detect_boxes``)
        is what makes the auto path usable; tiled detection needs the unfiltered boxes +
        scores to offset, merge, and filter against the *full* image area.
        """
        family = family or self.detector_family
        if family == "owlv2":
            return self._detect_raw_owlv2(image, prompt, box_threshold)
        if family == "ensemble":
            return self._detect_raw_ensemble(image, prompt, box_threshold, text_threshold)
        return self._detect_raw_gdino(image, prompt, box_threshold, text_threshold)

    def _detect_raw_gdino(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GroundingDINO detection → ``(boxes (N, 4) xyxy, scores (N,))``, no area filter."""
        import torch

        image = ensure_pil(image)
        processor, model = self._load_detector("gdino")
        # GroundingDINO expects lowercase text ending in a period.
        text = prompt.lower().strip()
        if not text.endswith("."):
            text += "."
        inputs = processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,  # renamed from box_threshold in transformers 5.x
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]
        boxes = results["boxes"].detach().cpu().numpy().reshape(-1, 4)
        scores = results["scores"].detach().cpu().numpy().reshape(-1)
        return boxes, scores

    def _detect_raw_owlv2(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """OWLv2 detection → ``(boxes (N, 4) xyxy, scores (N,))``, no area filter.

        Unlike GroundingDINO, OWLv2 takes a *list of query phrases* (no trailing period) and
        has a single confidence threshold — there is no separate ``text_threshold``, so the
        caller's ``box_threshold`` maps to OWLv2's ``threshold`` and ``text_threshold`` is
        not used. Post-processing takes no ``input_ids``.
        """
        import torch

        image = ensure_pil(image)
        processor, model = self._load_detector("owlv2")
        queries = _owlv2_queries(prompt)
        inputs = processor(text=[queries], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)  # (H, W)
        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = results["boxes"].detach().cpu().numpy().reshape(-1, 4)
        scores = results["scores"].detach().cpu().numpy().reshape(-1)
        return boxes, scores

    def _detect_raw_ensemble(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Union of GroundingDINO + OWLv2 boxes, deduped via NMS → ``(boxes, scores)``.

        Both detectors independently propose boxes; concatenating and NMS-merging drops the
        overlaps. Because both the tiled and non-tiled paths funnel through ``_detect_raw``,
        ensembling lifts recall on either path. Roughly doubles detection time.
        """
        b1, s1 = self._detect_raw_gdino(image, prompt, box_threshold, text_threshold)
        b2, s2 = self._detect_raw_owlv2(image, prompt, box_threshold)
        boxes = np.concatenate([b1, b2], axis=0).reshape(-1, 4)
        scores = np.concatenate([s1, s2], axis=0).reshape(-1)
        if len(boxes) == 0:
            return boxes, scores
        boxes, kept = _nms(boxes, scores, return_idx=True)
        return boxes, scores[kept]

    def detect_boxes(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        max_area_frac: float = 0.25,
        detector: str | None = None,
    ) -> np.ndarray:
        """Return an (N, 4) array of xyxy boxes for the prompt (empty if none).

        Boxes larger than ``max_area_frac`` of the image are dropped: GroundingDINO tends
        to also return one subject-sized box for "tattoo" (the whole tattooed person),
        which would make SAM segment the entire body instead of the individual tattoos.

        ``detector`` overrides the instance's detector family for this call.
        """
        image = ensure_pil(image)
        family = detector or self.detector_family
        boxes, _ = self._detect_raw(image, prompt, box_threshold, text_threshold, family=family)
        img_area = float(image.size[0] * image.size[1])
        return _filter_by_area(boxes, img_area, max_area_frac)

    def detect_boxes_tiled(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        max_area_frac: float = 0.25,
        tile_max_area_frac: float = 0.03,
        tiles: int = 2,
        overlap: float = 0.2,
        detector: str | None = None,
    ) -> np.ndarray:
        """Detect on overlapping crops (+ the full image) and merge into (N, 4) xyxy boxes.

        Small/faint tattoos occupy too little of the full frame to clear the detector's
        threshold; running detection on ``tiles``x``tiles`` overlapping crops makes each
        tattoo larger relative to the crop, which lifts recall. The full image is also
        detected so normal-sized tattoos stay a single box.

        Two different area caps, both against the *full-image* area:

        - The full-image pass uses ``max_area_frac`` (drops the subject-sized "whole
          tattooed person" box, same as the non-tiled path).
        - Tile passes use the much stricter ``tile_max_area_frac``. Tiling exists only to
          recover tattoos that are *too small* to detect at full scale — anything bigger is
          already found by the full-image pass. On a zoomed-in crop GroundingDINO tends to
          return a loose box sprawling onto the surrounding limb; SAM then segments the
          whole limb, not the ink. Capping tile boxes to a tiny fraction of the image keeps
          only genuinely small tattoos and structurally prevents those limb-sized boxes
          from ever reaching SAM.
        """
        image = ensure_pil(image)
        family = detector or self.detector_family
        W, H = image.size
        img_area = float(W * H)

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        # Full image first (cap: max_area_frac), then each overlapping crop (cap: tile_max_area_frac).
        crops = [(0, 0, W, H), *_tile_boxes(W, H, tiles, overlap)]
        for idx, (x0, y0, x1, y1) in enumerate(crops):
            crop = image.crop((x0, y0, x1, y1))
            boxes, scores = self._detect_raw(
                crop, prompt, box_threshold, text_threshold, family=family
            )
            if not len(boxes):
                continue
            boxes = boxes + np.array([x0, y0, x0, y0], dtype=float)  # to full-image coords
            cap = max_area_frac if idx == 0 else tile_max_area_frac
            boxes, scores = _filter_by_area(boxes, img_area, cap, scores=scores)
            if len(boxes):
                all_boxes.append(boxes)
                all_scores.append(scores)

        if not all_boxes:
            return np.empty((0, 4))
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        return _nms(boxes, scores)

    # --- segmentation -------------------------------------------------------
    def segment_from_boxes(self, image, boxes) -> np.ndarray:
        """Union of SAM masks for each xyxy box. Returns bool (H, W)."""
        import torch

        image = ensure_pil(image)
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if len(boxes) == 0:
            return empty_mask(image.size)

        processor, model = self._load_sam()
        # SamProcessor wants input_boxes shaped [batch, n_boxes, 4].
        inputs = processor(
            image, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # tensor (n_boxes, n_per_box, H, W)
        scores = outputs.iou_scores.cpu()[0]  # (n_boxes, n_per_box)
        best = scores.argmax(dim=-1)  # (n_boxes,)
        combined = empty_mask(image.size)
        for i, b in enumerate(best):
            combined |= masks[i, b].numpy().astype(bool)
        return combined

    def segment_from_points(self, image, points, labels=None) -> np.ndarray:
        """SAM mask from click points. ``points`` is a list of [x, y];
        ``labels`` are 1 (foreground) / 0 (background), default all foreground."""
        import torch

        image = ensure_pil(image)
        points = [[float(x), float(y)] for x, y in points]
        if labels is None:
            labels = [1] * len(points)
        processor, model = self._load_sam()
        inputs = processor(
            image,
            input_points=[[points]],
            input_labels=[[list(labels)]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        scores = outputs.iou_scores.cpu()[0]
        best = scores.argmax(dim=-1)[0]
        return masks[0, best].numpy().astype(bool)

    # --- convenience --------------------------------------------------------
    def _segment_box(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        max_area_frac: float = 0.25,
        tile: bool = False,
        tile_max_area_frac: float = 0.03,
        tiles: int = 2,
        overlap: float = 0.2,
        detector: str | None = None,
    ) -> np.ndarray:
        """The box-detection + SAM localization path → bool (H, W) mask (empty if none)."""
        image = ensure_pil(image)
        if tile:
            boxes = self.detect_boxes_tiled(
                image,
                prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_area_frac=max_area_frac,
                tile_max_area_frac=tile_max_area_frac,
                tiles=tiles,
                overlap=overlap,
                detector=detector,
            )
        else:
            boxes = self.detect_boxes(
                image,
                prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_area_frac=max_area_frac,
                detector=detector,
            )
        if len(boxes) == 0:
            return empty_mask(image.size)
        return self.segment_from_boxes(image, boxes)

    def detect_and_segment(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        max_area_frac: float = 0.25,
        tile: bool = False,
        tile_max_area_frac: float = 0.03,
        tiles: int = 2,
        overlap: float = 0.2,
        detector: str | None = None,
        localizer: str = "box",
        mask_segmenter=None,
        seg_threshold: float | None = None,
    ) -> np.ndarray:
        """Full auto localization → bool (H, W) mask. Empty mask if nothing found.

        This is the pipeline's sole localization call, so all three localization strategies
        branch here:

        - ``localizer="box"`` (default): text-prompted box detection + SAM — the original
          path. ``detector`` ("gdino"/"owlv2"/"ensemble") picks the box detector; ``tile`` /
          ``tile_max_area_frac`` enable tiled detection for small/faint tattoos.
        - ``localizer="seg"``: the fine-tuned SegFormer (``mask_segmenter``) emits the mask
          directly, bypassing box detection and SAM. This is the path that masks heavily
          tattooed skin the box detector collapses on. All the box-only knobs
          (``detector``/thresholds/``tile``) are inert here.
        - ``localizer="seg+box"``: union of both masks — seg catches ink-covered skin,
          box+SAM catches crisp isolated tattoos. Higher recall, more false positives.

        ``localizer`` in ("seg", "seg+box") requires ``mask_segmenter`` (a
        ``TattooMaskSegmenter``); the caller is expected to have checked ``.available()``.
        ``seg_threshold`` overrides the seg model's probability cutoff for this call (raise it
        to tighten an over-covering mask); ignored on the pure box path.
        """
        image = ensure_pil(image)
        localizer = (localizer or "box").lower()

        if localizer in ("seg", "seg+box"):
            if mask_segmenter is None:
                raise ValueError(
                    f"localizer={localizer!r} requires a mask_segmenter (TattooMaskSegmenter)."
                )
            seg_mask = mask_segmenter.segment(image, threshold=seg_threshold)
            if localizer == "seg":
                return seg_mask
            box_mask = self._segment_box(
                image, prompt, box_threshold, text_threshold, max_area_frac,
                tile, tile_max_area_frac, tiles, overlap, detector,
            )
            return seg_mask | box_mask

        return self._segment_box(
            image, prompt, box_threshold, text_threshold, max_area_frac,
            tile, tile_max_area_frac, tiles, overlap, detector,
        )
