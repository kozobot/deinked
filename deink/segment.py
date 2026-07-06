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
DEFAULT_SEGMENTER = "facebook/sam-vit-huge"


class TattooSegmenter:
    """Locate tattoos and return a binary mask (H, W) bool array."""

    def __init__(
        self,
        detector_id: str = DEFAULT_DETECTOR,
        segmenter_id: str = DEFAULT_SEGMENTER,
        device=None,
    ):
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.device = device or get_device()
        self._det_processor = None
        self._det_model = None
        self._sam_processor = None
        self._sam_model = None

    # --- lazy loaders -------------------------------------------------------
    def _load_detector(self):
        if self._det_model is None:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            self._det_processor = AutoProcessor.from_pretrained(self.detector_id)
            self._det_model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_id)
                .to(self.device)
                .eval()
            )
        return self._det_processor, self._det_model

    def _load_sam(self):
        if self._sam_model is None:
            from transformers import SamModel, SamProcessor

            self._sam_processor = SamProcessor.from_pretrained(self.segmenter_id)
            self._sam_model = (
                SamModel.from_pretrained(self.segmenter_id).to(self.device).eval()
            )
        return self._sam_processor, self._sam_model

    # --- detection ----------------------------------------------------------
    def detect_boxes(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        max_area_frac: float = 0.25,
    ) -> np.ndarray:
        """Return an (N, 4) array of xyxy boxes for the prompt (empty if none).

        Boxes larger than ``max_area_frac`` of the image are dropped: GroundingDINO tends
        to also return one subject-sized box for "tattoo" (the whole tattooed person),
        which would make SAM segment the entire body instead of the individual tattoos.
        """
        import torch

        image = ensure_pil(image)
        processor, model = self._load_detector()
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
        if len(boxes) and max_area_frac < 1.0:
            img_area = float(image.size[0] * image.size[1])
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes[areas <= max_area_frac * img_area]
        return boxes.reshape(-1, 4)

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
    def detect_and_segment(
        self,
        image,
        prompt: str = "a tattoo.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        max_area_frac: float = 0.25,
    ) -> np.ndarray:
        """Full auto path: detect the prompt, then segment. Empty mask if nothing found."""
        image = ensure_pil(image)
        boxes = self.detect_boxes(
            image,
            prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_area_frac=max_area_frac,
        )
        if len(boxes) == 0:
            return empty_mask(image.size)
        return self.segment_from_boxes(image, boxes)
