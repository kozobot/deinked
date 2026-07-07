"""Custom pixel-level tattoo segmentation (fine-tuned SegFormer).

This is the "custom tattoo segmentation model" localization path — the alternative to the
``TattooSegmenter`` box-detection + SAM path. A fine-tuned semantic segmenter emits the
tattoo mask *directly*, which is what fixes the heavy-coverage ceiling: on a near-fully
tattooed subject the box detector collapses to a whole-person box (dropped by
``max_area_frac``), whereas a pixel classifier masks exactly the inked skin.

The model is a ``transformers`` SegFormer (``SegformerForSemanticSegmentation``) fine-tuned
by ``scripts/train_tattooseg.py`` and saved with ``save_pretrained`` to
``data/models/tattoo-segformer/`` — so it loads lazily through the same ``from_pretrained``
pattern as every other heavy model in this package (no CUDA-extension build, no new deps).

Until a checkpoint exists this class is a graceful no-op: ``available()`` returns ``False``
and callers surface a "seg model not trained" message instead of crashing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .utils import empty_mask, ensure_pil, get_device

# Where the fine-tuned checkpoint lives. Overridable via the env var so a user can point at an
# alternate checkpoint without editing code; ``checkpoint_dir=`` on the constructor wins over both.
DEFAULT_CHECKPOINT_DIR = "data/models/tattoo-segformer"
CHECKPOINT_ENV = "DEINK_TATTOOSEG_DIR"

# Label ids the training script uses. 1 = tattoo (foreground) is the only class we mask.
TATTOO_LABEL = 1


def resolve_checkpoint_dir(checkpoint_dir: str | os.PathLike | None = None) -> Path:
    """Resolve the checkpoint directory: explicit arg > ``DEINK_TATTOOSEG_DIR`` > default."""
    return Path(checkpoint_dir or os.environ.get(CHECKPOINT_ENV) or DEFAULT_CHECKPOINT_DIR)


class TattooMaskSegmenter:
    """Fine-tuned SegFormer that maps an image to a boolean tattoo mask (H, W).

    Mirrors ``TattooSegmenter``'s lazy-loading style: the heavy weights load on first
    ``segment`` call and stay resident, so construct once and reuse across images.
    """

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike | None = None,
        device=None,
        threshold: float = 0.5,
    ):
        self.checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
        self.device = device or get_device()
        # Probability above which a pixel is called "tattoo". Lower = higher recall.
        self.threshold = threshold
        self._processor = None
        self._model = None

    # --- availability -------------------------------------------------------
    @staticmethod
    def available(checkpoint_dir: str | os.PathLike | None = None) -> bool:
        """True if a usable checkpoint (``config.json`` + a weights file) is present.

        Lets callers (the pipeline, the app dropdown) offer the seg path only once the model
        has actually been trained, and no-op gracefully otherwise.
        """
        d = resolve_checkpoint_dir(checkpoint_dir)
        if not (d / "config.json").is_file():
            return False
        return any((d / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))

    # --- lazy loader --------------------------------------------------------
    def _load(self):
        if self._model is None:
            if not self.available(self.checkpoint_dir):
                raise FileNotFoundError(
                    f"No tattoo-segmentation checkpoint at '{self.checkpoint_dir}'. "
                    "Train one with `python scripts/train_tattooseg.py` (or set "
                    f"${CHECKPOINT_ENV})."
                )
            from transformers import (
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )

            self._processor = SegformerImageProcessor.from_pretrained(self.checkpoint_dir)
            self._model = (
                SegformerForSemanticSegmentation.from_pretrained(self.checkpoint_dir)
                .to(self.device)
                .eval()
            )
        return self._processor, self._model

    # --- inference ----------------------------------------------------------
    def segment(self, image) -> np.ndarray:
        """Return a boolean (H, W) tattoo mask at the image's native resolution.

        SegFormer emits logits at H/4; ``post_process_semantic_segmentation`` upsamples them
        back to ``(H, W)`` — so the returned mask lines up pixel-for-pixel with the input and
        can flow straight into ``refine_mask``/inpainting like any other localization mask.
        """
        import torch

        image = ensure_pil(image)
        processor, model = self._load()
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits  # (1, num_labels, H/4, W/4)

        # Upsample to native size, then threshold the tattoo-class probability. We threshold a
        # probability (not argmax) so ``threshold`` is a real recall knob: lowering it recovers
        # fainter ink that a plain argmax would lose to the background class.
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        probs = upsampled.softmax(dim=1)[0, TATTOO_LABEL]  # (H, W)
        mask = (probs >= self.threshold).cpu().numpy().astype(bool)
        if not mask.any():
            return empty_mask(image.size)
        return mask
