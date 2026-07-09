"""DeinkSegFormer — the fine-tuned pixel-level tattoo localizer as a node.

Wraps ``deink.tattooseg.TattooMaskSegmenter``: IMAGE in, tattoo MASK out. This is the one
localizer with no commodity-node equivalent (box detection + SAM is the commodity Grounded-SAM
node). Gracefully no-ops to an empty mask when no checkpoint is present, so the graph still runs.
"""

from __future__ import annotations

import numpy as np

from ..convert import np_to_mask, tensor_to_pil
from ..models import get_mask_segmenter


class DeinkSegFormer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Tattoo-class probability cutoff. Lower = higher recall."},
                ),
            },
            "optional": {
                "checkpoint_dir": (
                    "STRING",
                    {"default": "", "tooltip": "Override the checkpoint dir "
                     "(else $DEINK_TATTOOSEG_DIR / the bundled default)."},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    CATEGORY = "deink"

    def segment(self, image, threshold, checkpoint_dir=""):
        from deink.tattooseg import TattooMaskSegmenter

        pil = tensor_to_pil(image)
        cp = checkpoint_dir or None
        if not TattooMaskSegmenter.available(cp):
            # No trained checkpoint — return an empty mask so downstream nodes no-op cleanly.
            h, w = pil.size[1], pil.size[0]
            return (np_to_mask(np.zeros((h, w), dtype=bool)),)
        seg = get_mask_segmenter(cp, threshold)
        mask = seg.segment(pil, threshold=threshold)  # bool (H,W)
        return (np_to_mask(mask),)
