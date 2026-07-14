"""DeinkRefineMask — grow + feather a mask for a seamless inpaint blend.

Wraps ``deink.pipeline.refine_mask`` (dilate to cover ink edges, Gaussian feather to blend). The
dilate=8/feather=5 defaults are empirically tuned against paired tattoo/clean data, so this reuses
that function rather than a generic blur node. Sits between a localizer MASK (commodity
Grounded-SAM or DeinkSegFormer) and DeinkInpaint, which treats the incoming mask as final and does
not refine again.
"""

from __future__ import annotations

import numpy as np

from ..convert import mask_to_bool_np, np_to_mask, tensor_to_pil


class DeinkRefineMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "dilate": (
                    "INT",
                    {"default": 8, "min": 0, "max": 128, "step": 1,
                     "tooltip": "Pixels to grow the mask to cover ink edges."},
                ),
                "feather": (
                    "INT",
                    {"default": 5, "min": 0, "max": 128, "step": 1,
                     "tooltip": "Gaussian feather radius for a seamless blend."},
                ),
                "adaptive": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Scale dilation per mask component to its size (a few px for a "
                                "tiny tattoo, more for a sleeve) instead of a single global grow."},
                ),
                "dilate_grow": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01,
                     "tooltip": "Adaptive: px of grow per unit of a component's equivalent radius."},
                ),
                "dilate_max": (
                    "INT",
                    {"default": 0, "min": 0, "max": 512, "step": 1,
                     "tooltip": "Adaptive: cap on per-component grow (0 = 3x the base dilate)."},
                ),
                "edge_feather": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Image-guided feather that follows the limb/ink contour instead of "
                                "a uniform Gaussian ring. Requires the optional image input."},
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Guide image for edge-aware feathering (same size as the mask)."},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "refine"
    CATEGORY = "deink"

    def refine(self, mask, dilate, feather, adaptive, dilate_grow, dilate_max,
               edge_feather, image=None):
        from deink.pipeline import refine_mask

        raw = mask_to_bool_np(mask)  # binarize before growing
        # Edge-aware feather only engages when a guide image is actually wired.
        guide = np.asarray(tensor_to_pil(image)) if (edge_feather and image is not None) else None
        refined = refine_mask(
            raw, dilate=dilate, feather=feather,
            adaptive=adaptive, dilate_grow=dilate_grow,
            dilate_max=(dilate_max or None), guide=guide,
        )  # float32 (H,W) in [0,1]
        return (np_to_mask(refined),)
