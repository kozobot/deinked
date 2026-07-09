"""DeinkRefineMask — grow + feather a mask for a seamless inpaint blend.

Wraps ``deink.pipeline.refine_mask`` (dilate to cover ink edges, Gaussian feather to blend). The
dilate=8/feather=5 defaults are empirically tuned against paired tattoo/clean data, so this reuses
that function rather than a generic blur node. Sits between a localizer MASK (commodity
Grounded-SAM or DeinkSegFormer) and DeinkInpaint, which treats the incoming mask as final and does
not refine again.
"""

from __future__ import annotations

from ..convert import mask_to_bool_np, np_to_mask


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
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "refine"
    CATEGORY = "deink"

    def refine(self, mask, dilate, feather):
        from deink.pipeline import refine_mask

        raw = mask_to_bool_np(mask)  # binarize before growing
        refined = refine_mask(raw, dilate=dilate, feather=feather)  # float32 (H,W) in [0,1]
        return (np_to_mask(refined),)
