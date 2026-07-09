"""DeinkSplitMaskBySize — route mask regions by size, as visible graph topology.

Wraps ``deink.pipeline._split_by_component_size``: partitions a mask's connected components into
``small`` and ``large`` by area fraction, so a graph can send small blobs to LaMa and large /
limb-spanning blobs to SDXL (or any other model) explicitly. The same policy is available inside
DeinkInpaint as ``backend="auto"`` for the one-node path; this node exposes it for composability.
"""

from __future__ import annotations

from ..convert import mask_to_bool_np, mask_to_float_np, np_to_mask


class DeinkSplitMaskBySize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "auto_area_frac": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                     "tooltip": "A component covering >= this fraction of the image is 'large'."},
                ),
            },
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("small", "large")
    FUNCTION = "split"
    CATEGORY = "deink"

    def split(self, mask, auto_area_frac):
        from deink.pipeline import _split_by_component_size

        binary = mask_to_bool_np(mask)
        soft = mask_to_float_np(mask)  # keep any feather, applied per side below
        h, w = binary.shape
        small, large = _split_by_component_size(binary, h * w, auto_area_frac)
        # Preserve soft edges within each side by masking the (possibly feathered) input.
        return (np_to_mask(soft * small), np_to_mask(soft * large))
