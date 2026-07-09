"""Node classes for the deink ComfyUI plugin."""

from .inpaint import DeinkInpaint
from .refine_mask import DeinkRefineMask
from .remove_tattoo import DeinkRemoveTattoo
from .segformer import DeinkSegFormer
from .split import DeinkSplitMaskBySize

__all__ = [
    "DeinkInpaint",
    "DeinkRefineMask",
    "DeinkRemoveTattoo",
    "DeinkSegFormer",
    "DeinkSplitMaskBySize",
]
