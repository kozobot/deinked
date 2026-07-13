"""Node classes for the deink ComfyUI plugin."""

from .backend import DeinkLamaBackend, DeinkSdxlBackend
from .inpaint import DeinkInpaint
from .refine_mask import DeinkRefineMask
from .remove_tattoo import DeinkRemoveTattoo
from .segformer import DeinkSegFormer
from .split import DeinkSplitMaskBySize

__all__ = [
    "DeinkInpaint",
    "DeinkLamaBackend",
    "DeinkRefineMask",
    "DeinkRemoveTattoo",
    "DeinkSdxlBackend",
    "DeinkSegFormer",
    "DeinkSplitMaskBySize",
]
