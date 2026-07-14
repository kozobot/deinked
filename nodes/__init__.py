"""Node classes for the deink ComfyUI plugin."""

from .backend import (
    DeinkFluxBackend,
    DeinkLamaBackend,
    DeinkMatBackend,
    DeinkMiganBackend,
    DeinkSdxlBackend,
    DeinkSdxlControlNetBackend,
    DeinkTwoStageBackend,
)
from .inpaint import DeinkInpaint
from .refine_mask import DeinkRefineMask
from .remove_tattoo import DeinkRemoveTattoo
from .segformer import DeinkSegFormer
from .split import DeinkSplitMaskBySize

__all__ = [
    "DeinkFluxBackend",
    "DeinkInpaint",
    "DeinkLamaBackend",
    "DeinkMatBackend",
    "DeinkMiganBackend",
    "DeinkRefineMask",
    "DeinkRemoveTattoo",
    "DeinkSdxlBackend",
    "DeinkSdxlControlNetBackend",
    "DeinkSegFormer",
    "DeinkSplitMaskBySize",
    "DeinkTwoStageBackend",
]
