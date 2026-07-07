"""deinked — segment-and-inpaint tattoo removal.

Public API:
    from deink import remove_tattoo, TattooSegmenter, Inpainter
"""

from .pipeline import remove_tattoo
from .segment import TattooSegmenter
from .tattooseg import TattooMaskSegmenter
from .inpaint import Inpainter

__all__ = ["remove_tattoo", "TattooSegmenter", "TattooMaskSegmenter", "Inpainter"]
