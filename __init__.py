"""deink — a ComfyUI plugin for tattoo removal (segment + inpaint).

The repo root is the installable ComfyUI custom-node package; the original ``deink`` pipeline
package and its training tooling live under ``core/``. We add ``core/`` to ``sys.path`` here so the
nodes can import and reuse the canonical ``deink`` code (no vendored copy), then register the nodes.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")

# Make the canonical `deink` package importable from core/.
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

# Default the SegFormer checkpoint to the in-repo location unless the user overrides it, so the
# node finds the trained model regardless of ComfyUI's working directory.
os.environ.setdefault(
    "DEINK_TATTOOSEG_DIR", os.path.join(_CORE, "data", "models", "tattoo-segformer")
)

from .nodes import (  # noqa: E402  (import after sys.path/env setup)
    DeinkInpaint,
    DeinkRefineMask,
    DeinkRemoveTattoo,
    DeinkSegFormer,
    DeinkSplitMaskBySize,
)

NODE_CLASS_MAPPINGS = {
    "DeinkSegFormer": DeinkSegFormer,
    "DeinkRefineMask": DeinkRefineMask,
    "DeinkSplitMaskBySize": DeinkSplitMaskBySize,
    "DeinkInpaint": DeinkInpaint,
    "DeinkRemoveTattoo": DeinkRemoveTattoo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeinkSegFormer": "Deink SegFormer (tattoo mask)",
    "DeinkRefineMask": "Deink Refine Mask",
    "DeinkSplitMaskBySize": "Deink Split Mask by Size",
    "DeinkInpaint": "Deink Inpaint",
    "DeinkRemoveTattoo": "Deink Remove Tattoo (all-in-one)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
