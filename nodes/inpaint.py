"""DeinkInpaint — fill a masked region using backend(s) wired in from the workflow.

Backends are configured on their own provider nodes (DeinkLamaBackend / DeinkSdxlBackend) and
plugged into this node's ``DEINK_BACKEND`` sockets; this node just auto-routes and runs them. We
own this node (rather than reuse a commodity inpainter) because the two behaviours that make
deink's fills clean must *wrap* each backend call:

- **crop-to-native** (``deink.pipeline._region_bbox`` + ``_NATIVE_RES``): each pass runs on a
  padded window around the mask at the backend's native resolution, so a small tattoo on a large
  photo isn't squashed into 1024px before SDXL sees it.
- **bit-identical composite** (``deink.pipeline._fill``): pixels outside the mask stay identical
  to the input.

**Auto-routing:** the mask is split into connected components, and each is routed to the wired
backend with the greatest ``min_area_frac`` that is still ``<=`` the component's image-area
fraction (``_route_by_component_size``), falling back to the smallest-``min_area_frac`` backend.
Passes run in ascending ``min_area_frac`` order, so a small-region backend (e.g. LaMa) runs first
and a large-region backend (e.g. SDXL) operates on its output — mirroring the old ``backend="auto"``
behaviour. With no backend wired, it falls back to a plain LaMa fill so the node works standalone.

The incoming MASK is treated as the **final** mask (feather included) — refine it upstream with
DeinkRefineMask; this node does not dilate/feather again.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..convert import mask_to_float_np, pil_to_tensor, tensor_to_pil
from ..models import get_inpainter


class DeinkInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "crop": ("BOOLEAN", {"default": True,
                          "tooltip": "Inpaint a native-res window around the mask (recommended)."}),
                "crop_pad": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.1,
                             "tooltip": "Context around the mask, as a fraction of its extent."}),
            },
            "optional": {
                "backend_1": ("DEINK_BACKEND", {
                    "tooltip": "A backend from a DeinkLamaBackend / DeinkSdxlBackend node. "
                               "Wire several; each mask region is routed by its min_area_frac."}),
                "backend_2": ("DEINK_BACKEND",),
                "backend_3": ("DEINK_BACKEND",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "deink"

    def inpaint(self, image, mask, crop, crop_pad, backend_1=None, backend_2=None, backend_3=None):
        from deink.pipeline import _NATIVE_RES, _fill, _region_bbox, _route_by_component_size

        pil = tensor_to_pil(image)
        refined = mask_to_float_np(mask)  # treat incoming mask as final (H,W) float
        if not (refined > 0).any():
            return (image,)  # nothing masked — passthrough, bit-identical

        backends = [b for b in (backend_1, backend_2, backend_3) if b is not None]
        if not backends:
            # No backend wired — behave like a plain LaMa fill so the node works standalone.
            backends = [{"name": "lama", "min_area_frac": 0.0, "kwargs": {}}]

        inpainter = get_inpainter()

        def fill_region(img: Image.Image, region: np.ndarray, be: str, **kw) -> Image.Image:
            """Mirror of deink._inpaint_region minus the internal refine: the given region float
            mask is used as-is, cropped to native res, filled, and pasted back bit-identically."""
            if crop:
                box = _region_bbox(region, img.size, pad_frac=crop_pad,
                                   min_size=_NATIVE_RES.get(be, 0))
                if box is not None and box != (0, 0, img.size[0], img.size[1]):
                    l, t, r, b = box
                    filled = _fill(inpainter, img.crop(box), region[t:b, l:r], be, **kw)
                    out = img.copy()
                    out.paste(filled, box)
                    return out
            return _fill(inpainter, img, region, be, **kw)

        binary = refined > 0.5
        parts = _route_by_component_size(
            binary, pil.size[0] * pil.size[1], [b["min_area_frac"] for b in backends]
        )
        result = pil
        for b, part in sorted(zip(backends, parts), key=lambda bp: bp[0]["min_area_frac"]):
            if part.any():
                result = fill_region(result, refined * part, b["name"], **b["kwargs"])

        return (pil_to_tensor(result),)
