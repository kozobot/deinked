"""DeinkInpaint — fill a masked region with LaMa / SDXL / auto, at the backend's native res.

We own this node (rather than reuse a commodity inpainter) because the two behaviours that make
deink's fills clean must *wrap* the backend call:

- **crop-to-native** (``deink.pipeline._region_bbox`` + ``_NATIVE_RES``): each pass runs on a
  padded window around the mask at the backend's native resolution, so a small tattoo on a large
  photo isn't squashed into 1024px before SDXL sees it.
- **bit-identical composite** (``deink.pipeline._fill``): pixels outside the mask stay identical
  to the input.

``backend="auto"`` routes each connected mask component by size (small -> LaMa, large -> SDXL) via
``_split_by_component_size``, mirroring ``remove_tattoo``.

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
                "backend": (["lama", "sdxl", "auto"], {"default": "lama"}),
                "crop": ("BOOLEAN", {"default": True,
                          "tooltip": "Inpaint a native-res window around the mask (recommended)."}),
                "crop_pad": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.1,
                             "tooltip": "Context around the mask, as a fraction of its extent."}),
            },
            "optional": {
                "auto_area_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                                   "tooltip": "backend=auto: component >= this fraction -> SDXL."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                            "tooltip": "SDXL only; blank uses deink's skin default."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "strength": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 150, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                          "tooltip": "-1 = random."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "deink"

    def inpaint(self, image, mask, backend, crop, crop_pad, auto_area_frac=0.02,
                prompt="", negative_prompt="", strength=0.99, guidance_scale=8.0,
                steps=30, seed=-1):
        from deink.pipeline import _NATIVE_RES, _fill, _region_bbox, _split_by_component_size

        pil = tensor_to_pil(image)
        refined = mask_to_float_np(mask)  # treat incoming mask as final (H,W) float
        if not (refined > 0).any():
            return (image,)  # nothing masked — passthrough, bit-identical

        inpainter = get_inpainter()

        sdxl_kwargs = {"strength": strength, "guidance_scale": guidance_scale,
                       "num_inference_steps": steps}
        if prompt:
            sdxl_kwargs["prompt"] = prompt
        if negative_prompt:
            sdxl_kwargs["negative_prompt"] = negative_prompt
        if seed is not None and seed >= 0:
            sdxl_kwargs["seed"] = seed

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

        if backend == "auto":
            binary = refined > 0.5
            small, large = _split_by_component_size(binary, pil.size[0] * pil.size[1], auto_area_frac)
            result = pil
            if small.any():
                result = fill_region(result, refined * small, "lama")
            if large.any():
                result = fill_region(result, refined * large, "sdxl", **sdxl_kwargs)
        elif backend == "sdxl":
            result = fill_region(pil, refined, "sdxl", **sdxl_kwargs)
        else:
            result = fill_region(pil, refined, "lama")

        return (pil_to_tensor(result),)
