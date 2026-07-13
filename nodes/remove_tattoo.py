"""DeinkRemoveTattoo — the whole pipeline in one node (the marimo app's equivalent).

Calls ``deink.pipeline.remove_tattoo`` (localize -> refine -> inpaint -> composite) for users who
don't want to wire a graph. Exposes the same controls as ``app.py``: backend, localizer (seg /
box / seg+box — deink's own GroundingDINO+SAM path, no commodity node needed), detector, the
detection thresholds + tiling, prompt, seg_threshold, dilate/feather, and an optional ``mask``
input that bypasses detection (the app's "upload your own mask" path). For the composable box path,
wire the commodity Grounded-SAM node into DeinkRefineMask -> DeinkInpaint instead.
"""

from __future__ import annotations

import numpy as np

from ..convert import mask_to_bool_np, np_to_mask, pil_to_tensor, tensor_to_pil
from ..models import get_inpainter, get_mask_segmenter, get_segmenter


class DeinkRemoveTattoo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "backend": (["lama", "sdxl", "auto", "twostage"], {"default": "auto"}),
                "localizer": (["seg", "box", "seg+box"], {"default": "seg"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional: supply a mask (white = remove) to bypass "
                          "detection entirely — the app's 'upload your own mask' path."}),
                "prompt": ("STRING", {"default": "a tattoo.",
                            "tooltip": "Detection prompt for the box/seg+box path."}),
                "seg_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilate": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 128, "step": 1}),
                "crop": ("BOOLEAN", {"default": True}),
                "crop_pad": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.1}),
                "detector": (["gdino", "owlv2", "ensemble"], {"default": "gdino",
                             "tooltip": "Open-vocab detector for the box/seg+box path."}),
                "tile": ("BOOLEAN", {"default": False,
                          "tooltip": "Tiled detection (higher recall on small/faint tattoos, slower)."}),
                "box_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_area_frac": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Drop detection boxes larger than this fraction of the image."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = "deink"

    def run(self, image, backend, localizer, mask=None, prompt="a tattoo.", seg_threshold=0.5,
            dilate=8, feather=5, crop=True, crop_pad=0.5, detector="gdino", tile=False,
            box_threshold=0.25, text_threshold=0.2, max_area_frac=0.25):
        from deink.pipeline import remove_tattoo

        pil = tensor_to_pil(image)
        supplied = None if mask is None else mask_to_bool_np(mask)
        # With a supplied mask, detection is skipped, so no localizer models are needed.
        needs_box = supplied is None and localizer in ("box", "seg+box")
        needs_seg = supplied is None and localizer in ("seg", "seg+box")
        result = remove_tattoo(
            pil,
            backend=backend,
            mask=supplied,
            localizer=localizer,
            prompt=prompt,
            seg_threshold=seg_threshold,
            dilate=dilate,
            feather=feather,
            crop=crop,
            crop_pad=crop_pad,
            detector=detector,
            tile=tile,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_area_frac=max_area_frac,
            inpainter=get_inpainter(),
            mask_segmenter=get_mask_segmenter(None, seg_threshold) if needs_seg else None,
            segmenter=get_segmenter() if needs_box else None,
        )
        # result.mask is a PIL 'L'; convert straight through as a float mask.
        mask_np = np.asarray(result.mask, dtype=np.float32) / 255.0
        return (pil_to_tensor(result.image), np_to_mask(mask_np))
