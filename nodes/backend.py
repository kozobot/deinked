"""Inpaint-backend provider nodes — configure a backend in the workflow, wire it into DeinkInpaint.

Each node emits a ``DEINK_BACKEND`` descriptor (a plain dict) that carries the backend name, the
minimum mask-component size that should be routed to it, and any backend-specific inpaint kwargs.
DeinkInpaint accepts several of these and auto-routes each connected mask component to the backend
whose ``min_area_frac`` best matches the component's size — so "which backend, with what settings"
lives in the graph, not on the inpaint node. Adding a new backend later is just another provider
node here (plus a case in ``deink.inpaint.Inpainter``); DeinkInpaint needs no changes.

``DEINK_BACKEND`` is a custom ComfyUI link type: no registration is needed, only that producer
``RETURN_TYPES`` and consumer ``INPUT_TYPES`` use the same string. The descriptor schema is::

    {"name": "lama" | "sdxl" | "twostage",  # consumed by deink.pipeline._fill / Inpainter.inpaint
     "min_area_frac": float,     # route components >= this fraction of the image area here
     "kwargs": {...}}            # backend-specific inpaint kwargs (empty for lama)
"""

from __future__ import annotations


class DeinkLamaBackend:
    """LaMa backend: fast, strong plain-skin texture fill. No inpaint params to configure."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_area_frac": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Route mask components >= this fraction of the image area to LaMa. "
                               "0 = the smallest / fallback tier."}),
            },
        }

    RETURN_TYPES = ("DEINK_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "make"
    CATEGORY = "deink"

    def make(self, min_area_frac):
        return ({"name": "lama", "min_area_frac": min_area_frac, "kwargs": {}},)


class DeinkSdxlBackend:
    """SDXL backend: slower diffusion fill that reconstructs structure across large holes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_area_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Route mask components >= this fraction of the image area to SDXL."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Blank uses deink's skin default."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "strength": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 150, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "-1 = random."}),
            },
        }

    RETURN_TYPES = ("DEINK_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "make"
    CATEGORY = "deink"

    def make(self, min_area_frac, prompt, negative_prompt, strength, guidance_scale, steps, seed):
        # Same kwargs assembly the DeinkInpaint node used to do inline, so SDXL semantics are
        # unchanged: blank prompt/negative fall back to deink's skin defaults; seed < 0 = random.
        kwargs = {"strength": strength, "guidance_scale": guidance_scale,
                  "num_inference_steps": steps}
        if prompt:
            kwargs["prompt"] = prompt
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None and seed >= 0:
            kwargs["seed"] = seed
        return ({"name": "sdxl", "min_area_frac": min_area_frac, "kwargs": kwargs},)


class DeinkTwoStageBackend:
    """Two-stage backend: LaMa roughs in structure, then a low-strength SDXL pass adds skin
    texture over it — coherent limbs without the plastic look of a single high-strength pass.

    Same params as the SDXL backend (they configure the second, texture stage), but ``strength``
    defaults to 0.5 so SDXL refines the LaMa fill rather than regenerating from scratch. Best
    wired for large / limb-spanning regions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_area_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Route mask components >= this fraction of the image area to the "
                               "two-stage fill."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Blank uses deink's skin default."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "SDXL texture-stage strength over the LaMa fill (~0.4-0.6)."}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 150, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "-1 = random."}),
            },
        }

    RETURN_TYPES = ("DEINK_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "make"
    CATEGORY = "deink"

    def make(self, min_area_frac, prompt, negative_prompt, strength, guidance_scale, steps, seed):
        # Kwargs flow to the SDXL (texture) stage of deink's inpaint_twostage; same assembly
        # rules as the SDXL backend. LaMa's structure stage takes no kwargs.
        kwargs = {"strength": strength, "guidance_scale": guidance_scale,
                  "num_inference_steps": steps}
        if prompt:
            kwargs["prompt"] = prompt
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None and seed >= 0:
            kwargs["seed"] = seed
        return ({"name": "twostage", "min_area_frac": min_area_frac, "kwargs": kwargs},)
