"""Inpaint-backend provider nodes — configure a backend in the workflow, wire it into DeinkInpaint.

Each node emits a ``DEINK_BACKEND`` descriptor (a plain dict) that carries the backend name, the
minimum mask-component size that should be routed to it, and any backend-specific inpaint kwargs.
DeinkInpaint accepts several of these and auto-routes each connected mask component to the backend
whose ``min_area_frac`` best matches the component's size — so "which backend, with what settings"
lives in the graph, not on the inpaint node. Adding a new backend later is just another provider
node here (plus a case in ``deink.inpaint.Inpainter``); DeinkInpaint needs no changes.

``DEINK_BACKEND`` is a custom ComfyUI link type: no registration is needed, only that producer
``RETURN_TYPES`` and consumer ``INPUT_TYPES`` use the same string. The descriptor schema is::

    {"name": "lama" | "sdxl" | "sdxl_controlnet" | "flux" | "twostage" | "migan" | "mat",  # Inpainter
     "min_area_frac": float,     # route components >= this fraction of the image area here
     "kwargs": {...}}            # backend-specific inpaint kwargs (empty for lama/migan)
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


class DeinkSdxlControlNetBackend:
    """SDXL + depth-ControlNet backend: SDXL guided by a depth map of the surrounding limb.

    A depth map is auto-estimated (Depth-Anything, transformers-native) from the crop and fed to
    an SDXL depth ControlNet so the generated skin follows the actual arm/leg geometry across the
    hole — curing the flat/warped anatomy plain SDXL invents on a limb-sized mask. Same params as
    the SDXL backend plus ``controlnet_conditioning_scale`` (how hard to hold the depth structure).
    Best wired for large / limb-spanning regions. Weights (small SDXL depth CN + depth model)
    download lazily from un-gated mirrors — no `hf auth login`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_area_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Route mask components >= this fraction of the image area here."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Blank uses deink's skin default."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "strength": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 150, "step": 1}),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0,
                    "step": 0.05, "tooltip": "How hard to hold the depth structure (higher = more)."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "-1 = random."}),
            },
        }

    RETURN_TYPES = ("DEINK_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "make"
    CATEGORY = "deink"

    def make(self, min_area_frac, prompt, negative_prompt, strength, guidance_scale, steps,
             controlnet_conditioning_scale, seed):
        # Same kwargs assembly as the SDXL backend, plus the ControlNet conditioning scale that
        # flows to deink's inpaint_sdxl_controlnet.
        kwargs = {"strength": strength, "guidance_scale": guidance_scale,
                  "num_inference_steps": steps,
                  "controlnet_conditioning_scale": controlnet_conditioning_scale}
        if prompt:
            kwargs["prompt"] = prompt
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None and seed >= 0:
            kwargs["seed"] = seed
        return ({"name": "sdxl_controlnet", "min_area_frac": min_area_frac, "kwargs": kwargs},)


class DeinkFluxBackend:
    """FLUX.1 Fill backend: SOTA diffusion inpainter, stronger structure/texture than SDXL.

    Loads a GGUF-quantized FLUX.1 Fill [dev] transformer so the 12B model fits 16 GB (the base
    repo is *gated* — accept its license and `hf auth login` before first use). FLUX is
    guidance-distilled, so there is **no negative prompt** and ``guidance_scale`` runs high (~30);
    the skin default is steered by the positive prompt alone."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_area_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Route mask components >= this fraction of the image area to FLUX."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Blank uses deink's skin default. FLUX takes no negative prompt."}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": "FLUX Fill wants high guidance (~30)."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 150, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "-1 = random."}),
            },
        }

    RETURN_TYPES = ("DEINK_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "make"
    CATEGORY = "deink"

    def make(self, min_area_frac, prompt, guidance_scale, steps, strength, seed):
        # Kwargs flow to deink's inpaint_flux; same assembly as the SDXL backend but with no
        # negative_prompt (FLUX is guidance-distilled and ignores it).
        kwargs = {"guidance_scale": guidance_scale, "num_inference_steps": steps,
                  "strength": strength}
        if prompt:
            kwargs["prompt"] = prompt
        if seed is not None and seed >= 0:
            kwargs["seed"] = seed
        return ({"name": "flux", "min_area_frac": min_area_frac, "kwargs": kwargs},)


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


class DeinkMiganBackend:
    """MI-GAN backend: fast feed-forward large-hole fill. No inpaint params to configure.

    A small pure-PyTorch generator (no diffusion) that reconstructs across big holes faster
    than the diffusion backends — a lightweight alternative to the two-stage/SDXL large tier.
    Weights download lazily from an un-gated mirror on first use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_area_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Route mask components >= this fraction of the image area to MI-GAN."}),
            },
        }

    RETURN_TYPES = ("DEINK_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "make"
    CATEGORY = "deink"

    def make(self, min_area_frac):
        return ({"name": "migan", "min_area_frac": min_area_frac, "kwargs": {}},)


class DeinkMatBackend:
    """MAT backend: feed-forward StyleGAN large-hole specialist (deink's default ``auto`` large tier).

    Pure-PyTorch, no diffusion — reconstructs structure across limb-sized holes fast. ``seed``
    redraws its latent for a different plausible fill (default: deterministic). Weights download
    lazily from an un-gated mirror on first use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_area_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Route mask components >= this fraction of the image area to MAT."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "-1 = deterministic (fixed latent)."}),
            },
        }

    RETURN_TYPES = ("DEINK_BACKEND",)
    RETURN_NAMES = ("backend",)
    FUNCTION = "make"
    CATEGORY = "deink"

    def make(self, min_area_frac, seed):
        kwargs = {}
        if seed is not None and seed >= 0:
            kwargs["seed"] = seed
        return ({"name": "mat", "min_area_frac": min_area_frac, "kwargs": kwargs},)
