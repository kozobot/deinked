"""Inpainting backends: LaMa (fast texture fill) and SDXL (semantic fill).

Both are loaded lazily and kept resident once used. On a 16 GB card, run one backend at
a time — SDXL inpainting is enabled with model CPU offload so it fits comfortably.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import ensure_pil, get_device, mask_to_pil

SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_SD_PROMPT = "bare skin, natural skin texture, seamless, photorealistic"
DEFAULT_SD_NEGATIVE = "tattoo, ink, drawing, text, blurry, deformed, artifacts"


class Inpainter:
    def __init__(self, device=None):
        self.device = device or get_device()
        self._lama = None
        self._sdxl = None

    # --- lazy loaders -------------------------------------------------------
    def _load_lama(self):
        if self._lama is None:
            from simple_lama_inpainting import SimpleLama

            self._lama = SimpleLama(device=self.device)
        return self._lama

    def _load_sdxl(self):
        if self._sdxl is None:
            import torch
            from diffusers import AutoPipelineForInpainting

            use_cuda = getattr(self.device, "type", str(self.device)) == "cuda"
            dtype = torch.float16 if use_cuda else torch.float32
            pipe = AutoPipelineForInpainting.from_pretrained(
                SDXL_INPAINT_ID, torch_dtype=dtype, variant="fp16" if use_cuda else None
            )
            if use_cuda:
                # Offload keeps peak VRAM well under 16 GB.
                pipe.enable_model_cpu_offload()
            self._sdxl = pipe
        return self._sdxl

    # --- backends -----------------------------------------------------------
    def inpaint_lama(self, image, mask) -> Image.Image:
        image = ensure_pil(image)
        mask_img = mask if isinstance(mask, Image.Image) else mask_to_pil(np.asarray(mask))
        lama = self._load_lama()
        result = lama(image, mask_img.convert("L"))
        return result.convert("RGB").resize(image.size)

    def inpaint_sdxl(
        self,
        image,
        mask,
        prompt: str = DEFAULT_SD_PROMPT,
        negative_prompt: str = DEFAULT_SD_NEGATIVE,
        strength: float = 0.99,
        guidance_scale: float = 8.0,
        num_inference_steps: int = 30,
        seed: int | None = None,
    ) -> Image.Image:
        """SDXL inpaint at 1024px, composited back onto the original at full resolution."""
        import torch

        image = ensure_pil(image)
        mask_img = (
            mask if isinstance(mask, Image.Image) else mask_to_pil(np.asarray(mask))
        ).convert("L")
        pipe = self._load_sdxl()

        # SDXL works best at 1024; run there then paste the result back at native size.
        work = 1024
        img_small = image.resize((work, work))
        mask_small = mask_img.resize((work, work))
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_small,
            mask_image=mask_small,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        out = out.resize(image.size)
        # Only replace masked pixels; keep the rest bit-identical to the input.
        return Image.composite(out, image, mask_img)

    def inpaint(self, image, mask, backend: str = "lama", **kwargs) -> Image.Image:
        if backend == "lama":
            return self.inpaint_lama(image, mask)
        if backend == "sdxl":
            return self.inpaint_sdxl(image, mask, **kwargs)
        raise ValueError(f"Unknown inpaint backend: {backend!r} (use 'lama' or 'sdxl')")
