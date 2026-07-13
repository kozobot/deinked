"""Inpainting backends: LaMa (fast texture fill), SDXL / FLUX.1 Fill (diffusion fills), and
MI-GAN / MAT (fast feed-forward large-hole specialists).

All are loaded lazily and kept resident once used. On a 16 GB card, run one backend at
a time — SDXL and FLUX inpainting are enabled with model CPU offload so they fit comfortably
(FLUX additionally loads a GGUF-quantized transformer). MI-GAN and MAT are small pure-PyTorch
generators (vendored under :mod:`deink.vendor`, no extra pip deps, un-gated weights that
download lazily — override the URLs via ``DEINK_MIGAN_URL`` / ``DEINK_MAT_URL``).
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .utils import ensure_pil, get_device, mask_to_pil

SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_SD_PROMPT = "bare skin, natural skin texture, seamless, photorealistic"
DEFAULT_SD_NEGATIVE = "tattoo, ink, drawing, text, blurry, deformed, artifacts"
# Second-stage SDXL strength for the two-stage fill: low enough to keep the LaMa structure it
# denoises from, high enough to lay down real skin texture over the smear.
DEFAULT_TWOSTAGE_STRENGTH = 0.5

# FLUX.1 Fill [dev] — SOTA inpainting, stronger structure/texture than SDXL. The base repo (VAE /
# text encoders / scheduler) is *gated*: accept its license and `hf auth login` before first use.
# The 12B transformer won't fit 16 GB unquantized, so it is loaded from a GGUF single-file build
# (city96's non-gated mirror) and paired with the base pipeline's other components. Override the
# GGUF (e.g. a smaller Q-level, or a local path) via DEINK_FLUX_GGUF.
FLUX_FILL_ID = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_GGUF_URL = os.environ.get(
    "DEINK_FLUX_GGUF",
    "https://huggingface.co/city96/FLUX.1-Fill-dev-gguf/blob/main/flux1-fill-dev-Q4_K_S.gguf",
)
# FLUX is guidance-distilled; the Fill model wants a high guidance_scale (~30) and no negative.
DEFAULT_FLUX_GUIDANCE = 30.0


class Inpainter:
    def __init__(self, device=None):
        self.device = device or get_device()
        self._lama = None
        self._sdxl = None
        self._flux = None
        self._migan = None
        self._mat = None

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

    def _load_flux(self):
        if self._flux is None:
            import torch
            from diffusers import (
                FluxFillPipeline,
                FluxTransformer2DModel,
                GGUFQuantizationConfig,
            )

            use_cuda = getattr(self.device, "type", str(self.device)) == "cuda"
            # FLUX runs in bfloat16; the transformer is dequantized from GGUF on the fly.
            compute_dtype = torch.bfloat16 if use_cuda else torch.float32
            transformer = FluxTransformer2DModel.from_single_file(
                FLUX_GGUF_URL,
                quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
                torch_dtype=compute_dtype,
                config=FLUX_FILL_ID,
                subfolder="transformer",
            )
            pipe = FluxFillPipeline.from_pretrained(
                FLUX_FILL_ID, transformer=transformer, torch_dtype=compute_dtype
            )
            if use_cuda:
                # Offload the resident text encoders / VAE so peak VRAM (with the GGUF
                # transformer) stays under 16 GB — same discipline as SDXL, don't also .to(cuda).
                pipe.enable_model_cpu_offload()
            self._flux = pipe
        return self._flux

    def _load_migan(self):
        if self._migan is None:
            from .vendor.migan import load_migan

            self._migan = load_migan(self.device)
        return self._migan

    def _load_mat(self):
        if self._mat is None:
            from .vendor.mat import load_mat

            # (generator, fixed latent z, null label, dtype) — reused across calls.
            self._mat = load_mat(self.device)
        return self._mat

    # --- backends -----------------------------------------------------------
    def inpaint_lama(self, image, mask) -> Image.Image:
        image = ensure_pil(image)
        mask_img = mask if isinstance(mask, Image.Image) else mask_to_pil(np.asarray(mask))
        lama = self._load_lama()
        result = lama(image, mask_img.convert("L"))
        return result.convert("RGB").resize(image.size)

    def inpaint_migan(self, image, mask) -> Image.Image:
        """MI-GAN feed-forward fill — fast, feed-forward, strong on large holes.

        Like LaMa, this returns the raw filled window (no internal composite); the caller
        (``pipeline._fill``) composites it back with the feathered mask so pixels outside
        the mask stay bit-identical.
        """
        from .vendor.migan import migan_infer

        image = ensure_pil(image)
        mask_img = (
            mask if isinstance(mask, Image.Image) else mask_to_pil(np.asarray(mask))
        ).convert("L")
        model = self._load_migan()
        out = migan_infer(
            model, np.asarray(image.convert("RGB")), np.asarray(mask_img), self.device
        )
        return Image.fromarray(out)

    def inpaint_mat(self, image, mask, seed: int | None = None) -> Image.Image:
        """MAT feed-forward fill — StyleGAN-based large-hole specialist.

        ``seed`` redraws MAT's latent for a different plausible fill (default: the fixed
        latent from load, so results are reproducible). Returns the raw filled window; the
        caller composites (see :meth:`inpaint_migan`)."""
        from .vendor.mat import mat_infer

        image = ensure_pil(image)
        mask_img = (
            mask if isinstance(mask, Image.Image) else mask_to_pil(np.asarray(mask))
        ).convert("L")
        G, z, label, dtype = self._load_mat()
        out = mat_infer(
            G, z, label, dtype,
            np.asarray(image.convert("RGB")), np.asarray(mask_img), self.device, seed=seed,
        )
        return Image.fromarray(out)

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

    def inpaint_flux(
        self,
        image,
        mask,
        prompt: str = DEFAULT_SD_PROMPT,
        guidance_scale: float = DEFAULT_FLUX_GUIDANCE,
        num_inference_steps: int = 30,
        strength: float = 1.0,
        seed: int | None = None,
        max_sequence_length: int = 512,
    ) -> Image.Image:
        """FLUX.1 Fill inpaint at 1024px, composited back onto the original at full resolution.

        Mirrors :meth:`inpaint_sdxl` (run square at 1024, resize back, composite on the mask so
        pixels outside it stay bit-identical). FLUX is guidance-distilled — it takes no
        ``negative_prompt`` and wants a high ``guidance_scale`` (~30) — so the skin default is
        steered by the positive ``prompt`` alone.
        """
        import torch

        image = ensure_pil(image)
        mask_img = (
            mask if isinstance(mask, Image.Image) else mask_to_pil(np.asarray(mask))
        ).convert("L")
        pipe = self._load_flux()

        # FLUX Fill works at 1024; run there then paste the result back at native size.
        work = 1024
        img_small = image.resize((work, work))
        mask_small = mask_img.resize((work, work))
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        out = pipe(
            prompt=prompt,
            image=img_small,
            mask_image=mask_small,
            height=work,
            width=work,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]
        out = out.resize(image.size)
        # Only replace masked pixels; keep the rest bit-identical to the input.
        return Image.composite(out, image, mask_img)

    def inpaint_twostage(
        self,
        image,
        mask,
        strength: float = DEFAULT_TWOSTAGE_STRENGTH,
        **sdxl_kwargs,
    ) -> Image.Image:
        """Two-stage fill: LaMa roughs in low-frequency structure, then a low-strength SDXL
        pass adds skin texture over that result.

        LaMa extends nearby texture into the hole but has no semantics, so on a limb-sized hole
        it collapses to a smear; a single high-strength SDXL pass reconstructs structure but can
        look plastic / re-invent detail. Running SDXL at ``strength`` ~0.5 over the LaMa fill
        seeds its denoising *from that fill* (noise ∝ strength added to the masked latents), so
        it refines the roughed-in structure into natural skin instead of generating from scratch.

        ``mask`` may be soft (feathered): the LaMa stage hard-thresholds it (LaMa wants a binary
        hole) and is feathered back on; the SDXL stage uses the soft mask directly and composites
        internally, so pixels outside the mask stay bit-identical to ``image``. ``**sdxl_kwargs``
        (prompt/negative_prompt/guidance_scale/num_inference_steps/seed) flow to the SDXL stage
        only.
        """
        image = ensure_pil(image)
        mask_img = (
            mask if isinstance(mask, Image.Image) else mask_to_pil(np.asarray(mask))
        ).convert("L")
        # Stage 1 — structure: LaMa on a hard binary hole, feathered back onto the input.
        hard = mask_to_pil(np.asarray(mask_img) > 127)
        interim = self.inpaint_lama(image, hard)
        interim = Image.composite(interim, image, mask_img)
        # Stage 2 — texture: low-strength SDXL over the LaMa result.
        return self.inpaint_sdxl(interim, mask_img, strength=strength, **sdxl_kwargs)

    def inpaint(self, image, mask, backend: str = "lama", **kwargs) -> Image.Image:
        if backend == "lama":
            return self.inpaint_lama(image, mask)
        if backend == "sdxl":
            return self.inpaint_sdxl(image, mask, **kwargs)
        if backend == "flux":
            return self.inpaint_flux(image, mask, **kwargs)
        if backend == "twostage":
            return self.inpaint_twostage(image, mask, **kwargs)
        if backend == "migan":
            return self.inpaint_migan(image, mask)
        if backend == "mat":
            return self.inpaint_mat(image, mask, **kwargs)
        raise ValueError(
            f"Unknown inpaint backend: {backend!r} "
            "(use 'lama', 'sdxl', 'flux', 'twostage', 'migan', or 'mat')"
        )
