"""Vendored, minimal-footprint inpainting model code (no extra pip deps).

Holds the feed-forward large-hole fill backends whose upstream implementations only
ship inside heavier packages we can't depend on (``iopaint`` pins ``diffusers==0.27.2``,
which conflicts with deink's modern diffusers used by the FLUX Fill backend). Each model
here is pure-PyTorch — no custom CUDA extension is built — and its weights download lazily
from un-gated GitHub-release assets (``github.com/Sanster/models``), so nothing is imported
until a backend is first used.

- :mod:`deink.vendor.migan` — MI-GAN (Picsart, ICCV 2023), a TorchScript-traced generator.
- :mod:`deink.vendor.mat` — MAT (Li et al., CVPR 2022), StyleGAN-based, loaded from a state dict.

Both are adapted from IOPaint (https://github.com/Sanster/IOPaint, Apache-2.0); see each
module's header for provenance.
"""
