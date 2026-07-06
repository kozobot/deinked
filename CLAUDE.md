# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`deinked` removes tattoos from images (video is a future goal). It **used to** train a
bespoke fastai GAN (a NoGAN port of [SkinDeep](https://github.com/vijishmadhavan/SkinDeep));
that never fully worked and has been retired. It is now a **segment-and-inpaint pipeline**
built on pretrained foundation models — no training. The code is a small importable Python
package (`deink/`) plus a marimo app, driven from a local conda environment.

## Running

Local conda env (no Docker), targeting an NVIDIA Blackwell GPU (sm_120, e.g. RTX 5070 Ti):

```bash
mamba env create -f environment.yml && conda activate deinked
```

PyTorch comes from a CUDA wheel that includes `sm_120` kernels. There is no CPU-only path
worth using — the diffusion/segmentation models assume a GPU.

- App: `marimo run app.py` (or `marimo edit app.py`).
- Smoke test: `python scripts/smoke_test.py --backend lama` → writes `scratch/smoke_*.png`.

## Architecture

The pipeline is `detect → segment → refine mask → inpaint → composite`, implemented in the
`deink/` package:

- **`deink/segment.py`** — `TattooSegmenter`. `detect_and_segment(image, prompt="a tattoo.")`
  runs GroundingDINO (text-prompted detection) then SAM (segmentation), both via HF
  `transformers`. Detection boxes larger than `max_area_frac` (default 0.25) of the image are
  dropped — GroundingDINO also returns a subject-sized "tattoo" box for the whole person,
  which would make SAM mask the entire body. `segment_from_points` / `segment_from_boxes`
  back the interactive path. Model ids: `IDEA-Research/grounding-dino-base`,
  `facebook/sam-vit-huge`.
- **`deink/inpaint.py`** — `Inpainter`. Two backends: `"lama"` (simple-lama-inpainting,
  fast, strong skin-texture fill) and `"sdxl"` (`diffusers` `AutoPipelineForInpainting` with
  `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, runs at 1024px with model CPU offload,
  composited back at native size). Models load lazily and stay resident.
- **`deink/pipeline.py`** — `remove_tattoo(image, backend=..., mask=None, dilate=8,
  feather=5, ...) -> RemovalResult`. Pass a `mask` to skip detection (interactive path).
  `refine_mask` dilates (cover ink edges) and feathers (seamless blend). Returns `.image`,
  `.mask`, `.raw_mask`, `.found`. Pixels outside the mask are left bit-identical to the input.
- **`deink/utils.py`** — `get_device`, `ensure_pil`, `mask_to_pil`, `empty_mask`.

Heavy model objects live on `TattooSegmenter` / `Inpainter` instances — construct once and
pass into `remove_tattoo` to reuse across calls (the marimo app caches singletons with
`functools.lru_cache`).

## Conventions that matter

- **`deink` must be importable.** `app.py` (repo root) imports it directly; `scripts/*.py`
  prepend the repo root to `sys.path`. For editable installs, `pip install -e .` (see
  `pyproject.toml`).
- **Masks are white = remove.** `refine_mask` returns float [0,1]; LaMa gets a hard binary
  mask, SDXL gets the soft one, and the final composite uses the feathered mask.
- **GPU expectations.** transformers/diffusers models auto-place on CUDA via `get_device()`.
  SDXL uses `enable_model_cpu_offload()` to stay under 16 GB — do not `.to("cuda")` the whole
  pipe as well.

## Data layout

Image data stays on disk, git-ignored: `data/deinked/{rawdata,tattoo,clean,gen,test}`,
`data/silhouette/*`, `data/stock/*`, `data/models/`. `data/deinked/test/` and
`data/stock/laser-removal/` are handy test inputs. Small history CSVs and sample outputs from
the old training era are committed as a record.

## Legacy

`legacy/` holds the retired fastai NoGAN notebooks (`Deink_00`–`Deink_05`) — kept because the
**silhouette mask-generation track and its curated data** are the one reusable asset (a future
custom tattoo-segmentation model could be fine-tuned from them). The dead generator/critic/GAN
notebooks (`Deink_06`–`Deink_09`) and scratch files were removed; they remain in git history
at commit `b313960`. See `legacy/README.md`.
