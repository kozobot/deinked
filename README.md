# deinked

Automatic tattoo removal from images (video is a future goal).

**This project pivoted.** It began as a fastai v2 port of
[SkinDeep](https://github.com/vijishmadhavan/SkinDeep) that *trained* a GAN to remove
tattoos. That approach never fully worked — the adversarial stage never converged and the
MSE-only generator only faded tattoos rather than reconstructing skin. Modern foundation
models make training a bespoke model unnecessary, so deinked is now a **segment-and-inpaint
pipeline** built on pretrained models:

```
image → detect the tattoo (GroundingDINO) → segment it (SAM) → grow/feather the mask
      → inpaint (LaMa or SDXL) → composite back at full resolution
```

The old training notebooks are archived under [`legacy/`](legacy/README.md).

## Setup

Everything runs locally on the GPU via a conda environment. Target hardware is an NVIDIA
Blackwell card (e.g. RTX 5070 Ti, 16 GB); any recent CUDA GPU with a driver new enough for
your PyTorch build works.

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate deinked
```

PyTorch is pulled from a CUDA wheel that ships `sm_120` (Blackwell) kernels. Verify the GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## Usage

### App (marimo)

```bash
marimo run app.py        # interactive: upload → remove → before/after
marimo edit app.py       # open as an editable reactive notebook
```

Upload a photo; the tool auto-detects and removes tattoos. If auto-detect misses, upload a
black/white mask (white = the area to remove). Toggle the inpaint backend (LaMa vs SDXL) and
tune mask grow/feather.

### Library / script

```python
from PIL import Image
from deink import remove_tattoo

img = Image.open("photo.jpg")
result = remove_tattoo(img, backend="lama")   # or backend="sdxl"
result.image.save("clean.jpg")
```

Quick end-to-end check on a sample image:

```bash
python scripts/smoke_test.py --backend lama
# writes scratch/smoke_lama.png (before | after) and scratch/smoke_lama_mask.png
```

## How it works

- **`deink/segment.py`** — `TattooSegmenter`: text-prompted detection (GroundingDINO) +
  SAM segmentation, via Hugging Face `transformers` (no CUDA-extension build). Subject-sized
  detection boxes are dropped so SAM masks the individual tattoos, not the whole person.
  `segment_from_points` backs the interactive fallback.
- **`deink/inpaint.py`** — `Inpainter`: **LaMa** (fast, excellent skin-texture fill) and
  **SDXL inpainting** (semantic fill for harder regions). Models load lazily; SDXL uses
  model CPU offload to fit in 16 GB.
- **`deink/pipeline.py`** — `remove_tattoo`: orchestrates detect → segment → refine mask
  (dilate + feather) → inpaint → feathered composite at full resolution. Pixels outside the
  mask stay bit-identical to the input.

## Backends

| Backend | Speed | Best for |
|---------|-------|----------|
| `lama`  | fast (~seconds) | most tattoos on skin; strong default |
| `sdxl`  | slower (diffusion) | large/complex regions needing semantic fill |

## Roadmap

- **Recall:** GroundingDINO's "tattoo" prompt misses faint marks. A custom tattoo
  segmentation model, fine-tuned from the `legacy/` silhouette masks, would improve auto-detect.
- **Video:** swap SAM for SAM 2 (native mask propagation across frames) and add temporal
  consistency to the inpaint step.
