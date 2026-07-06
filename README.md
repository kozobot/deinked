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

## Roadmap / follow-ups

Ordered roughly by value-to-effort. Nothing here is required for the tool to work today —
these are the next improvements.

### 1. Detection recall — the biggest quality lever

GroundingDINO's `"a tattoo."` prompt reliably finds bold pieces but misses faint/small marks
(thin script, tiny wrist/ankle tattoos, low-contrast grey work). Because the pipeline can only
remove what it masks, recall is currently the ceiling on quality. Options, cheapest first:

- **Prompt / threshold tuning.** Expose and sweep `box_threshold` / `text_threshold` /
  `max_area_frac` per image; try compound prompts (`"tattoo. ink drawing on skin."`). Cheap,
  partial gains. Already parameterized in `deink/pipeline.remove_tattoo`.
- **Tile-and-detect.** Run detection on overlapping crops and merge boxes, so small tattoos
  occupy more of the detector's field of view. Moderate effort, good recall gain, slower.
- **Custom tattoo segmentation model (highest value).** Fine-tune a segmenter specifically for
  tattoos and drop GroundingDINO from the auto path entirely. The `legacy/` silhouette track is
  exactly the reusable asset for this — it already learned tattoo masks and has curated
  `source`/`mask` pairs plus a synthetic-data generator (clipart composited onto skin). Retarget
  that data to fine-tune e.g. a SAM decoder, a U-Net, or a YOLO-seg head. This is the single
  change most likely to make auto-removal "just work." See `legacy/README.md`.
- **Human-in-the-loop mask editing.** Even with a great detector, a shippable tool wants a
  brush/eraser to correct masks. Today the app supports uploading a full mask; an in-app
  SAM-click ("click a tattoo, SAM grows the region") via `segment_from_points` is the natural
  upgrade (see §5).

### 2. Inpainting quality

- **Validate and tune SDXL.** Only the LaMa backend has been exercised end-to-end; smoke-test
  `--backend sdxl`, then tune `strength`, `guidance_scale`, steps, and the skin prompt in
  `deink/inpaint.py`. LaMa wins on speed and plain skin; SDXL should win on large/curved regions
  and areas crossing anatomical boundaries.
- **Better skin models.** Evaluate FLUX Fill (needs a quantized/GGUF build to fit 16 GB) and
  dedicated skin-retouch or object-removal models. Consider a two-pass approach: LaMa for
  structure, a light diffusion pass for texture realism.
- **Backend auto-selection.** Pick LaMa vs SDXL per region based on mask size / location
  (e.g. large or joint-crossing masks → SDXL) instead of a global toggle.
- **Mask refinement.** Per-region adaptive dilation, edge-aware feathering, and color-matching
  the fill to surrounding skin tone to kill seams and lighting mismatches.

### 3. Video (stated long-term goal)

The pipeline is architected for this but none of it is built yet. Plan:

- **Swap SAM → SAM 2** (`facebook/sam2-*`). SAM 2 natively propagates masks across frames from a
  single prompt, which solves per-frame re-detection and keeps masks temporally stable. This is
  the main reason SAM (not just GroundingDINO) is in the design.
- **Track, don't re-detect every frame.** Detect/seed tattoos on keyframes, propagate with SAM 2,
  re-seed only on scene cuts or when tracking confidence drops.
- **Temporal consistency in the fill.** Per-frame inpainting flickers. Mitigations, in order of
  effort: optical-flow-warp the previous frame's fill into the current mask as initialization;
  use a video-native inpainting model (e.g. a ProPainter-style flow-guided approach); or enforce
  temporal loss/latent consistency for diffusion backends.
- **I/O and scale.** Decode/encode with `ffmpeg` (already in the old container notes), process in
  chunks, and expose a frame-range selector. Runtime will be the constraint — LaMa-per-frame is
  feasible on the 5070 Ti; diffusion-per-frame is not real-time and will need batching/overnight
  runs or a smaller/faster model.
- **New module.** Add `deink/video.py` (`remove_tattoo_video(path, ...)`) reusing the image
  pipeline per frame, plus a mask-propagation helper. Keep the image path unchanged.

### 4. Robustness & correctness

- **Multi-person images.** Detection/segmentation currently treats all boxes uniformly; verify
  behavior with multiple people and overlapping skin.
- **Orientation & color profiles.** Honor EXIF orientation on load and preserve ICC profiles so
  results don't rotate or shift color vs. the input.
- **Large images / memory.** Very high-resolution inputs may exhaust VRAM in SAM or SDXL; add
  tiled processing or a max-working-resolution with full-res recomposite.
- **"Nothing found" UX.** When auto-detect returns an empty mask, guide the user to lower
  thresholds or draw a mask rather than silently returning the original.
- **Tests.** There is no test suite yet. Add fixtures (a few sample images + expected mask
  coverage) and a fast CPU-mockable path for `pipeline` logic (mask refine/composite) so
  regressions are caught without a GPU.

### 5. App & UX

- **In-app interactive masking.** Click-to-segment (SAM points via `segment_from_points`) and a
  brush/eraser to fix masks, instead of upload-a-mask-file. The biggest usability win.
- **Batch mode.** Point at a folder, process all images, write results + masks alongside.
- **Side-by-side + mask overlay, download button, and per-image parameter memory** in the marimo
  app.
- **Progress/timing feedback** for the slower SDXL path.

### 6. Packaging & deployment

- **`pip install -e .`** is wired via `pyproject.toml`; add a console entry point
  (`deink remove <img>`) for a proper CLI.
- **Model caching / offline.** Pre-fetch Hugging Face weights into a pinned cache so first run
  isn't a multi-GB download; document `HF_TOKEN` for rate limits.
- **Pin versions.** `environment.yml` is currently unpinned. Once stable, freeze exact versions
  (especially `torch`/CUDA and `transformers`, which had an API rename at v5) for reproducibility.
- **Optional hosted mode.** A cloud-API fallback (for machines without a Blackwell GPU) behind the
  same `remove_tattoo` interface.

### 7. Ethics & safety

Automated tattoo removal on photos of real people has obvious misuse potential (identity
alteration, non-consensual edits). Before any public/hosted deployment, decide on watermarking,
consent/usage terms, and NSFW handling, and document intended use.
