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
remove what it masks, recall is currently the ceiling on quality.

Evaluated against the paired `data/deinked/rawdata/retouchme-*_{tattoo,clean}` set (artist-
cleaned targets → derive a ground-truth mask from the tattoo/clean pixel diff), the auto path
works well on **discrete** tattoos but hits two ceilings on **heavy coverage**: (a) on a
near-fully-tattooed subject GroundingDINO's top box is the whole person (correctly dropped by
`max_area_frac`), so recall collapses; (b) a box detector + SAM can't cleanly separate ink from
skin when the tattoo *is* most of the visible skin. Both point at a pixel-level tattoo
segmenter. Options, cheapest first:

- **Prompt / threshold tuning.** Expose and sweep `box_threshold` / `text_threshold` /
  `max_area_frac` per image; try compound prompts (`"tattoo. ink drawing on skin."`). Cheap,
  partial gains. Already parameterized in `deink/pipeline.remove_tattoo`.
- **Tile-and-detect.** ✅ *Implemented.* Detection runs on overlapping crops (plus the full
  image); boxes are offset back to full-image coordinates and NMS-merged. Because tiling only
  needs to recover tattoos too small to catch at full scale, tile-derived boxes are capped hard
  (`tile_max_area_frac`, default 0.03 of the image): on a zoomed crop GroundingDINO returns a
  loose box that sprawls onto the limb, and SAM would then segment the whole limb — the strict
  cap keeps only genuinely small tattoos and drops those limb-sized boxes. The full-image pass
  keeps the normal `max_area_frac` cap. Enable with `remove_tattoo(..., tile=True)` (also
  `--tile` on `scripts/smoke_test.py` and the "Tile detect" checkbox in the app); tune
  `tile_max_area_frac` / `tiles` / `overlap`. See `TattooSegmenter.detect_boxes_tiled` in
  `deink/segment.py`.
- **Swap / upgrade the open-vocab detector.** GroundingDINO-base is the weakest link. Low-effort
  drop-in alternatives (all via HF `transformers`, all fit 16 GB) to A/B against it:
  - **GroundingDINO 1.5 / 1.6** (IDEA) — same API family, stronger detection.
  - **OWLv2** (Google) — open-vocab detector that tends to catch small objects GroundingDINO misses.
  - **YOLO-World** — open-vocab, fast, and *fine-tunable* (bridges to the custom-model bullet).
  - **Florence-2** (Microsoft, ~0.8B) — unified vision model; its dense-region / referring-
    segmentation modes can propose tattoo regions better than a single `"a tattoo."` prompt.
  Note: raising `box_threshold` to cut false positives was measured to cost far more recall than
  it saves (faint tattoos and false positives share the same low-score band) — prefer a better
  detector over threshold tuning.
- **Custom tattoo segmentation model (highest value).** Fine-tune a *pixel-level* segmenter
  specifically for tattoos and drop GroundingDINO + the box→SAM path from the auto flow entirely.
  This is the only thing that fixes the heavy-coverage ceiling above: it masks exactly the inked
  skin even at high body coverage, instead of collapsing to a whole-person box. The `legacy/`
  silhouette track is exactly the reusable asset — it already learned tattoo masks and has curated
  `source`/`mask` pairs plus a synthetic-data generator (clipart composited onto skin). Retarget
  that data to fine-tune e.g. a SAM/SAM 2 decoder, a Mask2Former/SegFormer/U-Net semantic head, or
  a YOLO-seg head. The single change most likely to make auto-removal "just work." See
  `legacy/README.md`.
- **Human-in-the-loop mask editing.** Even with a great detector, a shippable tool wants a
  brush/eraser to correct masks. Today the app supports uploading a full mask; an in-app
  SAM-click ("click a tattoo, SAM grows the region") via `segment_from_points` is the natural
  upgrade (see §5).

### 2. Inpainting quality

- **Backend auto-selection (highest value, cheap).** Route per mask instead of a global toggle:
  small masks → LaMa (fast, plain skin), large or limb-spanning masks → SDXL. Confirmed
  necessary on `retouchme-86` (full-sleeve tattoos): **LaMa dissolves the whole forearm into the
  background** on a limb-sized hole because it has no semantics, whereas **SDXL reconstructs a
  coherent arm/garment** (~11 s). Detection there was already correct (98% recall) — this is
  purely a fill-quality problem, so size-based routing fixes it without new dependencies.
- **Tune SDXL.** Now validated end-to-end; tune `strength`, `guidance_scale`, steps, and the skin
  prompt in `deink/inpaint.py`. LaMa wins on speed and plain skin; SDXL wins on large/curved
  regions and areas crossing anatomical boundaries.
- **Better fill models.** In rough order of quality/effort:
  - **FLUX.1 Fill [dev]** (Black Forest Labs) — current SOTA inpainting, better structure/texture
    than SDXL; needs a quantized (GGUF / nf4) build to fit 16 GB.
  - **PowerPaint** / **BrushNet** — diffusion inpainters with an explicit *object-removal* mode,
    designed for "remove this, fill plausibly."
  - **MAT** / **MI-GAN** / **ZITS** — large-hole specialists, better than LaMa on big masks if you
    want to stay feed-forward/fast.
  - **ControlNet (pose/depth) guidance** or a **two-pass** LaMa-structure → diffusion-texture combo
    to keep reconstructed limbs anatomically correct.
- **Mask refinement.** The default dilation was reduced 15 → 8 px (measured on the paired retouchme
  data: ~38% less clean-skin over-paint — the main "blur" source — at negligible recall cost).
  Remaining: per-region adaptive dilation, edge-aware feathering, and color-matching the fill to
  surrounding skin tone to kill seams and lighting mismatches.

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
