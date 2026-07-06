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
# tune detection: --detector gdino|owlv2|ensemble / --box-threshold / --text-threshold / --max-area-frac / --prompt / --tile
```

If auto-detect misses a tattoo, sweep detection settings on that image to find ones that catch
it (detection only, no inpaint — fast):

```bash
python scripts/sweep_detect.py path/to/image.jpg --out scratch/
# prints n_boxes + area fractions per prompt/threshold combo; --out writes box-overlay PNGs
```

## How it works

- **`deink/segment.py`** — `TattooSegmenter`: text-prompted detection + SAM segmentation, via
  Hugging Face `transformers` (no CUDA-extension build). The open-vocab detector is pluggable
  (`detector=`): GroundingDINO (default), OWLv2, or an `ensemble` union of both. Subject-sized
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

- **Prompt / threshold tuning.** ✅ *Exposed.* `box_threshold` / `text_threshold` /
  `max_area_frac` and the detection `prompt` are all parameters of
  `deink/pipeline.remove_tattoo`, sliders under **Advanced detection** in the app, and flags on
  `scripts/smoke_test.py` (`--box-threshold` / `--text-threshold` / `--max-area-frac`). Lower
  thresholds recover fainter tattoos; try compound prompts (`"tattoo. ink drawing on skin."`).
  `scripts/sweep_detect.py IMAGE` sweeps prompt/threshold/area combinations on a single image
  (detection only, fast) and reports box counts + area fractions to pick per-image settings.
  Cheap, partial gains. Caveat below still holds: raising `box_threshold` to cut false positives
  costs more recall than it saves — this knob is for *lowering* thresholds on hard images.
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
- **Swap / upgrade the open-vocab detector.** ✅ *Implemented (OWLv2 + ensemble).* The detector is
  now pluggable via a single `detector` knob — `remove_tattoo(img, detector="owlv2")` (or
  `"ensemble"`), the **Detector** dropdown in the app, and `--detector` / `--detectors` on
  `scripts/smoke_test.py` / `scripts/sweep_detect.py`. Default stays `"gdino"` (GroundingDINO), so
  nothing regresses; the alternatives are opt-in:
  - **OWLv2** (Google, `google/owlv2-base-patch16-ensemble`) — open-vocab detector that tends to
    catch small objects GroundingDINO misses. Single confidence threshold, so `text_threshold` is
    ignored on this path; takes a *list* of query phrases (the `.`-separated prompt is split, e.g.
    `"tattoo. ink drawing on skin."` → two queries). `google/owlv2-large-patch14-ensemble` is the
    higher-recall/heavier option if you have VRAM headroom.
  - **`ensemble`** — union of GroundingDINO + OWLv2 boxes, NMS-merged. Highest recall, ~2× detection
    time. Both feed the tiled and non-tiled paths, so it composes with `tile=True`.
  - **GroundingDINO 1.5 / 1.6** (IDEA) — stronger, but API-gated / not a clean HF `transformers`
    drop-in, so not wired up (the pipeline stays transformers-native, no CUDA-ext build).
  - **YOLO-World** — open-vocab, fast, *fine-tunable*, but needs `ultralytics` (extra dep) — deferred
    for the same transformers-native reason; it bridges to the custom-model bullet below.
  - **Florence-2** (Microsoft, ~0.8B) — unified vision model; its dense-region / referring-
    segmentation modes could propose tattoo regions better than a single `"a tattoo."` prompt.
  Use `scripts/sweep_detect.py IMAGE --detectors gdino owlv2 ensemble` to A/B them on one image.
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

**Small tattoos removed cleanly; sleeves and large pieces still blur or leave artifacts.** This is
the known failure mode, and it is a *fill* problem, not a detection one — a large mask is a large
hole, and the fill model has to invent plausible skin/limb across it. LaMa (feed-forward, FFC-based)
is excellent at *extending nearby texture* into a small hole but has no semantics, so on a
limb-sized hole it collapses to a smear; diffusion can reconstruct structure but degrades when the
hole is large relative to its working resolution. The suggestions below are ordered by
value-to-effort for exactly this large-region case.

- **Backend auto-selection (highest value, cheap).** Route per mask instead of a global toggle:
  small masks → LaMa (fast, plain skin), large or limb-spanning masks → SDXL. Confirmed
  necessary on `retouchme-86` (full-sleeve tattoos): **LaMa dissolves the whole forearm into the
  background** on a limb-sized hole because it has no semantics, whereas **SDXL reconstructs a
  coherent arm/garment** (~11 s). Detection there was already correct (98% recall) — this is
  purely a fill-quality problem, so size-based routing fixes it without new dependencies. **Route
  per connected mask component, not per image:** label the mask (`cv2.connectedComponents` / SAM
  already returns per-box masks), send each small blob to LaMa and each large blob to SDXL, so an
  image with both a wrist tattoo and a sleeve gets the right model for each instead of one global
  choice.
- **Crop-to-region at native model resolution (biggest lever for large tattoos).** The single most
  effective fix short of a new model. Today SDXL runs the *whole image* at 1024 px, so a sleeve is
  reconstructed from relatively few pixels and blurs. Instead, crop a tight bbox around each mask
  component **+ a skin margin** (e.g. 25–50%), inpaint that crop at the model's native resolution
  (1024 px of *just the arm*), and composite back. This multiplies the effective resolution on the
  inked area and is where most sleeve blur comes from. LaMa benefits too — it was trained at a
  limited resolution, so feeding it a downscaled full frame smears big holes; crop-and-fill (or
  downscale → inpaint → upscale the fill) keeps it near its training scale.
- **Progressive / "onion-peel" filling for big holes.** For any feed-forward model, iteratively
  shrink the hole from its boundary inward (fill a ring, add it to the known pixels, repeat) so
  structure and texture propagate instead of the center collapsing to an average. Cheap to
  implement on top of LaMa and directly attacks the center-of-a-sleeve smear.
- **Two-stage structure → texture.** Let one pass rough in low-frequency structure (LaMa, or a
  large-hole specialist) and a second diffusion pass (SDXL/FLUX with `strength` ~0.4–0.6 over the
  first result) add skin texture. Keeps limbs anatomically coherent while avoiding the plastic look
  of a single high-strength diffusion pass.
- **Guide the diffusion fill.** (a) **Prompt/negative-prompt** in `deink/inpaint.py`: describe the
  target ("bare skin, natural skin texture, even lighting, muscle") and negative-prompt the source
  ("tattoo, ink, text, lettering") so the model doesn't re-hallucinate ink — a common large-tattoo
  artifact. (b) **ControlNet (depth/pose)** or an **IP-Adapter** skin reference to hold limb shape
  and skin tone across the hole.
- **Tune SDXL.** Now validated end-to-end; tune `strength`, `guidance_scale`, steps, and the skin
  prompt in `deink/inpaint.py`. LaMa wins on speed and plain skin; SDXL wins on large/curved
  regions and areas crossing anatomical boundaries. For removal, keep `strength` high enough to
  erase bold ink but not so high it invents unrelated detail.
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
  Remaining, and directly relevant to large-tattoo artifacts:
  - **Per-region adaptive dilation.** Bold sleeve linework needs *more* dilation than a thin script
    tattoo or its dark ink edges bleed through the fill as a ghost/halo; scale dilation to the mask
    component's size (a few px for tiny tattoos, more for a sleeve) instead of a single global
    `dilate`. Under-covered bold ink is a frequent "odd artifact" source on large pieces.
  - **Edge-aware feathering** so the blend follows the limb contour, not a uniform Gaussian ring.
  - **Seam / color harmonization.** Even a good fill can leave a visible patch on a large region:
    color-match the fill to surrounding skin (histogram/reinhard transfer) and blend with
    **Poisson (`cv2.seamlessClone`) or Laplacian-pyramid** compositing instead of a straight
    feathered paste, to kill the halo and lighting mismatch at the boundary.

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
