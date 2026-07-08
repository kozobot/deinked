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
`deink/` package. Localization has two interchangeable strategies, chosen by the `localizer`
knob on `remove_tattoo` (`"box"` default / `"seg"` / `"seg+box"`) — see the `localizer` bullet
below. Both produce a `bool (H,W)` mask that flows into the same refine→inpaint→composite tail.

- **`deink/segment.py`** — `TattooSegmenter`. `detect_and_segment(image, prompt="a tattoo.")`
  runs an open-vocab detector (text-prompted) then SAM (segmentation), both via HF
  `transformers`. Detection boxes larger than `max_area_frac` (default 0.25) of the image are
  dropped — the detector also returns a subject-sized "tattoo" box for the whole person,
  which would make SAM mask the entire body. `segment_from_points` / `segment_from_boxes`
  back the interactive path. Model ids: `IDEA-Research/grounding-dino-base`,
  `google/owlv2-base-patch16-ensemble`, `facebook/sam-vit-huge`.
  - **Pluggable detector (`detector=` knob):** `"gdino"` (GroundingDINO, default — no
    regression), `"owlv2"` (OWLv2), or `"ensemble"` (union of both, concatenated then
    NMS-deduped for max recall). Threaded through `remove_tattoo`, the app dropdown, and both
    scripts as a per-call override, so the cached app singleton switches detectors without
    rebuilding. `_detect_raw` is the seam: a dispatcher over `_detect_raw_gdino` /
    `_detect_raw_owlv2`; everything downstream only consumes `(boxes Nx4 xyxy, scores N)`.
  - **Why only OWLv2 (not YOLO-World / GroundingDINO 1.5+):** the pipeline stays
    transformers-native (no CUDA-ext build, no extra pip deps) — that rules out YOLO-World
    (`ultralytics`) and the gated GroundingDINO 1.5/1.6 API models. OWLv2 is registered under
    the same `AutoModelForZeroShotObjectDetection` / `AutoProcessor` classes.
  - **OWLv2 API differences from GroundingDINO:** takes a *list* of query phrases
    (`text=[["a tattoo"]]`, no trailing period — the `.`-separated prompt is split via
    `_owlv2_queries`), has a *single* confidence threshold (so `text_threshold` is ignored on
    the OWLv2 path; `box_threshold` → OWLv2's `threshold`), and its
    `post_process_grounded_object_detection` takes **no `input_ids`**.
- **`deink/tattooseg.py`** — `TattooMaskSegmenter`. The **custom pixel-level localizer**: a
  fine-tuned SegFormer (`transformers` `SegformerForSemanticSegmentation`) that emits the tattoo
  mask *directly*, bypassing box detection + SAM. This is what masks heavily-inked skin the box
  detector collapses on (on a near-fully-tattooed subject GroundingDINO's top box is the whole
  person, dropped by `max_area_frac`). `segment(image) -> bool (H,W)` upsamples logits to native
  size via `post_process_semantic_segmentation` and thresholds the tattoo-class probability
  (`threshold` is a recall knob). Loads lazily from `data/models/tattoo-segformer/` (override via
  `DEINK_TATTOOSEG_DIR`); `available()` probes for a checkpoint so callers no-op gracefully until
  the model is trained.
  - **`localizer=` knob (not a 4th `detector`):** `detector=` selects a *box* family and only
    applies to the box path — a semantic segmenter emits a mask, not boxes, so it gets its own
    knob. `remove_tattoo(..., localizer=...)`: `"box"` (default, no regression) = detector+SAM;
    `"seg"` = SegFormer only; `"seg+box"` = union of both masks (trivial `|`, higher recall +
    false positives). `detect_and_segment` branches at the top (box path factored into
    `_segment_box`), preserving its "sole localization call" invariant; the app dropdown offers
    `seg`/`seg+box` only when `available()`, and `remove_tattoo` returns `found=False` with a
    "train the model" message if `seg` is requested with no checkpoint.
  - **Training/data scripts (no new deps — SegFormer ships in `transformers`, augments use
    `torchvision.transforms.v2`):** `scripts/derive_masks.py` differences aligned
    `rawdata/*_{tattoo,clean}` pairs (median-align → LAB ΔE → Otsu w/ absolute floor →
    morphology → ignore-band trimap) into real masks + a frozen `split.json`;
    `scripts/gen_synthetic_seg.py` composites `tattoo_clipart` onto `tattooless` skin (clipped to
    the `mask-predict` silhouette, faded/warped for realism) for unlimited perfect-label pairs;
    `scripts/train_tattooseg.py` fine-tunes `nvidia/mit-b2` with a curriculum (synthetic warmup →
    real+synthetic mix), CE(ignore_index=255)+Dice, bf16, `--smoke` for a quick loop check;
    `scripts/eval_seg.py` scores IoU/recall/precision (+ optional PSNR-to-`clean`) of seg vs. the
    box+SAM baseline on the held-out split.
- **`deink/inpaint.py`** — `Inpainter`. Two backends: `"lama"` (simple-lama-inpainting,
  fast, strong skin-texture fill) and `"sdxl"` (`diffusers` `AutoPipelineForInpainting` with
  `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, runs at 1024px with model CPU offload,
  composited back at native size). Models load lazily and stay resident.
- **`deink/pipeline.py`** — `remove_tattoo(image, backend=..., mask=None, dilate=8,
  feather=5, ...) -> RemovalResult`. Pass a `mask` to skip detection (interactive path).
  `refine_mask` dilates (cover ink edges) and feathers (seamless blend). Returns `.image`,
  `.mask`, `.raw_mask`, `.found`. Pixels outside the mask are left bit-identical to the input.
  - **Crop-to-region (`crop=` / `crop_pad=` knobs, default on):** each inpaint pass runs on a
    padded window cropped around the mask at the backend's *native* resolution (`_region_bbox`
    + `_NATIVE_RES`) instead of downscaling the whole frame. Previously a small tattoo on a
    large photo was squashed into SDXL's 1024px before it was ever seen; now SDXL gets a
    1024²-of-real-pixels window centered on the ink (LaMa gets a tight, focused crop). The box
    is padded by `crop_pad` (fraction of mask extent per side, 0.5 → ~2x context), grown to a
    centered square when feasible (SDXL runs square — avoids aspect distortion) with a
    `min_size` floor of the native res, and always fully contains the mask and stays in-bounds.
    `_inpaint_region` crops → `_fill` (the shared backend+composite seam) → pastes back into a
    copy, so the frame stays bit-identical outside the mask. `crop=False` restores the old
    full-frame behaviour. Applies to `"lama"`, `"sdxl"`, and both legs of `"auto"`.
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

**Seg-training data provenance (do NOT need to rerun the legacy notebooks).** The custom
segmenter's training inputs already live on disk; the legacy `Deink_01`–`03` notebooks that
originally produced them do not need to be re-run:

- `data/silhouette/{source,mask,tattooless}` and `data/silhouette/tattoo_clipart` are
  **hand-curated raw assets** — `Deink_01` merely resizes/renames `data/silhouette/rawdata/*_{source,mask,tattooless}.*`
  into them (no model involved). If ever regenerated, it's a trivial resize.
- `data/silhouette/mask-predict/` (body silhouettes) is the **one model-derived artifact** — the
  output of the retired silhouette U-Net (`Deink_02`/`03`). `gen_synthetic_seg.py` treats it as
  **optional**: a missing silhouette falls back to "whole image is body" (`--silhouette` to
  point elsewhere). It only sharpens synthetic realism (keeps composited tattoos on skin), so
  the retired U-Net is not on the critical path.
- The *essential* input — the real `data/deinked/rawdata/*_{tattoo,clean}` pairs that
  `derive_masks.py` differences — is independent of the silhouette track entirely.

## Legacy

`legacy/` holds the retired fastai NoGAN notebooks (`Deink_00`–`Deink_05`) — kept because the
**silhouette mask-generation track and its curated data** are the one reusable asset (a future
custom tattoo-segmentation model could be fine-tuned from them). The dead generator/critic/GAN
notebooks (`Deink_06`–`Deink_09`) and scratch files were removed; they remain in git history
at commit `b313960`. See `legacy/README.md`.
