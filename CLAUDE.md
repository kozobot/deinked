# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`deinked` removes tattoos from images (video is a future goal). It **used to** train a
bespoke fastai GAN (a NoGAN port of [SkinDeep](https://github.com/vijishmadhavan/SkinDeep));
that never fully worked and has been retired. It is now a **segment-and-inpaint pipeline**
built on pretrained foundation models — no training. The core is a small importable Python
package (`deink/`) plus a marimo app, driven from a local conda environment.

## Repository layout (root = ComfyUI plugin, `core/` = the project)

The **repo root is a ComfyUI custom-node plugin** (installable directly from the GitHub URL /
Manager / Registry). The **original project lives under [`core/`](core/)** — the `deink` package,
the marimo app, the training scripts, and `data/`. The plugin adds `core/` to `sys.path` (in the
root `__init__.py`) and **imports the canonical `deink` code** — there is no vendored copy.

- Plugin root: `__init__.py` (`NODE_CLASS_MAPPINGS` + the `sys.path`/`DEINK_TATTOOSEG_DIR` shim),
  `convert.py` (IMAGE/MASK tensor ↔ PIL/numpy), `models.py` (cached model singletons),
  `nodes/*.py` (the node classes), `pyproject.toml` (`[tool.comfy]`), `example_workflows/`.
- Nodes: `DeinkSegFormer`, `DeinkRefineMask`, `DeinkSplitMaskBySize`, `DeinkLamaBackend`,
  `DeinkSdxlBackend`, `DeinkFluxBackend`, `DeinkTwoStageBackend`, `DeinkMiganBackend`,
  `DeinkMatBackend`, `DeinkInpaint`, `DeinkRemoveTattoo`. Box localization
  reuses the commodity
  `comfyui_segment_anything` node (interop via the `MASK` type, not imported); we own `DeinkInpaint`
  so crop-to-native + bit-identical composite wrap the backend. `DeinkInpaint` treats its input mask
  as *final* (refine upstream with `DeinkRefineMask`; it does not re-refine).
  - **Backend as a wired input (`nodes/backend.py`):** `DeinkInpaint` has **no `backend` combo**.
    Instead `DeinkLamaBackend` / `DeinkSdxlBackend` / `DeinkFluxBackend` / `DeinkTwoStageBackend`
    / `DeinkMiganBackend` / `DeinkMatBackend` are provider nodes that emit a custom `DEINK_BACKEND`
    descriptor (`{"name", "min_area_frac", "kwargs"}`) into `DeinkInpaint`'s optional `backend_1..3`
    sockets; SDXL params (prompt/strength/steps/seed/…) live on the SDXL / two-stage providers (the
    two-stage provider defaults `strength` to 0.5 — its params configure the SDXL texture stage);
    the FLUX provider is the same minus `negative_prompt` (FLUX is guidance-distilled) with
    `guidance_scale` defaulting to ~30. The **MI-GAN / MAT providers are feed-forward** — MI-GAN has
    no params, MAT exposes only an optional `seed` (redraws its latent). `DeinkInpaint` auto-routes
    each connected mask component to the wired backend whose `min_area_frac` best matches the
    component's image-area fraction, via `deink.pipeline._route_by_component_size` (the N-tier
    generalization of `_split_by_component_size`); with the default pair (lama 0.0, sdxl 0.02) this
    reproduces the old plain-SDXL `backend="auto"` (which routes large regions to `twostage` — wire
    a `DeinkTwoStageBackend` in place of the SDXL provider to match current `auto`). No backend
    wired → a plain LaMa fill (standalone default). `DeinkRemoveTattoo` (all-in-one) keeps its own
    `backend` = `lama`/`sdxl`/`flux`/`auto`/`twostage`/`migan`/`mat` string knob.
- **Training is out of scope for ComfyUI** and stays as offline scripts in `core/scripts/`; the
  plugin only *consumes* the SegFormer checkpoint at `core/data/models/tattoo-segformer/`.

## Running

Everything below runs from **`core/`** (the project working dir), in a local conda env (no Docker),
targeting an NVIDIA Blackwell GPU (sm_120, e.g. RTX 5070 Ti):

```bash
cd core && mamba env create -f environment.yml && conda activate deinked
```

PyTorch comes from a CUDA wheel that includes `sm_120` kernels. There is no CPU-only path
worth using — the diffusion/segmentation models assume a GPU.

- App: `marimo run app.py` (or `marimo edit app.py`) from `core/`.
- Smoke test: `python scripts/smoke_test.py --backend lama` (from `core/`) → writes `scratch/smoke_*.png`.
- The ComfyUI plugin: clone the repo root into `ComfyUI/custom_nodes/`; see the root `README.md`.

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
- **`deink/inpaint.py`** — `Inpainter`. Backends: `"lama"` (simple-lama-inpainting,
  fast, strong skin-texture fill), `"sdxl"` (`diffusers` `AutoPipelineForInpainting` with
  `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, runs at 1024px with model CPU offload,
  composited back at native size), `"flux"` (`inpaint_flux`: `diffusers` `FluxFillPipeline` on
  `black-forest-labs/FLUX.1-Fill-dev` — SOTA fill, better structure/texture than SDXL; the 12B
  transformer is loaded **GGUF-quantized** via `FluxTransformer2DModel.from_single_file` +
  `GGUFQuantizationConfig` — `FLUX_GGUF_URL`, `DEINK_FLUX_GGUF`-overridable — so it fits 16 GB under
  CPU offload; base repo is **gated** (needs `hf auth login`); guidance-distilled so no negative
  prompt, `guidance_scale` ~30, also runs 1024px composited back at native size), and `"twostage"`
  (`inpaint_twostage`: LaMa structure pass then a low-strength SDXL texture pass —
  `DEFAULT_TWOSTAGE_STRENGTH = 0.5` — over the LaMa result; SDXL at `strength<1` denoises *from* the
  LaMa fill, refining not regenerating), plus two **feed-forward** GAN fills — `"migan"`
  (`inpaint_migan`) and `"mat"` (`inpaint_mat`). Both are fast, non-diffusion large-hole
  specialists **vendored pure-PyTorch under `deink/vendor/`** (no CUDA-ext build, no new pip deps;
  `iopaint`, which bundles them, pins `diffusers==0.27.2` and would conflict with the FLUX
  backend). Weights are **un-gated** GitHub-release assets that download lazily
  (`DEINK_MIGAN_URL` / `DEINK_MAT_URL`); MI-GAN is a TorchScript trace, MAT a fixed-512 StyleGAN
  state-dict. Like LaMa they fill a hard binary hole and are composited back by `_fill`
  (`_NATIVE_RES = 512`). Models load lazily and stay resident; two-stage adds no new model.
  `inpaint(..., backend=...)` dispatches all six.
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
    full-frame behaviour. Applies to `"lama"`, `"sdxl"`, `"twostage"`, and both legs of `"auto"`.
  - **Two-stage backend (`backend="twostage"`):** LaMa roughs in structure, then a low-strength
    SDXL pass adds skin texture over the LaMa result (see `deink/inpaint.py`); it flows through the
    same `_fill`/crop seam (`_NATIVE_RES["twostage"] = 1024`) with no `_fill` change. Extra
    `**inpaint_kwargs` (prompt/strength/…) reach the SDXL pass (or two-stage's SDXL stage).
  - **Auto large-component backend (`_AUTO_LARGE_BACKEND`, default `"twostage"`):** `backend="auto"`
    routes small components → `"lama"` and large/limb-spanning ones → this module constant in
    `deink/pipeline.py`. It defaults to **`"twostage"`** (domain-appropriate); MI-GAN/MAT ship
    **Places2 scene weights** (GPU-tested: they hallucinate scene fragments on big holes over people
    — see the migan-mat-places-domain memory), so they stay opt-in rather than the auto default.
    Flip the constant to `"mat"`/`"migan"` to trade quality for feed-forward speed. Because a
    feed-forward large backend takes no diffusion kwargs, `**inpaint_kwargs` are only forwarded to
    the large leg when it's a diffusion backend. The feed-forward set is `_FEEDFORWARD =
    ("lama", "migan", "mat")` — `_fill` gives all of them the hard-mask + feathered composite path.
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
