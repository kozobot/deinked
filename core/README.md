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

# backend="auto" routes each mask region by size: small blobs → LaMa, large → SDXL
result = remove_tattoo(img, backend="auto")

# Use the custom pixel-level tattoo segmenter instead of box-detection + SAM (see below):
result = remove_tattoo(img, localizer="seg")      # SegFormer only
result = remove_tattoo(img, localizer="seg+box")  # union of seg + box path (max recall)
```

Quick end-to-end check on a sample image:

```bash
python scripts/smoke_test.py --backend lama
# writes scratch/smoke_lama.png (before | after) and scratch/smoke_lama_mask.png
# tune detection: --detector gdino|owlv2|ensemble / --box-threshold / --text-threshold / --max-area-frac / --prompt / --tile
# pick the localizer: --localizer box|seg|seg+box   (seg / seg+box need a trained checkpoint — see below)
```

If auto-detect misses a tattoo, sweep detection settings on that image to find ones that catch
it (detection only, no inpaint — fast):

```bash
python scripts/sweep_detect.py path/to/image.jpg --out scratch/
# prints n_boxes + area fractions per prompt/threshold combo; --out writes box-overlay PNGs
```

### Custom tattoo segmentation model (`localizer="seg"`)

The box detector + SAM path misses on heavily-tattooed subjects (its top box becomes the whole
person, which gets dropped, so recall collapses). The fix is a **pixel-level** segmenter — a
fine-tuned [SegFormer](https://huggingface.co/nvidia/mit-b2) that masks the inked skin directly.
It stays transformers-native (no CUDA-ext build, no new deps). Once trained, select it with
`localizer="seg"` (or `"seg+box"` to union it with the box path); the app dropdown exposes both
automatically when a checkpoint is present. `"box"` remains the default.

**Training is a one-time GPU run.** Build the data, train, then evaluate against the baseline:

```bash
# 1. Data prep (uses the paired data under data/deinked + data/silhouette, git-ignored on disk).
python scripts/derive_masks.py          # real masks from tattoo/clean diffs → data/silhouette/derived/
                                        #   + a frozen train/eval split (split.json); QC sheet in scratch/
python scripts/gen_synthetic_seg.py --n 400   # composite clipart onto skin → data/silhouette/synthetic_gen/

# 2. Fine-tune SegFormer → data/models/tattoo-segformer/  (add --smoke for a fast loop check first)
python scripts/train_tattooseg.py

# 3. Compare the seg model against the box+SAM baseline on the held-out split
python scripts/eval_seg.py --methods gdino ensemble+tile seg seg+box --downstream
#   reports foreground IoU / recall / precision (+ PSNR-to-clean); the bar is: seg beats the
#   box baselines on recall without tanking precision.
```

Notes:
- The checkpoint lives at `data/models/tattoo-segformer/`; point elsewhere with the
  `DEINK_TATTOOSEG_DIR` env var. Until a checkpoint exists, `localizer="seg"` no-ops gracefully
  (returns "not found" with a message) and the app hides the seg options.
- `train_tattooseg.py` uses a curriculum (synthetic warm-up → real+synthetic fine-tune); tune
  `--encoder nvidia/mit-b1` (lighter, if it overfits), `--epochs-pretrain` / `--epochs-finetune`,
  `--fg-weight`, `--real-frac`. Trains at 512 px in bf16.
- **GPU memory.** Defaults (batch 4 + `--grad-accum 2` = effective batch 8, gradient checkpointing
  on) peak at **~2.6 GiB** — safe on a busy/shared 16 GB card. If you still OOM, drop `--batch 2`
  (raise `--grad-accum 4` to keep the effective batch) or `--size 384`; if you have the card to
  yourself, `--batch 8 --grad-accum 1 --no-grad-checkpoint` is faster. `eval_seg.py --downstream`
  loads several models at once (SAM + detectors + inpainter + seg) — run it without `--downstream`,
  or on a freer GPU, if it OOMs. The script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  to limit fragmentation.
- **Data provenance — you do *not* need to re-run the legacy notebooks.** `gen_synthetic_seg.py`
  reads `data/silhouette/{tattooless,tattoo_clipart}` (hand-curated raw assets — `Deink_01` only
  resized them from `data/silhouette/rawdata/`, no model) and `data/silhouette/mask-predict/`
  (body silhouettes from the retired U-Net). Those files already exist on disk; and the
  silhouette is **optional** — a missing one falls back to "whole image is body," so the retired
  U-Net is not required. The essential real signal comes from the `data/deinked/rawdata` pairs
  via `derive_masks.py`, which is independent of the silhouette track. See `CLAUDE.md` → *Data
  layout* for the full breakdown.
- See Roadmap §1 and `CLAUDE.md` for the data sources and design.

## How it works

- **`deink/segment.py`** — `TattooSegmenter`: text-prompted detection + SAM segmentation, via
  Hugging Face `transformers` (no CUDA-extension build). The open-vocab detector is pluggable
  (`detector=`): GroundingDINO (default), OWLv2, or an `ensemble` union of both. Subject-sized
  detection boxes are dropped so SAM masks the individual tattoos, not the whole person.
  `segment_from_points` backs the interactive fallback.
- **`deink/tattooseg.py`** — `TattooMaskSegmenter`: the custom pixel-level localizer, a
  fine-tuned SegFormer that emits the tattoo mask directly (no boxes, no SAM) so it masks
  heavily-inked skin the box detector collapses on. Selected with `localizer="seg"` /
  `"seg+box"`; loads a checkpoint trained by `scripts/train_tattooseg.py` (see Roadmap §1).
- **`deink/inpaint.py`** — `Inpainter`: **LaMa** (fast, excellent skin-texture fill),
  **SDXL inpainting** (semantic fill for harder regions), **FLUX.1 Fill** (SOTA diffusion fill,
  GGUF-quantized, gated base repo), a **two-stage** fill (LaMa structure → low-strength SDXL
  texture) for large regions, and **MI-GAN** / **MAT** — fast **feed-forward** (non-diffusion)
  large-hole GANs, vendored pure-PyTorch under `deink/vendor/` (no CUDA-ext build, no new deps;
  un-gated weights auto-download). Models load lazily; SDXL and FLUX use model CPU offload to fit
  in 16 GB.
- **`deink/pipeline.py`** — `remove_tattoo`: orchestrates detect → segment → refine mask
  (dilate + feather) → inpaint → feathered composite at full resolution. Pixels outside the
  mask stay bit-identical to the input.

## Backends

| Backend | Speed | Best for |
|---------|-------|----------|
| `lama`  | fast (~seconds) | most tattoos on skin; strong default |
| `sdxl`  | slower (diffusion) | large/complex regions needing semantic fill |
| `flux`  | slowest (12B diffusion) | best structure/texture — FLUX.1 Fill, GGUF-quantized; gated base repo (`hf auth login`) |
| `twostage` | slowest (LaMa + diffusion) | large/limb-spanning holes — LaMa structure then a low-strength (0.5) SDXL texture pass, coherent limbs without the plastic look |
| `migan` | fast (feed-forward) | opt-in feed-forward GAN fill; **Places2 scene weights** — fast but can hallucinate on skin, so not the `auto` default |
| `mat`   | fast (feed-forward) | opt-in feed-forward StyleGAN fill; same Places2-domain caveat as `migan` |
| `auto`  | per-region | mixed images — routes small blobs → LaMa, large/limb-spanning → two-stage |

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
- **Custom tattoo segmentation model (highest value).** ✅ *Implemented (SegFormer).* A
  *pixel-level* tattoo segmenter, selectable via the `localizer` knob, that emits the mask
  directly and drops GroundingDINO + the box→SAM path from the auto flow — the one thing that
  fixes the heavy-coverage ceiling above: it masks exactly the inked skin even at high body
  coverage instead of collapsing to a whole-person box. `remove_tattoo(img, localizer="seg")`
  (or `"seg+box"` to union with the box path; `"box"` stays the default until eval proves seg —
  same no-regression discipline as gdino/owlv2). The model is a fine-tuned `nvidia/mit-b2`
  SegFormer (`deink/tattooseg.py` `TattooMaskSegmenter`), staying transformers-native (no
  CUDA-ext build, no new deps — augments use `torchvision.transforms.v2`). Training reuses the
  `legacy/` silhouette track as intended, across three data sources: `scripts/gen_synthetic_seg.py`
  composites clipart onto tattoo-free skin (perfect labels, unlimited), `scripts/derive_masks.py`
  differences the aligned `data/deinked/rawdata/*_{tattoo,clean}` pairs into real masks, and the
  30 curated `source`/`mask` pairs. `scripts/train_tattooseg.py` fine-tunes with a
  synthetic→real curriculum; `scripts/eval_seg.py` scores recall/IoU/precision vs. the box+SAM
  baseline on a held-out split. Train with `python scripts/train_tattooseg.py`, then the app
  dropdown exposes `seg`/`seg+box`. See `legacy/README.md`.
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

- **Backend auto-selection (highest value, cheap).** ✅ *Implemented.* `remove_tattoo(...,
  backend="auto")` routes per mask instead of a global toggle: small masks → LaMa (fast, plain
  skin), large or limb-spanning masks → two-stage (LaMa structure + low-strength SDXL texture; was
  plain SDXL before the two-stage item below). Confirmed necessary on `retouchme-86` (full-sleeve
  tattoos): **LaMa dissolves the whole forearm into the background** on a limb-sized hole because
  it has no semantics, whereas **SDXL reconstructs a coherent arm/garment** (~11 s). Detection
  there was already correct (98% recall) — this is purely a fill-quality problem, so size-based
  routing fixes it without new dependencies. **Routing is per connected mask component, not per
  image:** the refined mask is labelled with `cv2.connectedComponents`, each blob covering ≥
  `auto_area_frac` of the image (default 0.02) goes to SDXL and the rest to LaMa, so an image with
  both a wrist tattoo and a sleeve gets the right model for each instead of one global choice.
  Wired through `remove_tattoo` (`backend="auto"`, `auto_area_frac=`), the app's **Inpaint
  backend** dropdown, and `scripts/smoke_test.py` (`--backend auto` / `--auto-area-frac`). See
  `_split_by_component_size` in `deink/pipeline.py`.
- **Crop-to-region at native model resolution (biggest lever for large tattoos).** ✅
  *Implemented (default on).* Each inpaint pass runs on a padded window cropped around the mask at
  the backend's native resolution instead of downscaling the whole frame. `_region_bbox` pads the
  mask bbox by `crop_pad` (default 0.5 → ~2× context) and grows it to a centered square (SDXL runs
  square — avoids aspect distortion) with a `min_size` floor of the native res (1024 for SDXL), so
  a small tattoo on a large photo gets a **1024²-of-real-pixels** window instead of being squashed
  into 1024 px before SDXL ever sees it (a sleeve likewise keeps its detail). `_inpaint_region`
  crops → fills → pastes back into a copy, leaving pixels outside the mask bit-identical. Knobs
  `crop=` / `crop_pad=` on `remove_tattoo` (set `crop=False` to restore the old full-frame path);
  applies to `"lama"`, `"sdxl"`, and both legs of `"auto"`. Validated on `vic_lari` and the
  laser-removal set: **cropped SDXL is ~40 % faster than full-frame** (~10 s vs ~17 s) at equal-or-
  better quality, and it cleanly erases the 130–150 px-deep holes (cherry-blossom sleeve, tribal
  dragon) that motivated this item. See `_region_bbox` / `_NATIVE_RES` in `deink/pipeline.py`.
- **Progressive / "onion-peel" filling for big holes.** ❌ *Won't do — built it, benchmarked it,
  it doesn't help.* Two findings killed it. (a) The literal outer-ring-inward peel is **wrong for a
  removal task**: leaving the inner hole unmasked treats the still-present ink as valid surrounding
  context, so the early passes *drag the ink inward* into a smeared blob — worse than a single
  pass. (A corrected variant — mask the *entire* not-yet-committed hole each pass so ink is never
  context, commit only the current outer ring, then re-fill the interior against the freshly
  committed skin — removes the dragging.) (b) Even corrected, onion-peel **never beat single-pass**
  on the three deepest holes in the test set (130–150 px), for either backend: SDXL onion
  *hallucinates spurious specks / a ghost of the original* (clearly worse), and LaMa onion only
  marginally softens its smear on a backend that loses to SDXL anyway — which `backend="auto"`
  already routes large holes to. Root cause: **crop-to-region (above) already gives the fill model
  a clean native-resolution view**, so the "center of a big hole collapses to an average" premise
  this item targeted no longer occurs. At ~`onion_layers`× the inpaint calls for no gain, not worth
  shipping.
- **Two-stage structure → texture.** ✅ *Implemented (`backend="twostage"`).* One pass roughs in
  low-frequency structure (LaMa) and a second, low-strength SDXL pass (`strength` default 0.5, in
  the 0.4–0.6 band) runs over that result to add skin texture. Because SDXL is wired as an inpaint
  pipeline, running it at `strength<1` over the LaMa fill **seeds its denoising from that fill**
  (noise ∝ strength on the masked latents) — so it *refines* the roughed-in structure into natural
  skin instead of generating from scratch, keeping limbs anatomically coherent while avoiding the
  plastic look of a single high-strength pass. No new dependency or model load: `inpaint_twostage`
  composes the existing `inpaint_lama` + `inpaint_sdxl`, and inherits crop-to-region + the
  bit-identical composite for free (`_NATIVE_RES["twostage"] = 1024`). Selectable as `backend=
  "twostage"` on `remove_tattoo`, the app dropdown, `scripts/smoke_test.py`, and the all-in-one
  node; wired into the composable graph via the `DeinkTwoStageBackend` provider node. **`backend=
  "auto"` now routes large/limb-spanning components through two-stage** instead of plain SDXL (so
  its output is no longer bit-identical to the retired sdxl-auto). FLUX.1 Fill now ships as its own
  `backend="flux"` (see "Better fill models" below); next, use it as the texture stage here and
  tune the per-stage strength on the deepest holes.
- **Guide the diffusion fill.** (a) **Prompt/negative-prompt** in `deink/inpaint.py`: describe the
  target ("bare skin, natural skin texture, even lighting, muscle") and negative-prompt the source
  ("tattoo, ink, text, lettering") so the model doesn't re-hallucinate ink — a common large-tattoo
  artifact. (b) **ControlNet (depth) guidance.** ✅ *Implemented (`backend="sdxl_controlnet"`).* A
  depth map of the surrounding limb is auto-estimated from the crop (Depth-Anything V2, transformers-
  native — no CUDA-ext build, no new pip dep, same discipline as OWLv2 over YOLO-World) and fed to an
  SDXL depth ControlNet, so the generated skin follows the actual arm/leg volume across the hole
  instead of the flat/warped anatomy plain SDXL invents on a limb-sized mask. `inpaint_sdxl_controlnet`
  mirrors `inpaint_sdxl` exactly (run square @1024, resize back, composite on the mask →
  bit-identical outside it) and flows through the same crop-to-region seam
  (`_NATIVE_RES["sdxl_controlnet"] = 1024`) — the depth is estimated *inside* the method from the
  already-cropped window, so nothing extra threads through `_fill`/`remove_tattoo`. Model ids are
  `diffusers/controlnet-depth-sdxl-1.0-small` and `depth-anything/Depth-Anything-V2-Small-hf` (both
  **un-gated** — no `hf auth`), overridable via `DEINK_SDXL_CN_DEPTH` / `DEINK_DEPTH_MODEL`;
  `controlnet_conditioning_scale` (default 0.5) trades depth adherence against fill freedom.
  Selectable on `remove_tattoo`, the app dropdown, `scripts/smoke_test.py`, and the all-in-one node;
  wired into the composable graph via `DeinkSdxlControlNetBackend`. **Pose (OpenPose/DWpose) and an
  IP-Adapter skin reference** remain future items — pose would need a new dep (`controlnet_aux`),
  which the pipeline avoids.
- **Tune SDXL.** Now validated end-to-end; tune `strength`, `guidance_scale`, steps, and the skin
  prompt in `deink/inpaint.py`. LaMa wins on speed and plain skin; SDXL wins on large/curved
  regions and areas crossing anatomical boundaries. For removal, keep `strength` high enough to
  erase bold ink but not so high it invents unrelated detail.
- **Better fill models.** In rough order of quality/effort:
  - **FLUX.1 Fill [dev]** (Black Forest Labs) — current SOTA inpainting, better structure/texture
    than SDXL. ✅ *Implemented (`backend="flux"`).* Loaded with a **GGUF-quantized** transformer
    (city96's non-gated mirror, `DEINK_FLUX_GGUF`-overridable) so the 12B model fits 16 GB, paired
    with the base pipeline's VAE / text encoders / scheduler under `enable_model_cpu_offload()`.
    The base repo `black-forest-labs/FLUX.1-Fill-dev` is **gated** — accept its license and
    `hf auth login` before first use. FLUX is guidance-distilled: no negative prompt, high
    `guidance_scale` (~30). Flows through the same crop-to-region + bit-identical composite seam as
    SDXL (`_NATIVE_RES["flux"] = 1024`); wired into the composable graph via `DeinkFluxBackend`.
    Next: FLUX Fill as the two-stage texture stage; benchmark vs. `twostage` on the deepest holes.
  - **PowerPaint** / **BrushNet** — diffusion inpainters with an explicit *object-removal* mode,
    designed for "remove this, fill plausibly."
  - **MI-GAN** / **MAT** — ✅ *Implemented (`backend="migan"` / `backend="mat"`).* Fast
    **feed-forward** (non-diffusion) large-hole specialists, better than LaMa on big masks while
    staying fast. Both are pure-PyTorch (no CUDA-ext build) and **vendored** under
    `deink/vendor/` (adapted from IOPaint) rather than added as a dep — `iopaint` pins
    `diffusers==0.27.2`, which conflicts with the modern diffusers the FLUX backend needs. Weights
    are **un-gated** GitHub-release assets that download lazily (override via `DEINK_MIGAN_URL` /
    `DEINK_MAT_URL`) — no `hf auth login`. Both flow through the same crop-to-region + hard-mask
    feathered composite as LaMa (`_NATIVE_RES = 512`); MAT is a fixed-512 StyleGAN net, MI-GAN a
    TorchScript trace. Wired into the composable graph via `DeinkMiganBackend` / `DeinkMatBackend`.
    **Caveat — they ship Places2 (scene) weights, not skin.** GPU-tested on the removal set they
    remove tattoos on small masks but are softer than LaMa with minor artifacts, and on large holes
    over a person they *hallucinate scene fragments*. So they stay **opt-in**: `backend="auto"`
    keeps routing large/limb-spanning components to `twostage` (domain-appropriate). Flip
    `_AUTO_LARGE_BACKEND` in `deink/pipeline.py` to `"mat"`/`"migan"` to trade quality for speed.
    **ZITS** — a 4-model wireframe/edge/structure pipeline — remains a future item (most code for
    least marginal gain once MI-GAN + MAT exist).
  - **ControlNet guidance** — ✅ *depth implemented (`backend="sdxl_controlnet"`, see "Guide the
    diffusion fill" above)*; pose remains a future item (needs `controlnet_aux`). The **two-pass**
    LaMa-structure → diffusion-texture combo is also shipped as `backend="twostage"`.
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
