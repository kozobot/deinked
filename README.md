# deinked — Tattoo Removal for ComfyUI

Remove tattoos from images with a segment-and-inpaint pipeline built on pretrained foundation
models — no training required at inference time. This repository is a **ComfyUI custom-node
plugin**; the underlying `deink` pipeline package and its (optional, offline) training tooling live
under [`core/`](core/).

The pipeline is `localize → refine mask → inpaint → composite`. The nodes here provide the pieces
that have no commodity-node equivalent and reuse ComfyUI's ecosystem for the rest.

## Install

**Via ComfyUI Manager** (recommended): search for *deinked* and install, or "Install via Git URL"
with this repo's URL. Manager clones the repo root into `ComfyUI/custom_nodes/` and installs
`requirements.txt`.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kozobot/deinked
pip install -r deinked/requirements.txt
```

### Requirements

- **torch/torchvision** come from your ComfyUI host and are *not* installed by this plugin. On an
  NVIDIA Blackwell GPU (sm_120, e.g. RTX 5070 Ti) you need a CUDA build with sm_120/cu128 kernels —
  match `core/environment.yml`.
- **Box-localization path** expects storyicon's
  [`comfyui_segment_anything`](https://github.com/storyicon/comfyui_segment_anything)
  (`GroundingDinoSAMSegment`) installed — a *soft* prerequisite interoperated via the standard
  `MASK` type, not imported. Install it via Manager. The SegFormer + inpaint path works without it.
- **SegFormer checkpoint** (for the `DeinkSegFormer` node): a fine-tuned model at
  `core/data/models/tattoo-segformer/` (or point `DEINK_TATTOOSEG_DIR` at a copy). Not shipped in
  git; train one with `core/scripts/train_tattooseg.py` or download a release checkpoint. The node
  no-ops to an empty mask when absent, so graphs still run.

## Nodes

| Node | In → Out | What it does |
|------|----------|--------------|
| **Deink SegFormer** | IMAGE → MASK | Fine-tuned pixel-level tattoo localizer (the one localizer commodity nodes lack). No-ops to empty mask with no checkpoint. |
| **Deink Refine Mask** | MASK → MASK | Dilate (cover ink edges) + Gaussian feather (seamless blend). Tuned defaults (8/5). |
| **Deink Split Mask by Size** | MASK → MASK, MASK | Split components into `small` / `large` by area, to route each to a different inpainter. |
| **Deink Inpaint** | IMAGE + MASK → IMAGE | LaMa / SDXL / `auto`, at the backend's *native resolution* (crop-to-region), compositing so pixels outside the mask stay bit-identical. Treats the mask as final — refine it upstream. |
| **Deink Remove Tattoo** | IMAGE → IMAGE, MASK | All-in-one: the whole pipeline in one node (defaults to the SegFormer localizer). |

## Typical graphs

**Box path (commodity localization):**
```
Load Image → GroundingDinoSAMSegment (prompt "a tattoo") → Deink Refine Mask → Deink Inpaint → Save
```

**Custom-segmenter path:**
```
Load Image → Deink SegFormer → Deink Refine Mask → Deink Inpaint → Save
```

**Interactive:** run either localizer, right-click the MASK → *Open in MaskEditor* to hand-correct,
then feed it to Deink Inpaint.

**Mixed sizes:** `Deink Split Mask by Size` → small→`Deink Inpaint(lama)`, large→`Deink Inpaint(sdxl)`
(or just use `Deink Inpaint` with `backend=auto`).

Example workflows live in [`example_workflows/`](example_workflows/).

## The `core/` directory

`core/` is the original `deinked` project — the importable `deink` pipeline package, the marimo
app, and the offline **training** scripts (`derive_masks.py`, `gen_synthetic_seg.py`,
`train_tattooseg.py`, `eval_seg.py`). Training is intentionally *not* a ComfyUI concern (ComfyUI is
an inference engine); the plugin only consumes the SegFormer checkpoint that training produces. The
plugin imports the canonical `deink` code from `core/` — there is no vendored copy to drift.
