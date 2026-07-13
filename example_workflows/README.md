# Example workflows

- **`app_equivalent.json`** — mirrors the marimo app: `Load Image → Deink Remove Tattoo → Save`,
  with Before / After / Mask previews. `Deink Remove Tattoo` is the all-in-one node exposing the
  app's controls (backend, localizer `seg`/`box`/`seg+box`, detector, tiling, detection
  thresholds, prompt, seg_threshold, dilate/feather, plus an optional `mask` input to bypass
  detection). Defaults to `backend=auto`, `localizer=seg` like the app — note `auto` sends large
  regions to the two-stage LaMa+SDXL fill, whose SDXL model downloads (~6.5 GB) on first use;
  switch to `lama` for a fast run.
  Install into ComfyUI's `user/default/workflows/` to have it appear in the **Workflows** tab.
- **`deinked_app_composable.json`** — the same app-mirroring graph as `app_equivalent.json`
  (`Load Image → … → Save`, with Before / After / Mask previews) but with the all-in-one
  `Deink Remove Tattoo` node **decomposed** into the discrete pipeline stages
  `Deink SegFormer → Deink Refine Mask → Deink Inpaint`, so you can inspect the intermediate mask
  and tune each stage independently (or hand-correct the mask via **Open in MaskEditor** before
  inpaint). `Deink Inpaint` gets its backends from a `Deink LaMa Backend` (`min_area_frac=0`) +
  `Deink SDXL Backend` (`min_area_frac=0.02`) pair wired into its `backend_*` sockets; it
  auto-routes each mask component to the backend whose `min_area_frac` matches the component's
  size (small → LaMa, large → SDXL). Uses the `seg` localizer with the app's defaults
  (`seg_threshold=0.5`, `dilate=8`, `feather=5`, `crop`, `crop_pad=0.5`), so — like the all-in-one
  `seg` path — it needs a trained SegFormer checkpoint (else `Deink SegFormer` yields an empty mask
  and the image passes through). Install into `user/default/workflows/` to see it in the
  **Workflows** tab.
- **`two_stage.json`** — the `deinked_app_composable.json` graph with the `Deink SDXL Backend`
  swapped for a **`Deink Two-Stage Backend`** (`min_area_frac=0.02`, `strength=0.5`): large /
  limb-spanning mask components get a LaMa structure pass followed by a low-strength SDXL texture
  pass (coherent limbs without the plastic look of a single high-strength diffusion pass), while
  small components still route to the `Deink LaMa Backend`. Same SegFormer-`seg` defaults and
  checkpoint requirement as the composable graph. This matches what the all-in-one
  `Deink Remove Tattoo` node's `backend=auto` now does under the hood.
- **`two_stage_segbox.json`** — the highest-recall two-stage path, built on the all-in-one
  **`Deink Remove Tattoo`** node: `localizer=seg+box` (SegFormer **∪** GroundingDINO+SAM box path,
  for max detection recall), `backend=auto` (small blobs → LaMa, large/limb-spanning → two-stage),
  `dilate=15` (covers bold-ink edges, avoids colour bleed). Use this when the plain `seg` localizer
  misses tattoos (it roughly doubled mask coverage on subjects with many small/discrete pieces).
  The box path loads GroundingDINO + SAM (~3 GB) on first use, so the first run is slow (~90 s);
  warm runs add only a few seconds. On near-fully-tattooed subjects the box path helps little
  (its person-sized box is dropped by `max_area_frac`), so recall there is still SegFormer-bound.
- **`segformer_path.json`** — self-contained, no commodity node needed:
  `Load Image → Deink SegFormer → Deink Refine Mask → Deink Inpaint → Save Image`, with a
  `Deink LaMa Backend` + `Deink SDXL Backend` pair feeding `Deink Inpaint`'s `backend_*` sockets
  (so it auto-routes by region size). Requires a trained SegFormer checkpoint at
  `core/data/models/tattoo-segformer/` (or `$DEINK_TATTOOSEG_DIR`); with none present,
  `Deink SegFormer` yields an empty mask and the graph passes the image through unchanged.

## Box path (commodity localization)

Install storyicon's [`comfyui_segment_anything`](https://github.com/storyicon/comfyui_segment_anything)
via ComfyUI Manager, then wire:

```
Load Image ─▶ GroundingDinoSAMSegment (prompt: "a tattoo")
              └─(MASK)─▶ Deink Refine Mask ─▶ Deink Inpaint ─▶ Save Image
Load Image ───────────────────────────────────(IMAGE)─────────▲
```

`GroundingDinoSAMSegment` also needs its `GroundingDinoModelLoader` + `SAMModelLoader` inputs
(both provided by that pack). We don't ship a JSON for this path so the repo has no example that
silently fails when the commodity pack isn't installed — build it from the four nodes above.

## Interactive masking

Run either localizer, then right-click the produced MASK → **Open in MaskEditor** to hand-correct
before `Deink Inpaint`.
