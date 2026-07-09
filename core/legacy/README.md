# Legacy notebooks (fastai NoGAN era)

These are kept for reference only — they are **not** part of the current tool. The project
originally tried to *train* a tattoo-removal generator (a fastai v2 NoGAN port of SkinDeep).
That approach was retired in favour of a segment-and-inpaint pipeline built on pretrained
foundation models (see the top-level `README.md`).

What's here and why it was kept:

- `Deink_00_Utils.ipynb` — shared config/dataloaders imported by the others.
- `Deink_01`–`Deink_04_Silhouette_*` — the **silhouette / synthetic-data track**: train a
  U-Net to predict tattoo masks, then composite tattoo clipart onto tattoo-free skin to
  synthesize `(tattooed, clean)` pairs. The **mask-generation idea and the curated data are
  the one genuinely reusable asset** — a future custom tattoo-segmentation model could be
  fine-tuned from this instead of relying on GroundingDINO's text prompt.
- `Deink_05_Process_Raw_Data.ipynb` — resize/normalize raw `*_clean` / `*_tattoo` pairs.

What was removed (recoverable from git history, commit `b313960`):

- `Deink_06`–`Deink_09` — the MSE U-Net generator, critic, and GAN fine-tune. The adversarial
  stage never converged (critic stuck at chance, GAN loss frozen), and the MSE-only generator
  only faded tattoos rather than reconstructing skin.
- Scratch notebooks (`Deink_ALL`, `Deink - Predict Image`, `Untitled`, `lesson2-sgd-in-action`).
