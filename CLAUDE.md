# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`deinked` trains a deep-learning model that automatically removes tattoos from images (with video as a future goal). It began as a port of [SkinDeep](https://github.com/vijishmadhavan/SkinDeep) from FastAI 1 to FastAI 2 and has since diverged into a different training approach. Everything is **fastai v2 / PyTorch** driven from **Jupyter notebooks** — there is no standalone Python package or test suite.

## Running

Notebooks run inside an extended fastai Docker container (fastai base + `nvidia/cuda:11.3.1-base-ubuntu20.04` for GPU, plus `ffmpeg libsm6 libxext6`, and pip `nvidia-ml-py3 opencv-python Pillow`). Typical launch:

```
docker run --rm --gpus all --privileged --name fastai --ipc=host \
    -p 8888:8888 -v `pwd`:/home local/fastai-ext jupyter notebook
```

GPU is assumed; `Deink_00_Utils.get_torch_device()` reports VRAM via `pynvml` and falls back to CPU only if CUDA is unavailable.

## Pipeline architecture

The numbered `Deink_NN_*.ipynb` notebooks are an **ordered pipeline** — run them in sequence. There are two model-training tracks feeding a final tattoo-removal model:

**Silhouette track** (produces synthetic training data by learning tattoo masks):
- `01_Silhouette_Process_Raw_Data` — prep raw silhouette data into `source`/`mask`/`tattooless` sets.
- `02_Silhouette_Train_Model` — U-Net that predicts a tattoo mask from an image → `silhouette-<arch>-epocs<N>.pkl`.
- `03_Silhouette_Predict_Masks` — run the silhouette model to generate masks into `data/silhouette/mask-predict`.
- `04_Silhouette_Generate_Deink_Data` — composite tattoo clipart onto tattoo-free images using masks to synthesize `(tattooed, clean)` training pairs into `data/deinked/{tattoo,clean}`.

**Deink (tattoo-removal) track:**
- `05_Process_Raw_Data` — resize/normalize real `*_clean` / `*_tattoo` raw pairs into `data/deinked/{clean,tattoo}`.
- `06_Train_Model` — U-Net generator that removes tattoos → `deinked-<arch>-epocs<N>.pkl`.
- `07_Save_Predictions` — run the generator over inputs, saving outputs into `data/deinked/gen` (these become "fake" examples for the critic).
- `08_Train_Critic` — binary classifier distinguishing generated (`gen`) from real (`tattoo`) images → `deinked-gen-critic-epocs<N>.pkl`.
- `09_Train_GAN` — loads a pretrained generator + critic and combines them via `GANLearner.from_learners` for adversarial fine-tuning.

`Deink_00_Utils.ipynb` is the **shared module** imported by every other notebook via `from ipynb.fs.full.Deink_00_Utils import *`. Edit it to change anything global: `image_size` (default `(480, 360)`), `batch_size`, all `data/` path constants, the `resize` transform, and the DataLoader factories (`get_sil_dls`, `get_dls`, `get_crit_dls`). Note the `get_y` helpers (`_get_sil_y`, `_get_y`) must be importable by name — they are referenced when a `.pkl` learner is reloaded, so keep their signatures stable.

## Conventions that matter

- **Naming drives everything.** Raw input pairs must be named `<name>_tattoo.<ext>` / `<name>_clean.<ext>` (deink track) or `<name>_source/_mask/_tattooless` (silhouette track); the process notebooks regex-match these suffixes and drop the suffix when saving. Files that match no convention are skipped with a warning.
- **Model/history filenames encode config**: `<track>-<arch>-epocs<N>.pkl` and matching `..._history.csv`. `<arch>` is the fastai backbone chosen in the training notebook's `bbone = ...` cell (`resnet18/34/50` or the local `xresnet34_deeper`); `<N>` is the epoch count.
- **Resumable training**: each training notebook periodically saves checkpoints to `models/*callback_saved_*_<epoch>.pth` and supports resuming from `start_epoch`. The `models/` directory and all `*.pkl` files are git-ignored (see `.gitignore`).
- **Directory roles** (from `Deink_00_Utils`): in fastai's low-res→high-res framing, the *source/tattoo* dir is the input (LR) and the *mask/clean* dir is the target (HR).

## Data layout

`data/deinked/{rawdata,tattoo,clean,gen,test}`, `data/silhouette/{rawdata,source,mask,mask-predict,tattooless,tattoo_clipart,test}`, `data/stock/{laser-removal,retouchme-source,retouchme-output,synthetic}`. Raw and generated image data is not committed.

## Non-pipeline files

`Deink - Predict Image.ipynb` loads an exported `.pkl` and runs it on an arbitrary image. `Deink_ALL.ipynb`, `lesson2-sgd-in-action.ipynb`, and `Untitled.ipynb` are scratch/legacy and not part of the pipeline.