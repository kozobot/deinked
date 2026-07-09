"""Derive real tattoo masks by differencing aligned tattoo/clean pairs.

The paired ``*_tattoo`` / ``*_clean`` photos in ``data/deinked/rawdata`` are the same image
before and after an artist removed the tattoo, so ``|tattoo - clean|`` localizes exactly the
inked pixels — free, real-image segmentation labels to train the custom tattoo segmenter on
(no hand-labeling). The catch is label noise: JPEG re-encoding, global tone/brightness shifts
from the edit, skin smoothing, and shadows all fire in a naive diff. This script fights that:

    1. Median-align clean -> tattoo per LAB channel to cancel the global tone shift (median is
       robust to the tattoo minority pixels, so it aligns on skin without knowing where skin is).
    2. LAB Delta-E (perceptual colour distance) instead of raw RGB diff.
    3. Per-image Otsu threshold (black ink on dark skin vs bright ink need different cutoffs).
    4. Morphological open->close + small-component area filter to kill speckle/halos.
    5. An *ignore band*: the high-Delta-E core is labelled tattoo (1), clearly-unchanged skin
       is background (0), and the uncertain mid-Delta-E ring is ignore (255) so the training
       loss never penalizes the model on genuinely ambiguous edge pixels.

Outputs 3-value label PNGs (0/1/255) to ``data/silhouette/derived/``, a QC contact sheet, and
a frozen train/eval split manifest (JSON) so eval images never leak into training.

Usage:
    python scripts/derive_masks.py [--limit N] [--rawdata DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BG, FG, IGNORE = 0, 1, 255  # label values in the emitted mask PNGs

# The three real-photo families in rawdata. retouchme is the largest/highest-quality slice;
# laser-removal + the studio "101000555image" set are held out entirely as an OOD eval set.
EVAL_FAMILIES = ("laser-removal", "101000555image")
N_RETOUCHME_EVAL = 20  # retouchme pairs also held out for in-distribution eval


def find_pairs(rawdata: str) -> list[tuple[str, str, str]]:
    """Discover (stem, tattoo_path, clean_path) triples from ``*_tattoo`` / ``*_clean`` files."""
    pairs = []
    for tat in sorted(glob.glob(os.path.join(rawdata, "*_tattoo.*"))):
        stem = os.path.basename(tat).rsplit("_tattoo.", 1)[0]
        clean_hits = glob.glob(os.path.join(rawdata, f"{stem}_clean.*"))
        if clean_hits:
            pairs.append((stem, tat, clean_hits[0]))
    return pairs


def _family(stem: str) -> str:
    for fam in EVAL_FAMILIES:
        if stem.startswith(fam):
            return fam
    return "retouchme"


def median_align(clean_lab: np.ndarray, tattoo_lab: np.ndarray) -> np.ndarray:
    """Shift each LAB channel of ``clean`` so its median matches ``tattoo``'s.

    Cancels the global tone/brightness shift the artist edit introduces. The median is robust
    to the tattoo pixels (a minority), so the two images align on skin without segmenting it.
    """
    out = clean_lab.astype(np.float32)
    for c in range(3):
        out[..., c] += float(np.median(tattoo_lab[..., c]) - np.median(clean_lab[..., c]))
    return np.clip(out, 0, 255)


def derive_label_map(
    tattoo_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    band: float = 0.5,
    min_area_frac: float = 0.0005,
    abs_floor: int = 12,
    open_ks: int = 3,
    close_ks: int = 7,
) -> np.ndarray:
    """Derive a 3-value (0/1/255) tattoo label map from an aligned tattoo/clean RGB pair.

    ``band`` sets the ignore ring width: pixels with Delta-E in ``[band*t, t)`` (t = threshold)
    are labelled ignore. ``min_area_frac`` drops connected tattoo components smaller than this
    fraction of the image (speckle/halo rejection). ``abs_floor`` is the minimum *absolute*
    Delta-E for a pixel to count as tattoo — the guard against a near-identical pair (only JPEG
    noise / tone drift) fabricating a "tattoo" out of noise, since Otsu on such a pair returns a
    near-zero threshold. Pure numpy/cv2 so it is unit-testable without a GPU.
    """
    assert tattoo_rgb.shape == clean_rgb.shape, "tattoo/clean must match dimensions"
    tat_lab = cv2.cvtColor(tattoo_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    clean_lab = median_align(cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2LAB).astype(np.float32), tat_lab)

    # Absolute LAB Delta-E (perceptual distance), NOT min-max normalized — normalizing would
    # stretch pure noise up to 255 and hallucinate tattoos on unchanged pairs.
    dE = np.sqrt(((tat_lab - clean_lab) ** 2).sum(axis=-1))  # (H, W)
    dE8 = np.clip(dE, 0, 255).astype(np.uint8)
    t_otsu, _ = cv2.threshold(dE8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # per-image Otsu
    t = max(float(t_otsu), float(abs_floor))  # floor keeps low-signal pairs from over-firing

    core = (dE8 >= t).astype(np.uint8)
    if open_ks:
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks)))
    if close_ks:
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks)))

    # Drop tiny connected components (JPEG speckle, removal halos).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    min_area = min_area_frac * core.size
    keep = np.zeros_like(core)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1

    label = np.full(core.shape, BG, dtype=np.uint8)
    label[(dE8 >= band * t) & (dE8 < t)] = IGNORE  # uncertain ring
    label[keep == 1] = FG
    return label


def overlay(tattoo_rgb: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Red = tattoo, yellow = ignore, over the tattoo image (for QC eyeballing)."""
    vis = tattoo_rgb.copy()
    vis[label == FG] = (0.4 * vis[label == FG] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    vis[label == IGNORE] = (0.6 * vis[label == IGNORE] + 0.4 * np.array([255, 255, 0])).astype(np.uint8)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rawdata", default="data/deinked/rawdata")
    ap.add_argument("--out", default="data/silhouette/derived")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N pairs (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = find_pairs(args.rawdata)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"No *_tattoo/*_clean pairs under {args.rawdata}")
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(pairs)} pairs from {args.rawdata} -> {args.out}")

    contact_rows, split = [], {"train": [], "eval": []}
    retouchme = [p[0] for p in pairs if _family(p[0]) == "retouchme"]
    random.Random(args.seed).shuffle(retouchme)
    eval_retouchme = set(retouchme[:N_RETOUCHME_EVAL])

    for i, (stem, tat_path, clean_path) in enumerate(pairs):
        tat = np.array(Image.open(tat_path).convert("RGB"))
        clean = np.array(Image.open(clean_path).convert("RGB"))
        if tat.shape != clean.shape:
            print(f"  skip {stem}: dim mismatch {tat.shape} vs {clean.shape}")
            continue

        label = derive_label_map(tat, clean)
        Image.fromarray(label, mode="L").save(os.path.join(args.out, f"{stem}.png"))

        is_eval = _family(stem) in EVAL_FAMILIES or stem in eval_retouchme
        split["eval" if is_eval else "train"].append(stem)

        fg_frac = float((label == FG).mean())
        print(f"  [{i:3d}] {stem:32} fg={fg_frac:.3f} {'EVAL' if is_eval else 'train'}")
        if len(contact_rows) < 8:  # a small QC sheet
            h = 240
            scale = h / tat.shape[0]
            w = int(tat.shape[1] * scale)
            row = np.hstack([
                cv2.resize(tat, (w, h)),
                cv2.resize(overlay(tat, label), (w, h)),
            ])
            contact_rows.append(row)

    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump(split, f, indent=2)
    print(f"split: {len(split['train'])} train / {len(split['eval'])} eval -> {args.out}/split.json")

    if contact_rows:
        os.makedirs("scratch", exist_ok=True)
        maxw = max(r.shape[1] for r in contact_rows)
        sheet = np.vstack([np.pad(r, ((0, 0), (0, maxw - r.shape[1]), (0, 0))) for r in contact_rows])
        out = "scratch/derived_masks_qc.png"
        Image.fromarray(sheet).save(out)
        print(f"QC contact sheet (left: tattoo, right: red=mask yellow=ignore) -> {out}")


if __name__ == "__main__":
    main()
