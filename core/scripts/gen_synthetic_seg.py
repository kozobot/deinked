"""Generate synthetic (image, mask) pairs by compositing tattoo clipart onto real skin.

A port of the legacy ``Deink_04_Silhouette_Generate_Deink_Data`` idea, retargeted from
GAN training data to *segmentation* labels: paste RGBA tattoo clipart onto a tattoo-free skin
photo, and the composited alpha channel IS a perfect mask — unlimited pixel-accurate labels,
for free. This warms up the SegFormer before it fine-tunes on the noisier (but real) diff-
derived masks.

Two things keep it honest rather than a shortcut the model can cheat:

- **Silhouette constraint.** Tattoos are placed and clipped to the body silhouette
  (``data/silhouette/mask-predict/<name>.png``) so ink lands on skin, never floating over the
  background — otherwise the model learns "find the pasted graphic on the wall".
- **Realism augments.** Hard-alpha clipart pasted flat is trivially separable and won't
  transfer to faded, skin-integrated real ink. So each clip is feathered, blended so skin
  texture shows through (multiply-ish), given random opacity, desaturated/colour-shifted
  toward faded blue-green, and perspective-warped to sit on the body's curve.

Outputs paired ``img/`` + ``mask/`` PNGs (mask 0/255) to ``data/silhouette/synthetic_gen/``.

Usage:
    python scripts/gen_synthetic_seg.py [--n 400] [--seed 0] [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLIPART_DIR = "data/silhouette/tattoo_clipart"
SKIN_DIR = "data/silhouette/tattooless"
SILHOUETTE_DIR = "data/silhouette/mask-predict"


def load_cliparts(clipart_dir: str) -> list[Image.Image]:
    """Load clipart as RGBA (palette + transparency -> real alpha)."""
    clips = []
    for f in sorted(glob.glob(os.path.join(clipart_dir, "*.png"))):
        clips.append(Image.open(f).convert("RGBA"))
    return clips


def _augment_clip(clip: Image.Image, rng: random.Random) -> Image.Image:
    """Rotate + faded-ink colour/opacity realism augment on a single RGBA clip."""
    clip = clip.rotate(rng.randint(0, 359), Image.BILINEAR, expand=True)
    arr = np.array(clip).astype(np.float32)
    rgb, a = arr[..., :3], arr[..., 3:]
    # Desaturate + shift toward faded blue-green (real removed/old ink is rarely saturated black).
    gray = rgb.mean(axis=-1, keepdims=True)
    desat = rng.uniform(0.3, 0.8)
    rgb = rgb * (1 - desat) + gray * desat
    rgb += np.array([rng.uniform(-20, 0), rng.uniform(-5, 15), rng.uniform(0, 25)])  # toward teal
    a *= rng.uniform(0.55, 0.95)  # random overall opacity (faded)
    out = np.clip(np.concatenate([rgb, a], axis=-1), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _perspective_warp(layer: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mild perspective warp (RGBA) so the tattoo layer follows body curvature."""
    h, w = layer.shape[:2]
    d = 0.12
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-d, d) * w, rng.uniform(-d, d) * h] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(layer, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def composite(
    skin_rgb: np.ndarray,
    silhouette: np.ndarray,
    clips: list[Image.Image],
    rng: random.Random,
    min_tatts: int = 2,
    max_tatts: int = 5,
    scale: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite random clips onto ``skin_rgb``, clipped to ``silhouette``. Returns (rgb, mask)."""
    H, W = skin_rgb.shape[:2]
    body = silhouette > 127
    layer = np.zeros((H, W, 4), dtype=np.float32)  # accumulated RGBA tattoo layer

    ys, xs = np.where(body)
    if len(xs) == 0:  # no body detected; place anywhere
        ys, xs = np.array([H // 2]), np.array([W // 2])

    for _ in range(rng.randint(min_tatts, max_tatts)):
        clip = _augment_clip(rng.choice(clips), rng)
        s = rng.uniform(0.15, scale)
        cw, ch = max(8, int(W * s)), max(8, int(H * s))
        clip = clip.resize((cw, ch), Image.BILINEAR)
        carr = _perspective_warp(np.array(clip).astype(np.float32), rng)

        # Anchor the clip centre on a random on-body point.
        k = rng.randrange(len(xs))
        cx, cy = int(xs[k]), int(ys[k])
        x0, y0 = cx - cw // 2, cy - ch // 2
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0 + cw), min(H, y0 + ch)
        if xb <= xa or yb <= ya:
            continue
        sub = carr[ya - y0:yb - y0, xa - x0:xb - x0]
        # alpha-over accumulate into the layer
        dst = layer[ya:yb, xa:xb]
        sa = sub[..., 3:] / 255.0
        dst[..., :3] = sub[..., :3] * sa + dst[..., :3] * (1 - sa)
        dst[..., 3:] = np.clip(sub[..., 3:] + dst[..., 3:] * (1 - sa), 0, 255)

    # Feather the alpha and clip to the body silhouette.
    alpha = cv2.GaussianBlur(layer[..., 3], (5, 5), 0) / 255.0
    alpha = alpha * body
    a3 = alpha[..., None]
    # Blend toward multiply so underlying skin texture shows through the ink (realism).
    ink = layer[..., :3]
    straight = ink * a3 + skin_rgb * (1 - a3)
    multiplied = (skin_rgb / 255.0) * (ink / 255.0) * 255.0
    out = straight * 0.65 + (multiplied * a3 + skin_rgb * (1 - a3)) * 0.35
    out = np.clip(out, 0, 255).astype(np.uint8)

    mask = (alpha > 0.25).astype(np.uint8) * 255
    return out, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="number of synthetic pairs to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/silhouette/synthetic_gen")
    ap.add_argument("--clipart", default=CLIPART_DIR)
    ap.add_argument("--skin", default=SKIN_DIR)
    ap.add_argument("--silhouette", default=SILHOUETTE_DIR)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    clips = load_cliparts(args.clipart)
    skins = sorted(glob.glob(os.path.join(args.skin, "*")))
    skins = [s for s in skins if s.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not clips or not skins:
        raise SystemExit(f"Need clipart in {args.clipart} and skins in {args.skin}")

    img_dir, mask_dir = os.path.join(args.out, "img"), os.path.join(args.out, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    print(f"{len(clips)} cliparts x {len(skins)} skins -> {args.n} pairs in {args.out}")

    for i in range(args.n):
        skin_path = skins[i % len(skins)] if i < len(skins) else rng.choice(skins)
        name = os.path.splitext(os.path.basename(skin_path))[0]
        skin = np.array(Image.open(skin_path).convert("RGB")).astype(np.float32)
        sil_path = os.path.join(args.silhouette, f"{name}.png")
        if os.path.isfile(sil_path):
            sil = np.array(Image.open(sil_path).convert("L").resize((skin.shape[1], skin.shape[0])))
        else:
            sil = np.full(skin.shape[:2], 255, np.uint8)  # no silhouette -> whole image is "body"

        img, mask = composite(skin, sil, clips, rng)
        Image.fromarray(img).save(os.path.join(img_dir, f"gen-{i:04d}.png"))
        Image.fromarray(mask, "L").save(os.path.join(mask_dir, f"gen-{i:04d}.png"))
        if i < 8:
            os.makedirs("scratch", exist_ok=True)
            side = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)])
            Image.fromarray(side).save(f"scratch/synth_gen_{i}.png")
    print(f"done. Sample previews (img|mask) in scratch/synth_gen_*.png")


if __name__ == "__main__":
    main()
