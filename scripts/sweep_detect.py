"""Sweep open-vocab detection settings on one image to pick per-image thresholds.

Detection only — no SAM, no inpaint — so it is fast enough to sweep a grid. For every
combination of detector x prompt x box-threshold x text-threshold x max-area-frac it prints
the number of surviving boxes and their per-box area fractions (box area / image area), so
you can see how recall responds and choose settings for a hard image. Lower thresholds
recover fainter tattoos at the cost of false positives; ``max_area_frac`` caps how large a
box may be (the guard against SAM masking the whole subject). Sweeping ``--detectors gdino
owlv2 ensemble`` A/Bs the detectors side by side (OWLv2 has no text threshold, so that column
is ignored for it).

Usage:
    python scripts/sweep_detect.py [IMAGE] \
        --detectors gdino owlv2 ensemble \
        --prompts "a tattoo." "tattoo. ink drawing on skin." \
        --box-thresholds 0.15 0.25 --text-thresholds 0.15 0.2 \
        --max-area-fracs 0.25 0.5 [--tile] [--out DIR]

With ``--out DIR`` each combination's boxes are drawn on the image and saved as
``sweep_<i>.png`` so results are eyeballable. Falls back to a sample under data/ when no
image is given (same lookup as smoke_test.py).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

# Make the repo root importable when run as `python scripts/sweep_detect.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from deink.segment import TattooSegmenter

from smoke_test import find_sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--detectors", nargs="+", default=["gdino"],
                    choices=["gdino", "owlv2", "ensemble"],
                    help="open-vocab detector(s) to sweep (default: gdino)")
    ap.add_argument("--prompts", nargs="+", default=["a tattoo.", "tattoo. ink drawing on skin."])
    ap.add_argument("--box-thresholds", nargs="+", type=float, default=[0.15, 0.25])
    ap.add_argument("--text-thresholds", nargs="+", type=float, default=[0.15, 0.2])
    ap.add_argument("--max-area-fracs", nargs="+", type=float, default=[0.25])
    ap.add_argument("--tile", action="store_true", help="tiled detection (higher recall, slower)")
    ap.add_argument("--out", default=None, help="directory to write per-combo box-overlay PNGs")
    args = ap.parse_args()

    src = args.image or find_sample()
    if not src:
        raise SystemExit("No image given and no sample found under data/.")

    img = Image.open(src).convert("RGB")
    W, H = img.size
    area = float(W * H)
    print(f"input: {src}  size: {W}x{H}  tile: {args.tile}")

    seg = TattooSegmenter()
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    combos = itertools.product(
        args.detectors, args.prompts, args.box_thresholds, args.text_thresholds, args.max_area_fracs
    )
    for i, (det, prompt, box_t, text_t, maf) in enumerate(combos):
        if args.tile:
            boxes = seg.detect_boxes_tiled(
                img, prompt, box_threshold=box_t, text_threshold=text_t,
                max_area_frac=maf, detector=det,
            )
        else:
            boxes = seg.detect_boxes(
                img, prompt, box_threshold=box_t, text_threshold=text_t,
                max_area_frac=maf, detector=det,
            )
        fracs = [round(((b[2] - b[0]) * (b[3] - b[1])) / area, 3) for b in boxes]
        print(
            f"[{i:2d}] det={det:<8} prompt={prompt!r:45} box={box_t:<4} text={text_t:<4} "
            f"maf={maf:<4} n={len(boxes):2d}  area_fracs={fracs}"
        )

        if args.out:
            canvas = img.copy()
            draw = ImageDraw.Draw(canvas)
            for b in boxes:
                draw.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                               outline="red", width=3)
            path = os.path.join(args.out, f"sweep_{i}.png")
            canvas.save(path)
            print(f"      wrote {path}")


if __name__ == "__main__":
    main()
