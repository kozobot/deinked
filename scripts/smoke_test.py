"""Quick end-to-end check for the segment-and-inpaint pipeline.

Usage:
    python scripts/smoke_test.py [IMAGE] [--backend lama|sdxl] [--prompt "a tattoo."]

Picks a sample from data/ if no image is given. Saves a side-by-side to
scratch/smoke_<backend>.png and prints timing + whether a tattoo was found.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# Make the repo root importable when run as `python scripts/smoke_test.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from deink import remove_tattoo


def find_sample() -> str | None:
    for pat in (
        "data/deinked/test/*",
        "data/stock/laser-removal/*",
        "data/deinked/tattoo/*",
    ):
        hits = sorted(glob.glob(pat))
        hits = [h for h in hits if h.lower().endswith((".jpg", ".jpeg", ".png"))]
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--backend", default="lama", choices=["lama", "sdxl"])
    ap.add_argument("--prompt", default="a tattoo.")
    ap.add_argument("--tile", action="store_true", help="tiled detection (higher recall, slower)")
    args = ap.parse_args()

    src = args.image or find_sample()
    if not src:
        raise SystemExit("No image given and no sample found under data/.")
    print(f"input: {src}  backend: {args.backend}")

    img = Image.open(src).convert("RGB")
    t0 = time.time()
    res = remove_tattoo(img, backend=args.backend, prompt=args.prompt, tile=args.tile)
    dt = time.time() - t0
    print(f"found tattoo: {res.found}   elapsed: {dt:.1f}s")

    os.makedirs("scratch", exist_ok=True)
    w, h = img.size
    board = Image.new("RGB", (w * 2, h), "white")
    board.paste(img, (0, 0))
    board.paste(res.image, (w, 0))
    out = f"scratch/smoke_{args.backend}.png"
    board.save(out)
    res.mask.save(f"scratch/smoke_{args.backend}_mask.png")
    print(f"wrote {out} (left: input, right: result)")


if __name__ == "__main__":
    main()
