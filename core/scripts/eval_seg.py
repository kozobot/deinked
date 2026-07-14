"""Compare tattoo localizers on the held-out real pairs: seg vs. the box+SAM baseline.

Recall is the whole point of the custom segmenter, so this measures it directly. For every
localizer it reports, over the frozen eval split (``data/silhouette/derived/split.json``):

  - **Mask metrics** vs. the diff-derived ground truth (ignore band excluded): foreground
    IoU / recall / precision. Recall is primary (the goal), precision guards against
    over-segmentation → over-inpainting.
  - **Downstream PSNR-to-clean** (``--downstream``): actually inpaint through the mask and
    compare the tattoo region of the result to the artist-cleaned ``clean`` image. ``clean``
    *is* ground-truth removal, so this sidesteps diff-mask label noise and scores the real
    objective. Slower (runs LaMa), so it is opt-in.

Establish the **baseline first** (before/without a trained seg model):
    python scripts/eval_seg.py --methods gdino ensemble+tile
Then, once trained, add the seg rows:
    python scripts/eval_seg.py --methods gdino ensemble+tile seg seg+box --downstream

Localizers requiring a seg checkpoint are skipped (with a note) when none is available.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deink import Inpainter, TattooMaskSegmenter, TattooSegmenter
from deink.pipeline import _fill, refine_mask

IGNORE = 255

# method -> kwargs for TattooSegmenter.detect_and_segment / the seg path.
METHODS = {
    "gdino": dict(localizer="box", detector="gdino"),
    "owlv2": dict(localizer="box", detector="owlv2"),
    "ensemble": dict(localizer="box", detector="ensemble"),
    "ensemble+tile": dict(localizer="box", detector="ensemble", tile=True),
    "seg": dict(localizer="seg"),
    "seg+box": dict(localizer="seg+box", detector="gdino"),
}


def eval_split(derived="data/silhouette/derived", rawdata="data/deinked/rawdata"):
    """Yield (stem, tattoo_path, clean_path, gt_label) for the frozen eval split."""
    split = json.load(open(os.path.join(derived, "split.json")))["eval"]
    for stem in split:
        tat = glob.glob(os.path.join(rawdata, f"{stem}_tattoo.*"))
        clean = glob.glob(os.path.join(rawdata, f"{stem}_clean.*"))
        gt_path = os.path.join(derived, f"{stem}.png")
        if tat and clean and os.path.isfile(gt_path):
            yield stem, tat[0], clean[0], np.array(Image.open(gt_path).convert("L"))


def mask_scores(pred: np.ndarray, gt: np.ndarray):
    """Foreground IoU / recall / precision, excluding ignore(255) pixels."""
    valid = gt != IGNORE
    p = pred & valid
    t = (gt == 1) & valid
    inter = (p & t).sum()
    iou = inter / ((p | t).sum() + 1e-9)
    recall = inter / (t.sum() + 1e-9)
    precision = inter / (p.sum() + 1e-9)
    return iou, recall, precision


def psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float | None:
    """PSNR between a and b over ``region`` (bool). None if region empty."""
    if region.sum() == 0:
        return None
    diff = (a[region].astype(np.float32) - b[region].astype(np.float32)) ** 2
    mse = diff.mean()
    return 99.0 if mse < 1e-6 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["gdino", "ensemble+tile"],
                    choices=list(METHODS))
    ap.add_argument("--downstream", action="store_true", help="inpaint + PSNR-to-clean (slower)")
    ap.add_argument("--adaptive-dilate", action="store_true",
                    help="downstream A/B: per-region adaptive dilation")
    ap.add_argument("--edge-feather", action="store_true",
                    help="downstream A/B: image-guided (edge-aware) feather")
    ap.add_argument("--harmonize", action="store_true",
                    help="downstream A/B: skin color-match + Poisson seam")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = list(eval_split())
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No eval pairs. Run scripts/derive_masks.py first.")

    seg = TattooSegmenter()
    mask_seg = TattooMaskSegmenter()
    inpainter = Inpainter() if args.downstream else None
    seg_ok = TattooMaskSegmenter.available()

    print(f"eval set: {len(rows)} pairs   methods: {args.methods}   downstream: {args.downstream}")
    agg = {m: {"iou": [], "recall": [], "prec": [], "psnr": []} for m in args.methods}

    for stem, tat_path, clean_path, gt in rows:
        image = Image.open(tat_path).convert("RGB")
        clean = np.array(Image.open(clean_path).convert("RGB"))
        for m in args.methods:
            kw = METHODS[m]
            if kw["localizer"] in ("seg", "seg+box") and not seg_ok:
                continue  # skip seg methods until a checkpoint exists
            pred = seg.detect_and_segment(image, mask_segmenter=mask_seg, **kw)
            iou, recall, prec = mask_scores(pred, gt)
            agg[m]["iou"].append(iou)
            agg[m]["recall"].append(recall)
            agg[m]["prec"].append(prec)
            if args.downstream and pred.any():
                # Route through the shared refine + _fill seam so the A/B toggles exercise the
                # real mask-refinement code path (adaptive dilation, edge-aware feather,
                # harmonize) on the LaMa fill — same math the diffusion backends hit.
                guide = np.asarray(image) if args.edge_feather else None
                refined = refine_mask(pred, adaptive=args.adaptive_dilate, guide=guide)
                out = _fill(inpainter, image, refined, "lama",
                            harmonize=args.harmonize, harmonize_kw={"ring_px": 8})
                # score over the ground-truth tattoo region: did removal make it look clean?
                region = (gt == 1)
                pv = psnr(np.array(out), clean, region)
                if pv is not None:
                    agg[m]["psnr"].append(pv)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n{'method':14} {'IoU':>7} {'recall':>7} {'prec':>7} {'PSNR→clean':>11}  (n)")
    for m in args.methods:
        a = agg[m]
        if not a["iou"]:
            print(f"{m:14} {'—  skipped (no seg checkpoint)':>40}")
            continue
        psnr_s = f"{mean(a['psnr']):11.2f}" if a["psnr"] else f"{'—':>11}"
        print(f"{m:14} {mean(a['iou']):7.3f} {mean(a['recall']):7.3f} "
              f"{mean(a['prec']):7.3f} {psnr_s}  ({len(a['iou'])})")
    print("\nBar: seg should beat the box baselines on recall without tanking precision.")


if __name__ == "__main__":
    main()
