"""Diagnostic: show what GroundingDINO detects for various prompts/thresholds."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from deink.segment import TattooSegmenter

img = Image.open("data/deinked/test/0010-15.jpg").convert("RGB")
W, H = img.size
print(f"image size {W}x{H}  area={W*H}")
seg = TattooSegmenter()

for prompt in ["a tattoo.", "tattoo.", "tattoos on skin.", "a small tattoo. ink drawing on skin."]:
    for thr in (0.15, 0.25):
        boxes = seg.detect_boxes(img, prompt, box_threshold=thr, text_threshold=thr)
        frac = [round(((b[2]-b[0])*(b[3]-b[1]))/(W*H), 3) for b in boxes]
        print(f"prompt={prompt!r:45} thr={thr}  n={len(boxes):2d}  area_fracs={frac}")
