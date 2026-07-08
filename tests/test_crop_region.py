"""GPU-free unit tests for the crop-to-region window logic in ``deink.pipeline``.

``_region_bbox`` decides the padded, native-resolution window an inpaint pass runs on. These
check the invariants the paste-back relies on (box covers the whole mask, stays inside the
image) plus the padding/squaring behaviour — no model load, runs on CPU in CI.
"""

import numpy as np

from deink.pipeline import _region_bbox


def _mask(h, w, box):
    """Bool (h, w) mask with ``box`` = (l, t, r, b) set True."""
    m = np.zeros((h, w), dtype=bool)
    l, t, r, b = box
    m[t:b, l:r] = True
    return m


def test_empty_mask_returns_none():
    assert _region_bbox(np.zeros((100, 100), dtype=bool), (100, 100)) is None


def test_box_contains_mask_and_pads():
    # A 20x20 blob at (40,40) with pad_frac=0.5 grows ~10px each side.
    m = _mask(200, 200, (40, 40, 60, 60))
    l, t, r, b = _region_bbox(m, (200, 200), pad_frac=0.5, min_size=0)
    assert l <= 40 and t <= 40 and r >= 60 and b >= 60  # fully covers the mask
    assert (l, t) == (30, 30) and (r, b) == (70, 70)    # 10px pad, then squared (already square)


def test_box_clamped_to_image_bounds():
    # Blob hugging the top-left corner: padding must not run negative.
    m = _mask(200, 300, (0, 0, 20, 20))
    l, t, r, b = _region_bbox(m, (300, 200), pad_frac=1.0, min_size=0)
    assert l >= 0 and t >= 0 and r <= 300 and b <= 200
    assert l == 0 and t == 0


def test_square_expansion_for_min_size():
    # Small mask with a native-res floor pulls in a min_size-sided square window.
    m = _mask(4000, 4000, (2000, 2000, 2050, 2050))
    l, t, r, b = _region_bbox(m, (4000, 4000), pad_frac=0.5, min_size=1024)
    assert r - l == 1024 and b - t == 1024              # square at the native floor
    assert l <= 2000 and t <= 2000 and r >= 2050 and b >= 2050


def test_min_size_clamped_to_short_side():
    # min_size larger than the image collapses to the short side, still a valid in-bounds box.
    m = _mask(500, 800, (300, 200, 340, 240))
    l, t, r, b = _region_bbox(m, (800, 500), pad_frac=0.5, min_size=1024)
    assert r - l == 500 and b - t == 500                # capped at min(W, H)
    assert 0 <= l and 0 <= t and r <= 800 and b <= 500


def test_wide_mask_keeps_rect_when_square_infeasible():
    # Mask wider than the short (vertical) side: a square can't cover it, so the padded rect
    # is kept — and must still contain the whole mask.
    m = _mask(200, 600, (10, 90, 590, 110))
    l, t, r, b = _region_bbox(m, (600, 200), pad_frac=0.1, min_size=0)
    assert l <= 10 and r >= 590 and t <= 90 and b >= 110
    assert r - l != b - t                               # non-square rect retained
    assert 0 <= l and 0 <= t and r <= 600 and b <= 200
