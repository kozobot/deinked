"""GPU-free unit tests for the mask-refinement extras in ``deink.pipeline``.

Covers the three README §2 "Mask refinement" sub-items — per-region adaptive dilation,
edge-aware (guided-filter) feathering, and seam/color harmonization — plus the invariants the
composite relies on. No model load: a stub inpainter stands in for the real backends, so this
runs on CPU in CI. The overriding contract is *bit-identical outside the feather support*
(``refined == 0``) and *byte-identical output when every knob is off*.
"""

import cv2
import numpy as np
from PIL import Image

from deink.pipeline import _fill, _harmonize, refine_mask


def _two_blob_mask(h=240, w=240):
    """A tiny blob and a big blob, so per-component behaviour is observable."""
    m = np.zeros((h, w), dtype=bool)
    m[20:28, 20:40] = True           # tiny script-like blob
    m[80:200, 80:220] = True         # big sleeve-like blob
    return m


class _StubInpainter:
    """Returns a solid fill. ``feedforward`` mirrors LaMa (raw fill); otherwise it mirrors the
    diffusion backends, which composite internally against the soft mask (see ``inpaint_sdxl``)."""

    def __init__(self, feedforward=True):
        self.feedforward = feedforward

    def inpaint(self, image, mask, backend, **kw):
        solid = np.asarray(image.convert("RGB")).copy()
        solid[:] = (0, 180, 60)
        solid_img = Image.fromarray(solid)
        if self.feedforward:
            return solid_img
        mask_img = mask if isinstance(mask, Image.Image) else Image.fromarray(mask)
        return Image.composite(solid_img, image, mask_img.convert("L"))


# --- default path is byte-identical to the original dilate + Gaussian --------------------

def test_default_refine_byte_identical():
    m = _two_blob_mask()
    old = (m > 0).astype(np.uint8) * 255
    old = cv2.dilate(old, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    old = cv2.GaussianBlur(old, (11, 11), 0).astype(np.float32) / 255.0
    new = refine_mask(m, dilate=8, feather=5)
    assert new.dtype == np.float32
    assert np.array_equal(old, new)


# --- per-region adaptive dilation --------------------------------------------------------

def test_adaptive_grows_large_more_than_small():
    m = _two_blob_mask()
    ad = refine_mask(m, dilate=8, feather=0, adaptive=True) > 0.5
    base = refine_mask(m, dilate=8, feather=0) > 0.5
    # Coverage is a superset of the uniform path (floor == global dilate).
    assert ad.sum() >= base.sum()
    # The big blob's extra grow (adaptive - uniform) exceeds the tiny blob's.
    small_extra = (ad[15:40, 15:55].sum()) - (base[15:40, 15:55].sum())
    large_extra = (ad[70:210, 70:230].sum()) - (base[70:210, 70:230].sum())
    assert large_extra > small_extra


def test_adaptive_cap_bounds_growth():
    m = _two_blob_mask()
    capped = refine_mask(m, dilate=8, feather=0, adaptive=True, dilate_max=10) > 0.5
    loose = refine_mask(m, dilate=8, feather=0, adaptive=True, dilate_max=200) > 0.5
    assert capped.sum() < loose.sum()


# --- edge-aware feather ------------------------------------------------------------------

def test_guided_feather_zero_beyond_band_and_in_range():
    m = np.zeros((120, 120), dtype=bool)
    m[40:80, 40:80] = True
    guide = (np.random.default_rng(0).random((120, 120, 3)) * 255).astype(np.uint8)
    feather = 5
    ef = refine_mask(m, dilate=8, feather=feather, guide=guide)
    assert ef.dtype == np.float32 and ef.min() >= 0.0 and ef.max() <= 1.0
    # Guided filter support is bounded by the dilate + 2*feather band; far outside stays exactly 0.
    assert ef[0, 0] == 0.0 and ef[-1, -1] == 0.0
    # It is a genuinely different feather from the uniform Gaussian.
    gauss = refine_mask(m, dilate=8, feather=feather)
    assert not np.array_equal(ef, gauss)


def test_guided_feather_follows_luminance_edge():
    # Guide has a hard vertical luminance edge; the feather should snap to it rather than
    # bleed a symmetric ring across it.
    guide = np.zeros((120, 120, 3), dtype=np.uint8)
    guide[:, 60:] = 255
    m = np.zeros((120, 120), dtype=bool)
    m[40:80, 30:70] = True  # straddles the edge at x=60
    ef = refine_mask(m, dilate=6, feather=6, guide=guide)
    row = ef[60]
    # The steepest step in the feather sits at/near the luminance edge (x≈60), not far from it.
    step_x = int(np.argmax(np.abs(np.diff(row))))
    assert 52 <= step_x <= 68


# --- seam / color harmonization ----------------------------------------------------------

def _rand_img(h=120, w=120, seed=1):
    return Image.fromarray((np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8))


def test_harmonize_outside_support_bit_identical_via_fill():
    base = _rand_img()
    m = np.zeros((120, 120), dtype=bool)
    m[30:70, 30:90] = True
    refined = refine_mask(m, dilate=8, feather=5)
    outside = ~(refined > 0)
    b = np.asarray(base)
    for feedforward, be in ((True, "lama"), (False, "sdxl")):
        out = np.asarray(
            _fill(_StubInpainter(feedforward), base, refined, be,
                  harmonize=True, harmonize_kw={"ring_px": 8})
        )
        assert out.shape == b.shape
        assert np.array_equal(out[outside], b[outside])


def test_fill_harmonize_off_is_unchanged():
    base = _rand_img(seed=2)
    m = np.zeros((120, 120), dtype=bool)
    m[30:70, 30:90] = True
    refined = refine_mask(m, dilate=8, feather=5)
    # Feed-forward with harmonize off == plain feathered composite of the raw fill.
    plain = np.asarray(_fill(_StubInpainter(True), base, refined, "lama", harmonize=False))
    solid = np.asarray(base).copy()
    solid[:] = (0, 180, 60)
    expect = np.asarray(
        Image.composite(Image.fromarray(solid), base,
                        Image.fromarray((refined * 255).astype(np.uint8), "L"))
    )
    assert np.array_equal(plain, expect)


def test_harmonize_degenerate_masks_do_not_raise():
    base = _rand_img(seed=3)
    fill = _rand_img(seed=4)
    # Empty, full-frame (border-touching), and tiny masks all fall back gracefully.
    for m in (np.zeros((120, 120), bool),
              np.ones((120, 120), bool),
              (lambda z: (z.__setitem__((slice(59, 61), slice(59, 61)), True), z)[1])(
                  np.zeros((120, 120), bool))):
        refined = m.astype(np.float32)
        out = _harmonize(fill, base, refined, ring_px=8)
        assert np.asarray(out).shape == np.asarray(base).shape
