"""GPU-free unit tests for the custom-segmenter pure logic.

Covers the checkpoint-discovery contract (``TattooMaskSegmenter.available`` /
``resolve_checkpoint_dir``) and the diff-based mask-derivation math
(``scripts/derive_masks.derive_label_map`` / ``median_align``) without loading any model or
GPU. The model-backed seg inference is exercised on-GPU by ``scripts/smoke_test.py --localizer
seg`` and ``scripts/eval_seg.py``.
"""

import os
import sys

import numpy as np

from deink.tattooseg import (
    CHECKPOINT_ENV,
    DEFAULT_CHECKPOINT_DIR,
    TattooMaskSegmenter,
    resolve_checkpoint_dir,
)

# derive_masks lives in scripts/ (not the package); import it the way sweep_detect imports smoke_test.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from derive_masks import BG, FG, IGNORE, derive_label_map, median_align  # noqa: E402


# --- checkpoint discovery -------------------------------------------------
def test_resolve_checkpoint_dir_precedence(monkeypatch):
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    assert str(resolve_checkpoint_dir()) == DEFAULT_CHECKPOINT_DIR
    # env var overrides the default...
    monkeypatch.setenv(CHECKPOINT_ENV, "/tmp/from-env")
    assert str(resolve_checkpoint_dir()) == "/tmp/from-env"
    # ...and an explicit arg overrides the env var.
    assert str(resolve_checkpoint_dir("/tmp/explicit")) == "/tmp/explicit"


def test_available_requires_config_and_weights(tmp_path):
    assert TattooMaskSegmenter.available(tmp_path) is False
    (tmp_path / "config.json").write_text("{}")
    assert TattooMaskSegmenter.available(tmp_path) is False  # config alone is not enough
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert TattooMaskSegmenter.available(tmp_path) is True


def test_available_accepts_bin_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")
    assert TattooMaskSegmenter.available(tmp_path) is True


# --- mask derivation ------------------------------------------------------
def test_median_align_cancels_tone_shift():
    # A uniform brightness offset between clean and tattoo should be removed by median align.
    tat = np.full((16, 16, 3), 120, np.float32)
    clean = np.full((16, 16, 3), 90, np.float32)  # globally darker
    aligned = median_align(clean, tat)
    assert np.allclose(np.median(aligned, axis=(0, 1)), np.median(tat, axis=(0, 1)), atol=1.0)


def test_derive_label_map_finds_the_difference():
    # Identical skin, except a bright patch present only in the "tattoo" image → that patch is FG.
    rng = np.random.default_rng(0)
    clean = rng.integers(140, 160, (64, 64, 3), dtype=np.uint8)
    tattoo = clean.copy()
    tattoo[24:40, 24:40] = [10, 10, 10]  # a dark "tattoo" block
    label = derive_label_map(tattoo, clean, min_area_frac=0.0)

    assert set(np.unique(label)).issubset({BG, FG, IGNORE})
    patch = label[24:40, 24:40]
    # The changed block is overwhelmingly foreground; the untouched border is not.
    assert (patch == FG).mean() > 0.7
    assert (label[:8, :8] == FG).mean() < 0.05


def test_derive_label_map_all_background_when_identical():
    img = np.random.default_rng(1).integers(100, 180, (48, 48, 3), dtype=np.uint8)
    label = derive_label_map(img, img.copy())
    # No real difference → nothing should be confidently labelled tattoo.
    assert (label == FG).sum() == 0


def test_derive_label_map_shape_assert():
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.zeros((10, 12, 3), np.uint8)
    try:
        derive_label_map(a, b)
        assert False, "expected an assertion on mismatched dimensions"
    except AssertionError as e:
        assert "match" in str(e)
