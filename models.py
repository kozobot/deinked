"""Cached heavy-model singletons, so weights load once and stay resident across node runs.

ComfyUI keeps the Python process alive between prompt executions, so an ``Inpainter`` /
``TattooMaskSegmenter`` built on the first run is reused on every subsequent run — matching the
`functools.lru_cache` singletons the marimo app uses. Everything is imported lazily inside the
getters so importing this module (at node-registration time) never pulls torch/transformers.
"""

from __future__ import annotations

_INPAINTER = None
_SEGMENTERS: dict[str, object] = {}
_BOX_SEGMENTER = None


def get_segmenter():
    """Shared ``deink.segment.TattooSegmenter`` (box detection + SAM) for the box localizer path."""
    global _BOX_SEGMENTER
    if _BOX_SEGMENTER is None:
        from deink.segment import TattooSegmenter

        _BOX_SEGMENTER = TattooSegmenter()
    return _BOX_SEGMENTER


def get_inpainter():
    """Shared ``deink.inpaint.Inpainter`` (LaMa + SDXL load lazily on first use)."""
    global _INPAINTER
    if _INPAINTER is None:
        from deink.inpaint import Inpainter

        _INPAINTER = Inpainter()
    return _INPAINTER


def get_mask_segmenter(checkpoint_dir: str | None, threshold: float):
    """Shared ``deink.tattooseg.TattooMaskSegmenter`` keyed by checkpoint dir.

    ``threshold`` is passed per-call to ``segment`` by the node, so it does not key the cache;
    a single instance per checkpoint keeps the SegFormer weights resident.
    """
    from deink.tattooseg import TattooMaskSegmenter, resolve_checkpoint_dir

    key = str(resolve_checkpoint_dir(checkpoint_dir or None))
    seg = _SEGMENTERS.get(key)
    if seg is None:
        seg = TattooMaskSegmenter(checkpoint_dir=checkpoint_dir or None, threshold=threshold)
        _SEGMENTERS[key] = seg
    return seg
