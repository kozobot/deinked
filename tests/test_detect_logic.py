"""GPU-free unit tests for the detector-abstraction pure logic in ``deink.segment``.

These cover the seams introduced by the open-vocab detector upgrade (family classification,
OWLv2 prompt normalization, NMS index return, and the ensemble union dedup) without loading
any model — they run on CPU in CI. The model-backed detection paths are exercised on-GPU by
``scripts/smoke_test.py`` / ``scripts/sweep_detect.py``.
"""

import numpy as np

from deink.segment import _detector_family, _nms, _owlv2_queries


def test_detector_family_gdino():
    assert _detector_family("IDEA-Research/grounding-dino-base") == "gdino"
    assert _detector_family("IDEA-Research/grounding-dino-tiny") == "gdino"
    # Unknown ids default to the GroundingDINO path.
    assert _detector_family("some/other-detector") == "gdino"


def test_detector_family_owlv2():
    assert _detector_family("google/owlv2-base-patch16-ensemble") == "owlv2"
    assert _detector_family("google/owlv2-large-patch14-ensemble") == "owlv2"
    assert _detector_family("google/owlvit-base-patch32") == "owlv2"


def test_owlv2_queries_simple():
    # The default GroundingDINO-style prompt collapses to a single OWLv2 query, no period.
    assert _owlv2_queries("a tattoo.") == ["a tattoo"]
    assert _owlv2_queries("a tattoo") == ["a tattoo"]


def test_owlv2_queries_compound():
    # A compound "." -separated prompt becomes multiple OWLv2 queries.
    assert _owlv2_queries("tattoo. ink drawing on skin.") == ["tattoo", "ink drawing on skin"]


def test_owlv2_queries_empty_falls_back():
    assert _owlv2_queries("") == ["a tattoo"]
    assert _owlv2_queries("   .  ") == ["a tattoo"]


def test_nms_return_idx_matches_boxes():
    # Boxes 0 and 1 overlap heavily; box 2 is disjoint. Highest score wins each cluster.
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    kept_boxes, kept_idx = _nms(boxes, scores, return_idx=True)
    assert kept_idx.tolist() == [0, 2]
    # Returned boxes correspond exactly to the kept indices.
    assert np.array_equal(kept_boxes, boxes[kept_idx])
    # Scores can be sliced by the returned indices (what the ensemble branch relies on).
    assert scores[kept_idx].tolist() == [0.9, 0.7]


def test_nms_return_idx_empty():
    boxes = np.empty((0, 4))
    scores = np.empty((0,))
    kept_boxes, kept_idx = _nms(boxes, scores, return_idx=True)
    assert kept_boxes.shape == (0, 4)
    assert kept_idx.shape == (0,)


def test_nms_default_return_unchanged():
    # Without return_idx the legacy call returns just boxes (tiled path depends on this).
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
    scores = np.array([0.9, 0.8])
    out = _nms(boxes, scores)
    assert isinstance(out, np.ndarray) and out.shape == (1, 4)


def test_ensemble_union_dedup():
    # Simulate the ensemble concat+NMS: two detectors, one overlapping box and two unique.
    gdino_boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    gdino_scores = np.array([0.9, 0.6])
    owlv2_boxes = np.array([[1, 1, 11, 11], [200, 200, 220, 220]], dtype=float)
    owlv2_scores = np.array([0.7, 0.5])

    boxes = np.concatenate([gdino_boxes, owlv2_boxes], axis=0)
    scores = np.concatenate([gdino_scores, owlv2_scores], axis=0)
    merged, kept = _nms(boxes, scores, return_idx=True)

    # The overlapping pair collapses to one box → 3 survive (gdino[0], gdino[1], owlv2[1]).
    assert len(merged) == 3
    # gdino[0] (0.9) beats owlv2[0] (0.7) in the overlapping cluster; owlv2[0] is suppressed.
    assert kept.tolist() == [0, 1, 3]
    assert scores[kept].tolist() == [0.9, 0.6, 0.5]
