"""Tests for face alignment geometry helpers."""

import pytest

from morphace.prep_alignment_geometry import calculate_alignment_quad
from morphace.prep_face_alignment import FaceAlignmentOptions


def _synthetic_landmarks() -> list[tuple[float, float]]:
    """Return simple non-degenerate 68-point face landmarks."""
    landmarks = [(50.0, 50.0) for _ in range(68)]
    for index in range(36, 42):
        landmarks[index] = (40.0 + index - 36, 40.0)
    for index in range(42, 48):
        landmarks[index] = (70.0 + index - 42, 40.0)
    landmarks[48] = (45.0, 80.0)
    landmarks[54] = (75.0, 80.0)
    return landmarks


def test_calculate_alignment_quad_rejects_wrong_landmark_count() -> None:
    """Verify geometry requires the 68-point landmark model."""
    with pytest.raises(ValueError, match="68-point landmark model"):
        calculate_alignment_quad([(1.0, 2.0)], FaceAlignmentOptions())


def test_calculate_alignment_quad_rejects_degenerate_geometry() -> None:
    """Verify geometry rejects landmarks with no usable orientation."""
    landmarks = [(10.0, 10.0) for _ in range(68)]

    with pytest.raises(ValueError, match="Degenerate face geometry"):
        calculate_alignment_quad(landmarks, FaceAlignmentOptions())


def test_calculate_alignment_quad_returns_quad_and_crop_size() -> None:
    """Verify non-degenerate landmarks produce a square crop quad."""
    quad, crop_size = calculate_alignment_quad(
        _synthetic_landmarks(),
        FaceAlignmentOptions(),
    )

    assert quad.shape == (4, 2)
    assert crop_size > 0
