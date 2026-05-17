"""Tests for face landmark preprocessing helpers."""

import numpy as np

from face_morphing.face_landmark_detection import calculate_margin_help


def test_calculate_margin_help_returns_shape_offsets() -> None:
    """Calculate shape differences and averages for two images."""
    img1 = np.zeros((10, 20, 3), dtype=np.uint8)
    img2 = np.zeros((14, 12, 3), dtype=np.uint8)

    result = calculate_margin_help(img1, img2)

    assert result == ((10, 20, 3), (14, 12, 3), 2, 4, 12, 16)
