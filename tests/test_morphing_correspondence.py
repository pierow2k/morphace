"""Tests for morphing image correspondence helpers."""

import numpy as np

from morphace.morphing.correspondence import match_image_sizes


def test_match_image_sizes_downscales_larger_image() -> None:
    """Verify a larger image is resized and cropped to the smaller image."""
    image1 = np.zeros((4, 6, 3), dtype=np.uint8)
    image2 = np.zeros((8, 12, 3), dtype=np.uint8)

    matched1, matched2 = match_image_sizes(image1, image2)

    assert matched1.shape == (4, 6, 3)
    assert matched2.shape == (4, 6, 3)


def test_match_image_sizes_crops_mixed_dimensions() -> None:
    """Verify mixed dimensions are center-cropped to shared minimums."""
    image1 = np.zeros((4, 10, 3), dtype=np.uint8)
    image2 = np.zeros((8, 6, 3), dtype=np.uint8)

    matched1, matched2 = match_image_sizes(image1, image2)

    assert matched1.shape == (4, 6, 3)
    assert matched2.shape == (4, 6, 3)
