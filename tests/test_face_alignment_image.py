"""Tests for face alignment image helpers."""

import numpy as np
import PIL.Image
import pytest

from morphace.face_alignment_image import (
    prepare_alignment_canvas,
    warp_aligned_face,
)
from morphace.prep_face_alignment import FaceAlignmentOptions


def test_prepare_alignment_canvas_noops_for_full_image_quad() -> None:
    """Verify canvas preparation keeps an already-contained image unchanged."""
    image = PIL.Image.new("RGB", (64, 64), color=(10, 20, 30))
    quad = np.array([[5.0, 5.0], [5.0, 59.0], [59.0, 59.0], [59.0, 5.0]])

    prepared_image, prepared_quad = prepare_alignment_canvas(
        image,
        quad,
        54.0,
        FaceAlignmentOptions(output_size=64, transform_size=64),
    )

    assert prepared_image.size == image.size
    np.testing.assert_array_equal(prepared_quad, quad)


def test_prepare_alignment_canvas_respects_padding_disabled() -> None:
    """Verify out-of-bounds quads are not padded when padding is disabled."""
    image = PIL.Image.new("RGB", (64, 64), color=(10, 20, 30))
    quad = np.array(
        [[-10.0, -10.0], [-10.0, 40.0], [40.0, 40.0], [40.0, -10.0]]
    )

    prepared_image, prepared_quad = prepare_alignment_canvas(
        image,
        quad,
        50.0,
        FaceAlignmentOptions(
            output_size=64,
            transform_size=64,
            enable_padding=False,
        ),
    )

    assert prepared_image.size == (45, 45)
    np.testing.assert_array_equal(prepared_quad, quad)


def test_warp_aligned_face_rejects_invalid_quad_shape() -> None:
    """Verify warp validates the alignment quad shape."""
    image = PIL.Image.new("RGB", (64, 64), color=(10, 20, 30))
    invalid_quad = np.array([[0.0, 0.0], [10.0, 10.0]])

    with pytest.raises(ValueError, match="Invalid quad shape"):
        warp_aligned_face(
            image,
            invalid_quad,
            FaceAlignmentOptions(output_size=64, transform_size=64),
        )
