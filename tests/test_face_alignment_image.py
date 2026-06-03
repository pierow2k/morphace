"""Tests for face alignment image helpers."""

import numpy as np
import PIL.Image
import pytest

from morphace.alignment import FaceAlignmentOptions
from morphace.alignment.image import (
    _crop_image,
    _shrink_image,
    prepare_alignment_canvas,
    warp_aligned_face,
)


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
        [[-10.0, -10.0], [-10.0, 40.0], [40.0, 40.0], [40.0, -10.0]],
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


@pytest.mark.parametrize(("alpha", "mode"), [(False, "RGB"), (True, "RGBA")])
def test_prepare_alignment_canvas_pads_with_requested_alpha_mode(
    alpha: bool,
    mode: str,
) -> None:
    """Verify padded canvases use RGB or RGBA output as requested."""
    image = PIL.Image.new("RGB", (64, 64), color=(10, 20, 30))
    quad = np.array(
        [[-10.0, -10.0], [-10.0, 40.0], [40.0, 40.0], [40.0, -10.0]],
    )
    original_quad = quad.copy()

    prepared_image, prepared_quad = prepare_alignment_canvas(
        image,
        quad,
        50.0,
        FaceAlignmentOptions(
            output_size=64,
            transform_size=64,
            alpha=alpha,
        ),
    )

    assert prepared_image.mode == mode
    assert prepared_image.size == (75, 75)
    np.testing.assert_array_equal(
        prepared_quad,
        original_quad + np.array([15, 15]),
    )


def test_shrink_image_downscales_large_crop() -> None:
    """Verify oversized crops shrink the image and quad together."""
    image = PIL.Image.new("RGB", (20, 12), color=(10, 20, 30))
    quad = np.array([[0.0, 0.0], [0.0, 8.0], [16.0, 8.0], [16.0, 0.0]])

    shrunk_image, shrunk_quad, shrunk_crop_size = _shrink_image(
        image,
        quad,
        400.0,
        FaceAlignmentOptions(output_size=50, transform_size=50),
    )

    assert shrunk_image.size == (5, 3)
    np.testing.assert_array_equal(shrunk_quad, quad / 4)
    assert shrunk_crop_size == 100.0  # noqa: PLR2004


def test_crop_image_returns_original_for_outside_quad() -> None:
    """Verify impossible crop boxes leave the image and quad unchanged."""
    image = PIL.Image.new("RGB", (10, 10), color=(10, 20, 30))
    quad = np.array([[20.0, 20.0], [20.0, 25.0], [25.0, 25.0], [25.0, 20.0]])

    cropped_image, cropped_quad = _crop_image(image, quad, border=0)

    assert cropped_image is image
    np.testing.assert_array_equal(cropped_quad, quad)


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


def test_warp_aligned_face_downscales_transform_canvas() -> None:
    """Verify warp resizes the transform canvas down to output size."""
    image = PIL.Image.new("RGB", (8, 8), color=(10, 20, 30))
    quad = np.array([[0.0, 0.0], [0.0, 7.0], [7.0, 7.0], [7.0, 0.0]])

    warped = warp_aligned_face(
        image,
        quad,
        FaceAlignmentOptions(output_size=4, transform_size=8),
    )

    assert warped.size == (4, 4)


def test_warp_aligned_face_keeps_matching_transform_size() -> None:
    """Verify warp skips resizing when transform and output sizes match."""
    image = PIL.Image.new("RGB", (8, 8), color=(10, 20, 30))
    quad = np.array([[0.0, 0.0], [0.0, 7.0], [7.0, 7.0], [7.0, 0.0]])

    warped = warp_aligned_face(
        image,
        quad,
        FaceAlignmentOptions(output_size=8, transform_size=8),
    )

    assert warped.size == (8, 8)


def test_warp_aligned_face_upscales_transform_canvas() -> None:
    """Verify warp handles output sizes larger than transform size."""
    image = PIL.Image.new("RGB", (8, 8), color=(10, 20, 30))
    quad = np.array([[0.0, 0.0], [0.0, 7.0], [7.0, 7.0], [7.0, 0.0]])

    warped = warp_aligned_face(
        image,
        quad,
        FaceAlignmentOptions(output_size=8, transform_size=4),
    )

    assert warped.size == (8, 8)
