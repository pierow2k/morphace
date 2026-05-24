"""Tests for morphing image correspondence helpers."""

from collections.abc import Iterator
from typing import Any, cast  # pylint: disable=unused-import

import numpy as np
import pytest

from morphace.morphing import correspondence
from morphace.morphing.correspondence import align_faces, match_image_sizes


def test_match_image_sizes_downscales_larger_image() -> None:
    """Verify a larger image is resized and cropped to the smaller image."""
    image1 = np.zeros((4, 6, 3), dtype=np.uint8)
    image2 = np.zeros((8, 12, 3), dtype=np.uint8)

    matched1, matched2 = match_image_sizes(image1, image2)

    assert matched1.shape == (4, 6, 3)
    assert matched2.shape == (4, 6, 3)


def test_match_image_sizes_downscales_img1_when_img2_smaller() -> None:
    """Verify img1 is downscaled and cropped when img2 is strictly smaller."""
    image1 = np.zeros((8, 12, 3), dtype=np.uint8)
    image2 = np.zeros((4, 6, 3), dtype=np.uint8)

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


def test_align_faces_requires_loaded_models() -> None:
    """Verify correspondence alignment requires detector and predictor."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Dlib models are not loaded"):
        align_faces(image, image)


def test_align_faces_builds_correspondences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify landmark points and boundary points are combined."""
    image1 = np.zeros((4, 4, 3), dtype=np.uint8)
    image2 = np.ones((4, 4, 3), dtype=np.uint8)
    detector = object()
    predictor = cast("Any", object())
    first_points = [(index, index + 1) for index in range(68)]
    second_points = [(index + 2, index + 5) for index in range(68)]
    landmark_sets = [first_points, second_points]

    def fake_detect_all_landmarks(
        img: np.ndarray,
        input_detector: object,
        input_predictor: object,
    ) -> Iterator[list[tuple[int, int]]]:
        """Return one landmark set for each input image."""
        assert input_detector is detector
        assert input_predictor is predictor
        assert img.shape == (4, 4, 3)
        return iter([landmark_sets.pop(0)])

    monkeypatch.setattr(
        correspondence,
        "detect_all_landmarks",
        fake_detect_all_landmarks,
    )

    result = align_faces(image1, image2, detector, predictor)

    assert result.size == (4, 4)
    assert result.image1 is image1
    assert result.image2 is image2
    assert result.points1[:68] == first_points
    assert result.points2[:68] == second_points
    assert len(result.points1) == 76  # noqa:PLR2004
    assert len(result.points2) == 76  # noqa:PLR2004
    np.testing.assert_array_equal(
        result.average_landmarks[0],
        np.array([1.0, 3.0]),
    )
    np.testing.assert_array_equal(
        result.average_landmarks[-8:],
        np.array(
            [
                (1, 1),
                (3, 1),
                (3, 3),
                (1, 3),
                (1, 1),
                (3, 1),
                (1, 3),
                (1, 1),
            ]
        ),
    )
