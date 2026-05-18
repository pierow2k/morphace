"""Tests for the public no-face exception."""

from typing import Any, cast

import numpy as np
import pytest

import face_morphing
from face_morphing import face_landmark_detection, prep_landmarks


def _empty_detector(image: Any, upsample_num_times: int) -> list[Any]:
    """Return no face detections."""
    del image, upsample_num_times
    return []


def test_public_no_face_found_error_is_landmark_error() -> None:
    """Verify the package root re-exports the landmark exception."""
    assert (
        face_morphing.NoFaceFoundError
        is face_landmark_detection.NoFaceFoundError
    )


def test_prep_no_face_found_error_is_landmark_error() -> None:
    """Verify prep landmark detection uses the public no-face exception."""
    assert prep_landmarks.NoFaceFoundError is face_morphing.NoFaceFoundError


def test_align_faces_raises_public_no_face_found_error() -> None:
    """Verify align_faces raises the public no-face exception."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(face_morphing.NoFaceFoundError):
        face_landmark_detection.align_faces(
            image,
            image,
            _empty_detector,
            cast("Any", object()),
        )
