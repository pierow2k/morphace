"""Tests for shared dlib landmark detection helpers."""

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from morphace import landmarks

if TYPE_CHECKING:
    import dlib

EXPECTED_FACE_COUNT = 2


class _Point:
    """Fake dlib point."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class _Shape:
    """Fake dlib full object detection."""

    def __init__(self, points: list[_Point]) -> None:
        self._points = points

    def parts(self) -> list[_Point]:
        """Return fake landmark points."""
        return self._points


class _Predictor:
    """Fake dlib shape predictor."""

    def __init__(self) -> None:
        self.seen_rects: list[object] = []

    def __call__(self, image: Any, rect: object) -> _Shape:
        """Return a shape whose coordinates depend on the rectangle."""
        del image
        self.seen_rects.append(rect)
        index = len(self.seen_rects)
        return _Shape([_Point(index, index + 1), _Point(index + 2, index + 3)])


def _empty_detector(image: Any, upsample_num_times: int) -> list[object]:
    """Return no face detections."""
    del image, upsample_num_times
    return []


def _multi_face_detector(image: Any, upsample_num_times: int) -> list[object]:
    """Return two fake face detections."""
    del image, upsample_num_times
    return [object(), object()]


def test_detect_all_landmarks_raises_no_face_error() -> None:
    """Verify no-face detection raises the shared public exception."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(landmarks.NoFaceFoundError):
        next(
            landmarks.detect_all_landmarks(
                image,
                _empty_detector,
                cast("dlib.shape_predictor", _Predictor()),
            )
        )


def test_detect_all_landmarks_yields_each_face_coordinates() -> None:
    """Verify all detections are converted into coordinate lists."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    predictor = _Predictor()

    detected_landmarks = list(
        landmarks.detect_all_landmarks(
            image,
            _multi_face_detector,
            cast("dlib.shape_predictor", predictor),
        )
    )

    assert detected_landmarks == [[(1, 2), (3, 4)], [(2, 3), (4, 5)]]
    assert len(predictor.seen_rects) == EXPECTED_FACE_COUNT
