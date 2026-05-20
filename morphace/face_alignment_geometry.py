"""Geometry helpers for deriving face-alignment crop coordinates.

The functions in this module translate 68-point facial landmarks into the
quadrilateral used by Pillow's perspective transform. They keep the landmark
indexing, eye-mouth orientation math, and crop-size heuristics together so the
alignment model can be understood in one place.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ._typing import FloatPoint, Point

type AlignmentQuad = NDArray[Any]


class GeometryOptions(Protocol):
    """Option fields used by face-alignment geometry."""

    @property
    def x_scale(self) -> float:
        """Horizontal scale factor for the aligned crop."""
        raise NotImplementedError

    @property
    def y_scale(self) -> float:
        """Vertical scale factor for the aligned crop."""
        raise NotImplementedError

    @property
    def em_scale(self) -> float:
        """Offset factor from the eyes toward the mouth."""
        raise NotImplementedError


# Constants for the standard 68-point facial landmark model.
_LM_LEFT_EYE = slice(36, 42)
_LM_RIGHT_EYE = slice(42, 48)
_LM_MOUTH_OUTER = slice(48, 60)
_EXPECTED_LANDMARK_COUNT = 68
_FLOAT_TOLERANCE = 1e-7


def _rotate_90_ccw(vec: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector 90 degrees counter-clockwise."""
    return np.array([-vec[1], vec[0]])


def calculate_alignment_quad(
    face_landmarks: Sequence[FloatPoint | Point],
    options: GeometryOptions,
) -> tuple[AlignmentQuad, float]:
    """Calculate the alignment quadrilateral and crop size for a face image.

    This function computes a bounding quadrilateral based on facial landmarks
    to align the face. It stabilizes the roll angle by combining the eye-line
    vector with the facial vertical axis from the eyes toward the mouth.

    Args:
        face_landmarks: Sequence of (x, y) landmark coordinate pairs. Must
            conform to the 68-point landmark model format.
        options: Configuration containing scale factors (x_scale, y_scale,
            em_scale) for the alignment.

    Raises:
        ValueError: If the landmark count is not 68 points.
        ValueError: If the calculated alignment vector has zero magnitude,
            indicating degenerate face geometry.

    Returns:
        A tuple containing the crop quadrilateral and scalar square crop size.
    """
    landmarks = np.array(face_landmarks)

    if landmarks.shape[0] != _EXPECTED_LANDMARK_COUNT:
        raise ValueError(
            f"This function requires the 68-point landmark model. "
            f"Received {landmarks.shape[0]} points."
        )

    # Extract landmark groups using standard 68-point model indices.
    lm_eye_left = landmarks[_LM_LEFT_EYE]
    lm_eye_right = landmarks[_LM_RIGHT_EYE]
    lm_mouth_outer = landmarks[_LM_MOUTH_OUTER]

    # Calculate geometric centers of facial features.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eyes_midpoint = (eye_left + eye_right) * 0.5
    eye_line_vec = eye_right - eye_left

    left_mouth_corner = lm_mouth_outer[0]
    mouth_right_corner = lm_mouth_outer[6]
    mouth_avg = (left_mouth_corner + mouth_right_corner) * 0.5
    face_axis_vec = mouth_avg - eyes_midpoint

    # Determine the orientation of the crop rectangle. Combining the eye-line
    # vector with a rotated face-axis vector stabilizes the crop against head
    # rotation (roll).
    crop_vec_x = eye_line_vec - _rotate_90_ccw(face_axis_vec)

    # Normalize the vector before applying face-size scaling.
    x_norm = np.hypot(*crop_vec_x)
    if x_norm < _FLOAT_TOLERANCE:
        raise ValueError("Degenerate face geometry: alignment vector is zero.")

    crop_vec_x /= x_norm

    # Scale the vector based on facial dimensions. The crop size is determined
    # by the larger of eye width or eye-to-mouth height so the full face stays
    # inside the aligned crop.
    crop_vec_x *= max(
        np.hypot(*eye_line_vec) * 2.0,
        np.hypot(*face_axis_vec) * 1.8,
    )
    crop_vec_x *= options.x_scale

    # Rotate the X vector to derive the crop height direction.
    crop_vec_y = _rotate_90_ccw(crop_vec_x) * options.y_scale

    # Offset the crop center from the eyes toward the mouth based on em_scale.
    crop_center = eyes_midpoint + face_axis_vec * options.em_scale

    # Pillow expects quad corners in top-left, bottom-left, bottom-right,
    # top-right order.
    crop_corners = np.stack(
        [
            crop_center - crop_vec_x - crop_vec_y,  # Top-left.
            crop_center - crop_vec_x + crop_vec_y,  # Bottom-left.
            crop_center + crop_vec_x + crop_vec_y,  # Bottom-right.
            crop_center + crop_vec_x - crop_vec_y,  # Top-right.
        ]
    )

    # The X vector represents half the square crop width.
    crop_size = float(np.hypot(*crop_vec_x) * 2)

    return crop_corners, crop_size
