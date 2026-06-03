"""Geometry helpers for deriving face-alignment crop coordinates.

The functions in this module translate 68-point facial landmarks into the
quadrilateral used by Pillow's perspective transform. They keep the landmark
indexing, eye-mouth orientation math, and crop-size heuristics together so the
alignment model can be understood in one place.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from morphace._typing import FloatPoint, Point

from .options import FaceAlignmentOptions

# A 4x2 array representing the corners of the alignment quadrilateral in the
# order expected by Pillow's perspective transform: top-left, bottom-left,
# bottom-right, top-right.
type AlignmentQuad = NDArray[Any]


# Constants for the standard 68-point facial landmark model (dlib/IBUG format).
# Indices use anatomical convention: "left" means the person's left side.
_LM_LEFT_EYE = slice(36, 42)  # Points 36-41: left eye landmarks.
_LM_RIGHT_EYE = slice(42, 48)  # Points 42-47: right eye landmarks.
_LM_MOUTH_OUTER = slice(48, 60)  # Points 48-59: outer lip contour.
_EXPECTED_LANDMARK_COUNT = 68
_FLOAT_TOLERANCE = 1e-7


@dataclass(frozen=True)
class _FaceFeatureGeometry:
    """Measured face geometry used to orient the alignment crop."""

    eyes_midpoint: np.ndarray
    eye_line_vec: np.ndarray
    face_axis_vec: np.ndarray


def _rotate_90_ccw(vec: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector 90 degrees counter-clockwise.

    Args:
        vec: A 2-element array representing a 2D vector.

    Returns:
        The input vector rotated 90 degrees counter-clockwise.
        For input (x, y), returns (-y, x).
    """
    return np.array([-vec[1], vec[0]])


def _landmark_array(
    face_landmarks: Sequence[FloatPoint | Point],
) -> np.ndarray:
    """Return validated landmark coordinates as a NumPy array."""
    landmarks = np.array(face_landmarks)

    if landmarks.shape[0] != _EXPECTED_LANDMARK_COUNT:
        message = (
            f"This function requires the 68-point landmark model. "
            f"Received {landmarks.shape[0]} points."
        )
        raise ValueError(message)

    return landmarks


def _measure_face_features(landmarks: np.ndarray) -> _FaceFeatureGeometry:
    """Measure eye and mouth geometry from 68-point landmarks."""
    lm_eye_left = landmarks[_LM_LEFT_EYE]
    lm_eye_right = landmarks[_LM_RIGHT_EYE]
    lm_mouth_outer = landmarks[_LM_MOUTH_OUTER]

    # Using anatomical naming: "left" is the person's left (viewer's right).
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eyes_midpoint = (eye_left + eye_right) * 0.5
    eye_line_vec = eye_right - eye_left

    # Mouth corners: point 48 is anatomical left, point 54 is anatomical right.
    mouth_left_corner = lm_mouth_outer[0]
    mouth_right_corner = lm_mouth_outer[6]
    mouth_center = (mouth_left_corner + mouth_right_corner) * 0.5

    return _FaceFeatureGeometry(
        eyes_midpoint=eyes_midpoint,
        eye_line_vec=eye_line_vec,
        face_axis_vec=mouth_center - eyes_midpoint,
    )


def calculate_alignment_quad(
    face_landmarks: Sequence[FloatPoint | Point],
    options: FaceAlignmentOptions,
) -> tuple[AlignmentQuad, float]:
    """Calculate the alignment quadrilateral and crop size for a face image.

    This function computes a bounding quadrilateral based on facial landmarks
    to align the face. It stabilizes the roll angle by combining the eye-line
    vector with the facial vertical axis from the eyes toward the mouth.

    Args:
        face_landmarks: Sequence of (x, y) landmark coordinate pairs. Must
            conform to the 68-point landmark model format (dlib/IBUG).
        options: Configuration containing scale factors (x_scale, y_scale,
            em_scale) for the alignment.

    Raises:
        ValueError: If the landmark count is not 68 points.
        ValueError: If the calculated alignment vector has zero magnitude,
            indicating degenerate face geometry.

    Returns:
        A tuple containing:
            - crop_corners: 4x2 array of quad corners in Pillow order
              (top-left, bottom-left, bottom-right, top-right).
            - crop_size: Scalar value representing the square crop dimension.
    """
    features = _measure_face_features(_landmark_array(face_landmarks))

    # Determine the orientation of the crop rectangle. Combining the eye-line
    # vector with a rotated face-axis vector stabilizes the crop against head
    # rotation (roll). The result is a diagonal vector whose direction blends
    # horizontal (eye-line) and vertical (face-axis) information.
    crop_vec_x = features.eye_line_vec - _rotate_90_ccw(features.face_axis_vec)

    # Normalize the vector before applying face-size scaling.
    x_norm = np.hypot(*crop_vec_x)
    if x_norm < _FLOAT_TOLERANCE:
        message = "Degenerate face geometry: alignment vector is zero."
        raise ValueError(message)

    crop_vec_x /= x_norm

    # Scale the vector based on facial dimensions. The crop size is determined
    # by the larger of eye separation (width proxy) or eye-to-mouth distance
    # (height proxy), ensuring the full face stays inside the aligned crop.
    face_width_estimate = np.hypot(*features.eye_line_vec) * 2.0
    face_height_estimate = np.hypot(*features.face_axis_vec) * 1.8
    crop_vec_x *= max(face_width_estimate, face_height_estimate)
    crop_vec_x *= options.x_scale

    # Rotate the X vector to derive the crop height direction.
    crop_vec_y = _rotate_90_ccw(crop_vec_x) * options.y_scale

    # Offset the crop center from the eyes toward the mouth based on em_scale.
    # This controls vertical framing: 0.0 centers on eyes, higher values include
    # more of the lower face.
    crop_center = (
        features.eyes_midpoint + features.face_axis_vec * options.em_scale
    )

    # Pillow expects quad corners in top-left, bottom-left, bottom-right,
    # top-right order. Each corner is computed by adding/subtracting the
    # half-extent vectors from the crop center.
    crop_corners = np.stack(
        [
            crop_center - crop_vec_x - crop_vec_y,  # Top-left.
            crop_center - crop_vec_x + crop_vec_y,  # Bottom-left.
            crop_center + crop_vec_x + crop_vec_y,  # Bottom-right.
            crop_center + crop_vec_x - crop_vec_y,  # Top-right.
        ],
    )

    # The X vector magnitude represents half the square crop width.
    crop_size = float(np.hypot(*crop_vec_x) * 2)

    return crop_corners, crop_size
