"""Detect 68-point facial landmarks in image files using dlib.

This module provides utilities to load images and extract facial feature
coordinates using dlib's shape predictor.

Functions:
    get_landmarks: Generator that yields landmark coordinates for
    detected faces.

Exceptions:
    NoFaceFoundError: Raised when an image contains no detectable faces.
"""

import logging
import pathlib
from collections.abc import Iterator
from typing import Any

import dlib

from ._typing import LandmarkList, PathInput
from .morph_landmark_detection import NoFaceFoundError

logger = logging.getLogger(__name__)


def get_landmarks(
    image: PathInput,
    detector: Any | None = None,
    predictor: dlib.shape_predictor | None = None,
) -> Iterator[LandmarkList]:
    """Detect 68-point facial landmarks with dlib.

    Args:
        image: Path to the image to scan.
        detector: Pre-loaded dlib detector.
        predictor: Pre-loaded dlib predictor

    Yields:
        A list of ``(x, y)`` landmark coordinate pairs for a single
        detected face.
    """
    if detector is None or predictor is None:
        raise RuntimeError("Dlib models are not loaded. Cannot process faces.")

    # Resolve path to ensure it works with relative paths or Path objects.
    img_path = pathlib.Path(image).expanduser().resolve()
    img = dlib.load_rgb_image(str(img_path))

    detections = detector(img, 1)

    if len(detections) == 0:
        logger.error("Unable to find a face in the image.")
        raise NoFaceFoundError("Unable to find a face in the image.")

    for detection in detections:
        shape = predictor(img, detection)
        face_landmarks = [(p.x, p.y) for p in shape.parts()]
        yield face_landmarks
