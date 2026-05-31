"""Dlib model and landmark detection helpers."""

from __future__ import annotations

import logging
import os
import pathlib
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dlib
from platformdirs import user_data_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._typing import ImageArray, LandmarkList, PathInput

MODEL_FILENAME = "shape_predictor_68_face_landmarks_GTX.dat"
MODEL_ENV_VAR = "MORPHACE_LANDMARK_MODEL"

logger = logging.getLogger(__name__)

__all__ = [
    "LandmarkModelNotFoundError",
    "NoFaceFoundError",
    "default_landmark_model_path",
    "detect_all_landmarks",
    "get_detector",
    "get_landmarks",
    "get_predictor",
    "resolve_landmark_model_path",
]


class LandmarkModelNotFoundError(FileNotFoundError):
    """Raised when the dlib landmark model file cannot be found."""


class NoFaceFoundError(Exception):
    """Raised when there is no face found."""


@lru_cache(maxsize=1)
def get_detector() -> Any:
    """Lazy-load the dlib face detector."""
    return dlib.get_frontal_face_detector()


@lru_cache(maxsize=1)
def _load_predictor(model_path: str) -> dlib.shape_predictor:
    """Load and cache the dlib shape predictor."""
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dlib model file not found at: {path}\n"
            "Pass --landmark-model, set MORPHACE_LANDMARK_MODEL, "
            "or place the model file in the expected location."
        )

    return dlib.shape_predictor(str(path))


def get_predictor(landmark_model_path: Path) -> dlib.shape_predictor:
    """Lazy-load the dlib shape predictor."""
    model_path = landmark_model_path.expanduser().resolve()
    return _load_predictor(str(model_path))


def detect_all_landmarks(
    img: ImageArray,
    detector: Any,
    predictor: dlib.shape_predictor,
) -> Iterator[LandmarkList]:
    """Detect and extract landmarks for every face in an image.

    Args:
        img: Image array to scan for faces.
        detector: Dlib-compatible face detector.
        predictor: Dlib shape predictor used to extract landmarks.

    Yields:
        Landmark coordinate lists for each detected face.

    Raises:
        NoFaceFoundError: If the detector finds no faces.
    """
    detections = detector(img, 1)

    if len(detections) == 0:
        logger.error("Unable to find a face in the image.")
        raise NoFaceFoundError("Unable to find a face in the image.")

    for rect in detections:
        shape = predictor(img, rect)
        yield [(point.x, point.y) for point in shape.parts()]


def get_landmarks(
    image: PathInput,
    detector: Any | None = None,
    predictor: dlib.shape_predictor | None = None,
) -> Iterator[LandmarkList]:
    """Detect 68-point facial landmarks in an image file.

    Args:
        image: Path to the image to scan.
        detector: Pre-loaded dlib detector.
        predictor: Pre-loaded dlib predictor.

    Yields:
        Landmark coordinate lists for each detected face.

    Raises:
        RuntimeError: If either dlib model helper is missing.
    """
    if detector is None or predictor is None:
        raise RuntimeError("Dlib models are not loaded. Cannot process faces.")

    img_path = pathlib.Path(image).expanduser().resolve()
    img = dlib.load_rgb_image(str(img_path))

    yield from detect_all_landmarks(img, detector, predictor)


def default_landmark_model_path() -> Path:
    """Return the default application data directory for the landmark model."""
    return (
        user_data_path(
            appname="morphace",
            appauthor=False,
            ensure_exists=True,
        )
        / MODEL_FILENAME
    )


def _require_file(path: Path, source: str) -> Path:
    """Return path if it exists, otherwise raise a helpful error."""
    if not path.is_file():
        raise LandmarkModelNotFoundError(
            f"{source} does not point to an existing file: {path}"
        )

    return path


def resolve_landmark_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve the dlib landmark model path.

    Resolution order:

    1. Explicit CLI path.
    2. MORPHACE_LANDMARK_MODEL environment variable.
    3. Default application data directory.

    Args:
        model_path: Optional explicit path supplied by the caller.

    Returns:
        Path to an existing landmark model file.

    Raises:
        LandmarkModelNotFoundError: If no usable model file is found.
    """
    if model_path is not None:
        return _require_file(Path(model_path).expanduser(), "--landmark-model")

    env_value = os.environ.get(MODEL_ENV_VAR)
    if env_value:
        return _require_file(Path(env_value).expanduser(), MODEL_ENV_VAR)

    default_path = default_landmark_model_path()
    if default_path.is_file():
        return default_path

    raise LandmarkModelNotFoundError(
        "Could not find the dlib landmark model. "
        f"Pass --landmark-model /path/to/{MODEL_FILENAME}, set "
        f"{MODEL_ENV_VAR}, or place the file at {default_path}."
    )
