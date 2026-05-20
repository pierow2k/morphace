"""Helpers for resolving and loading face landmark models."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import dlib
from platformdirs import user_data_path

MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
MODEL_ENV_VAR = "MORPHACE_LANDMARK_MODEL"


class LandmarkModelNotFoundError(FileNotFoundError):
    """Raised when the dlib landmark model file cannot be found."""


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


def default_landmark_model_path() -> Path:
    """Return the default per-user location for the landmark model."""
    return (
        user_data_path(
            appname="morphace",
            appauthor=False,
            ensure_exists=True,
        )
        / MODEL_FILENAME
    )


def resolve_landmark_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve the dlib landmark model path.

    Resolution order:

    1. Explicit CLI path.
    2. MORPHACE_LANDMARK_MODEL environment variable.
    3. Default per-user app data directory.

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


def _require_file(path: Path, source: str) -> Path:
    """Return path if it exists, otherwise raise a helpful error."""
    if not path.is_file():
        raise LandmarkModelNotFoundError(
            f"{source} does not point to an existing file: {path}"
        )

    return path
