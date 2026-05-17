"""Face morphing package initialization."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import dlib

from ._typing import Any, ImageArray
from .delaunay_triangulation import compute_delaunay_triangles
from .face_landmark_detection import NoFaceFoundError, align_faces
from .face_morph import generate_morph_sequence

logger = logging.getLogger(__name__)


# Define the path relative to this file
_MODEL_PATH = (
    Path(__file__).parent / "models" / "shape_predictor_68_face_landmarks.dat"
)


@lru_cache(maxsize=1)
def _get_detector() -> Any:
    """Lazy-loads the dlib face detector."""
    return dlib.get_frontal_face_detector()


@lru_cache(maxsize=1)
def _get_predictor() -> dlib.shape_predictor:
    """Lazy-loads the dlib shape predictor."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Dlib model file not found at: {_MODEL_PATH}\n"
            "Please ensure the model file is included in the package."
        )
    return dlib.shape_predictor(str(_MODEL_PATH))


@dataclass
class MorphConfig:
    """Configuration for morph video output.

    Args:
        duration: Duration of the morphing sequence in seconds.
        frame_rate: Number of frames per second.
        output: Path to save the output video.
    """

    duration: int
    frame_rate: int
    output: str


def morph_faces(
    img1: ImageArray,
    img2: ImageArray,
    config: MorphConfig,
    show_triangles: bool = False,
) -> Path:
    """Perform face morphing between two images.

    Args:
        img1: The first input image.
        img2: The second input image.
        config: Configuration for the morph output.
        show_triangles: Whether to show triangulation lines.

    Returns:
        The path to the generated video file.

    Raises:
        RuntimeError: If dlib models are not loaded.
        NoFaceFoundError: If a face cannot be detected in input images.
    """
    detector = _get_detector()
    predictor = _get_predictor()

    # Detect facial landmarks and create correspondence between images.
    size, img1, img2, points1, points2, avg_landmarks = align_faces(
        img1, img2, detector, predictor
    )

    # Create a Delaunay triangulation from average landmark points.
    tri = compute_delaunay_triangles(size[1], size[0], avg_landmarks)

    # Generate a face morphing sequence and save it as a video.
    generate_morph_sequence(
        (img1, img2),
        (points1, points2),
        tri,
        (config.duration, config.frame_rate, size, config.output),
        show_triangles,
    )

    return Path(config.output)


# Expose the public API .
__all__ = ["MorphConfig", "NoFaceFoundError", "morph_faces"]
