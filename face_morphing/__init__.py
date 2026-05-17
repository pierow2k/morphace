"""Face morphing package initialization."""

import logging
from dataclasses import dataclass
from pathlib import Path

import dlib

from ._typing import ImageArray
from .delaunay_triangulation import compute_delaunay_triangles
from .face_landmark_detection import align_faces
from .face_morph import generate_morph_sequence

logger = logging.getLogger(__name__)


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


# Load models once at module level.
try:
    # Robust path handling relative to this file
    _MODEL_PATH = (
        Path(__file__).parent
        / "utils"
        / "shape_predictor_68_face_landmarks.dat"
    )
    if not _MODEL_PATH.exists():
        # Fallback if running from different directory structure
        _MODEL_PATH = (
            Path("face_morphing")
            / "utils"
            / "shape_predictor_68_face_landmarks.dat"
        )

    _DETECTOR = dlib.get_frontal_face_detector()
    _PREDICTOR = dlib.shape_predictor(str(_MODEL_PATH))
except RuntimeError as e:
    logger = logging.getLogger(__name__)
    logger.warning("Could not load dlib models: %s", e)
    _DETECTOR = None
    _PREDICTOR = None


def do_morphing(
    img1: ImageArray,
    img2: ImageArray,
    config: MorphConfig,
    show_triangles: bool = False,
) -> None:
    """Perform face morphing between two images.

    Args:
        img1: The first input image.
        img2: The second input image.
        config: Configuration for the morph output.
        show_triangles: Whether to show triangulation lines.
    """
    # Detect facial landmarks and create correspondence between images.
    size, img1, img2, points1, points2, list3 = align_faces(
        img1, img2, _DETECTOR, _PREDICTOR
    )

    # Create a Delaunay triangulation from a provided list of points.
    tri = compute_delaunay_triangles(size[1], size[0], list3)

    # Generate a face morphing sequence and save it as a video.
    generate_morph_sequence(
        (img1, img2),
        (points1, points2),
        tri,
        (config.duration, config.frame_rate, size, config.output),
        show_triangles,
    )


# Expose the public API .
__all__ = ["MorphConfig", "do_morphing"]
