# workflow.py

"""High-level face morphing workflow."""

from pathlib import Path

from ._typing import ImageArray
from .config import MorphConfig
from .delaunay_triangulation import compute_delaunay_triangles
from .face_landmark_detection import align_faces
from .face_morph import generate_morph_sequence
from .models import get_detector, get_predictor


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
    """
    detector = get_detector()
    predictor = get_predictor(config.landmark_model_path)

    size, img1, img2, points1, points2, avg_landmarks = align_faces(
        img1,
        img2,
        detector,
        predictor,
    )

    triangles = compute_delaunay_triangles(size[1], size[0], avg_landmarks)

    generate_morph_sequence(
        (img1, img2),
        (points1, points2),
        triangles,
        (config.duration, config.frame_rate, size, config.output),
        show_triangles,
    )

    return Path(config.output)
