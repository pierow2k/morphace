"""High-level face morphing workflow.

Orchestrates the complete face morphing pipeline by coordinating
face alignment, Delaunay triangulation, and sequence generation.
"""

from pathlib import Path

from morphace._typing import ImageArray
from morphace.landmarks import get_detector, get_predictor

from .config import MorphConfig, MorphVideoConfig
from .correspondence import align_faces
from .frames import generate_morph_sequence
from .triangulation import compute_delaunay_triangles


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

    correspondences = align_faces(
        img1,
        img2,
        detector,
        predictor,
    )

    triangles = compute_delaunay_triangles(
        correspondences.size[1],
        correspondences.size[0],
        correspondences.average_landmarks,
    )

    generate_morph_sequence(
        (correspondences.image1, correspondences.image2),
        (correspondences.points1, correspondences.points2),
        triangles,
        MorphVideoConfig(
            duration=config.duration,
            frame_rate=config.frame_rate,
            size=correspondences.size,
            output=config.output,
        ),
        show_triangles,
    )

    return Path(config.output)
