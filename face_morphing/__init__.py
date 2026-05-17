"""Face morphing package initialization."""

import logging
from dataclasses import dataclass

from ._typing import ImageArray
from .delaunay_triangulation import make_delaunay
from .face_landmark_detection import generate_face_correspondences
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
    size, img1, img2, points1, points2, list3 = generate_face_correspondences(
        img1, img2
    )

    tri = make_delaunay(size[1], size[0], list3)

    generate_morph_sequence(
        (img1, img2),
        (points1, points2),
        tri,
        (config.duration, config.frame_rate, size, config.output),
        show_triangles,
    )


# Expose the public API cleanly
__all__ = ["MorphConfig", "do_morphing"]
