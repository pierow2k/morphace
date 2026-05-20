"""Align detected faces into canonical square crops using facial landmarks.

This module provides the high-level face-alignment pipeline. It derives a
transformation quadrilateral from 68-point facial landmarks, prepares the
source image so that crop can be sampled cleanly, then warps the result to a
standard square output image.

Geometry and raster operations live in focused helper modules, but this module
keeps the end-to-end flow visible for callers that need to align and save a
single detected face.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import PIL.Image

from ._typing import FloatPoint, PathInput, Point
from .prep_alignment_geometry import calculate_alignment_quad
from .prep_alignment_image import prepare_alignment_canvas, warp_aligned_face

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceAlignmentOptions:
    """Configuration for aligning a detected face.

    Attributes:
        output_size: Final square image dimension in pixels.
        transform_size: Intermediate square transform dimension in pixels.
        enable_padding: Whether to synthesize reflected image padding.
        x_scale: Horizontal scale factor for the aligned crop.
        y_scale: Vertical scale factor for the aligned crop.
        em_scale: Offset factor from the eyes toward the mouth.
        alpha: Whether to include an alpha mask for padded regions.
    """

    output_size: int = 1024
    transform_size: int = 4096
    enable_padding: bool = True
    x_scale: float = 1.0
    y_scale: float = 1.0
    em_scale: float = 0.1
    alpha: bool = False


def align_face_image(
    src_file: PathInput,
    dst_file: PathInput,
    face_landmarks: Sequence[FloatPoint | Point],
    options: FaceAlignmentOptions | None = None,
) -> None:
    """Align and save a face crop from a source image.

    The pipeline is intentionally small at this level: compute the face crop
    geometry, open the image, prepare its canvas for the requested crop, warp
    the face into the canonical square, and save it as PNG.

    Args:
        src_file: Path to the source image.
        dst_file: Path where the aligned PNG should be written.
        face_landmarks: Sequence of 68 ``(x, y)`` facial landmark points.
        options: Optional alignment configuration.
    """
    options = options or FaceAlignmentOptions()

    if not Path(src_file).is_file():
        logger.error("Cannot find source image.")
        return

    quad, crop_size = calculate_alignment_quad(face_landmarks, options)
    image = PIL.Image.open(src_file).convert("RGBA").convert("RGB")
    image, quad = prepare_alignment_canvas(image, quad, crop_size, options)
    image = warp_aligned_face(image, quad, options)
    image.save(dst_file, "PNG")
