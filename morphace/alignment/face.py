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
from pathlib import Path

import PIL.Image

from morphace._typing import FloatPoint, PathInput, Point

from .geometry import calculate_alignment_quad
from .image import prepare_alignment_canvas, warp_aligned_face
from .options import FaceAlignmentOptions

logger = logging.getLogger(__name__)


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
