"""Align faces and producing cropped images from a directory of raw images.

Detects each face in a given image, defers to `image_align` to align and
crop the face, and saves the result in the output directory with the face
number appended to the filename.
"""

# pylint: disable=broad-exception-caught

import logging
from dataclasses import dataclass
from pathlib import Path

from .models import get_detector, get_predictor
from .prep_face_alignment import AlignmentOptions, image_align
from .prep_landmarks import get_landmarks

logger = logging.getLogger(__name__)


@dataclass
class AlignmentConfig:
    """Configuration for the alignment pipeline."""

    raw_dir: Path
    aligned_dir: Path
    landmark_model_path: Path
    output_size: int = 1024
    x_scale: float = 1.0
    y_scale: float = 1.0
    em_scale: float = 0.1
    use_alpha: bool = False


def align_faces(config: AlignmentConfig, overwrite: bool = False) -> None:
    """Produce aligned face crops from raw images.

    Args:
        config: Pipeline configuration options.
        overwrite: Whether to overwrite existing aligned images.
    """
    alignment_options = AlignmentOptions(
        output_size=config.output_size,
        x_scale=config.x_scale,
        y_scale=config.y_scale,
        em_scale=config.em_scale,
        alpha=config.use_alpha,
    )

    detector = get_detector()
    predictor = get_predictor(config.landmark_model_path)

    for raw_img_path in config.raw_dir.iterdir():
        logger.info("Aligning %s ...", raw_img_path.name)
        try:
            first_face_name = f"{raw_img_path.stem}_face01.png"
            if (
                config.aligned_dir / first_face_name
            ).is_file() and not overwrite:
                logger.info("skipping existing file %s", first_face_name)
                continue
            logger.info("Getting landmarks...")
            for i, face_landmarks in enumerate(
                get_landmarks(
                    image=raw_img_path, detector=detector, predictor=predictor
                ),
                start=1,
            ):
                try:
                    logger.info("Starting face alignment...")
                    face_img_name = f"{raw_img_path.stem}_face{i:02d}.png"
                    aligned_face_path = config.aligned_dir / face_img_name
                    image_align(
                        raw_img_path,
                        aligned_face_path,
                        face_landmarks,
                        alignment_options,
                    )
                    logger.info("Wrote result %s", aligned_face_path)
                except Exception:
                    logger.exception("Exception in face alignment.")
        except Exception:
            logger.exception("Exception in landmark detection.")
