"""Align faces and producing cropped images from a directory of raw images.

Detects each face in a given image, defers to `align_face_image` to align and
crop the face, and saves the result in the output directory with the face
number appended to the filename.
"""

# pylint: disable=broad-exception-caught

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from morphace.landmarks import get_detector, get_landmarks, get_predictor

from .face import align_face_image
from .options import FaceAlignmentOptions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for the image alignment pipeline.

    Attributes:
        raw_dir: Directory containing source images.
        aligned_dir: Directory where aligned images are written.
        landmark_model_path: Path to the dlib landmark predictor model.
        face_alignment: Options used when aligning each detected face.
    """

    raw_dir: Path
    aligned_dir: Path
    landmark_model_path: Path
    face_alignment: FaceAlignmentOptions = field(
        default_factory=FaceAlignmentOptions
    )


def _aligned_face_path(
    config: AlignmentConfig,
    raw_img_path: Path,
    index: int,
) -> Path:
    """Return the output path for one aligned face."""
    return config.aligned_dir / f"{raw_img_path.stem}_face{index:02d}.png"


def _should_skip_existing(
    config: AlignmentConfig,
    raw_img_path: Path,
    overwrite: bool,
) -> bool:
    """Return whether an image should be skipped because output exists."""
    first_face_path = _aligned_face_path(config, raw_img_path, 1)
    if first_face_path.is_file() and not overwrite:
        logger.info("skipping existing file %s", first_face_path.name)
        return True

    return False


def _align_detected_faces(
    config: AlignmentConfig,
    raw_img_path: Path,
    detector: Any,
    predictor: Any,
) -> None:
    """Align every detected face for one source image."""
    logger.info("Getting landmarks...")
    for index, face_landmarks in enumerate(
        get_landmarks(
            image=raw_img_path,
            detector=detector,
            predictor=predictor,
        ),
        start=1,
    ):
        try:
            logger.info("Starting face alignment...")
            aligned_face_path = _aligned_face_path(config, raw_img_path, index)
            align_face_image(
                raw_img_path,
                aligned_face_path,
                face_landmarks,
                config.face_alignment,
            )
            logger.info("Wrote result %s", aligned_face_path)
        except Exception:
            logger.exception("Exception in face alignment.")


def align_faces(config: AlignmentConfig, overwrite: bool = False) -> None:
    """Produce aligned face crops from raw images.

    Args:
        config: Pipeline configuration options.
        overwrite: Whether to overwrite existing aligned images.
    """
    detector = get_detector()
    predictor = get_predictor(config.landmark_model_path)

    for raw_img_path in config.raw_dir.iterdir():
        logger.info("Aligning %s ...", raw_img_path.name)
        try:
            if _should_skip_existing(config, raw_img_path, overwrite):
                continue
            _align_detected_faces(config, raw_img_path, detector, predictor)
        except Exception:
            logger.exception("Exception in landmark detection.")
