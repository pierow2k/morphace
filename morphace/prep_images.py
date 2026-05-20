"""Align faces and producing cropped images from a directory of raw images.

Detects each face in a given image, defers to `align_face_image` to align and
crop the face, and saves the result in the output directory with the face
number appended to the filename.
"""

# pylint: disable=broad-exception-caught

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .models import get_detector, get_predictor
from .prep_face_alignment import FaceAlignmentOptions, align_face_image
from .prep_landmarks import get_landmarks

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
                    align_face_image(
                        raw_img_path,
                        aligned_face_path,
                        face_landmarks,
                        config.face_alignment,
                    )
                    logger.info("Wrote result %s", aligned_face_path)
                except Exception:
                    logger.exception("Exception in face alignment.")
        except Exception:
            logger.exception("Exception in landmark detection.")
