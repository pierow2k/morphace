"""Align faces and produce cropped images from raw image sources.

Detects each face in each source image, delegates to `align_face_image` to
align and crop the face, and saves the result in the output directory with
the face number appended to the filename. The source may be a directory of
images or a single image file.
"""

# pylint: disable=broad-exception-caught

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from morphace.landmarks import get_detector, get_landmarks, get_predictor

from .face import align_face_image
from .options import FaceAlignmentOptions

# Suffixes for common image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for the image alignment pipeline.

    Attributes:
        source: Directory containing source images or a single image file.
        aligned_dir: Directory where aligned images are written.
        landmark_model_path: Path to the dlib landmark predictor model.
        face_alignment: Options used when aligning each detected face.
    """

    source: Path
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
    """Return whether to skip an image because its first output exists."""
    first_face_path = _aligned_face_path(config, raw_img_path, 1)
    if first_face_path.is_file() and not overwrite:
        logger.info("skipping existing file %s", first_face_path.name)
        return True

    return False


def _source_image_paths(source: Path) -> list[Path]:
    """Return source image paths selected for alignment."""
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            return [source]
        return []

    if source.is_dir():
        return [
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    raise ValueError(
        f"Source path does not exist or is not accessible: {source}"
    )


def _align_detected_faces(
    config: AlignmentConfig,
    raw_img_path: Path,
    detector: Any,
    predictor: Any,
) -> None:
    """Align and crop every detected face for one source image."""
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

    Writes one aligned image per detected face to `config.aligned_dir`,
    creating the directory if it does not exist. When `config.source` is a
    directory, only immediate children with recognized image extensions are
    processed and subdirectories are ignored. When `config.source` is a file,
    it is processed only if its extension is recognized. Existing outputs are
    skipped unless `overwrite` is true. Detection and alignment failures for
    individual images are logged and do not stop the whole run.

    Args:
        config: Pipeline configuration options.
        overwrite: Whether to overwrite existing aligned images.
    """
    source_image_paths = _source_image_paths(config.source)

    # If AlignmentConfig.aligned_dir does not exist, create it.
    config.aligned_dir.mkdir(parents=True, exist_ok=True)

    detector = get_detector()
    predictor = get_predictor(config.landmark_model_path)

    for raw_img_path in source_image_paths:
        logger.info("Aligning %s ...", raw_img_path.name)
        try:
            if _should_skip_existing(config, raw_img_path, overwrite):
                continue
            _align_detected_faces(config, raw_img_path, detector, predictor)
        except Exception:
            logger.exception("Exception in landmark detection.")
