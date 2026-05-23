"""Command-line interface for the morphace application.

Parses command-line arguments, validates inputs, and delegates
to the morphace workflow to generate a face morphing video.

Usage:
    morphace <image1> <image2> [options]

Example:
    morphace source1.jpg source2.jpg --output morph.mp4 --duration 5
"""

import argparse
import logging
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2

from morphace.landmarks import (
    LandmarkModelNotFoundError,
    NoFaceFoundError,
    resolve_landmark_model_path,
)
from morphace.morphing import MorphConfig, morph_faces

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a face morphing video."
    )
    parser.add_argument(
        "img1",
        help="Path to the first image",
    )
    parser.add_argument(
        "img2",
        help="Path to the second image",
    )
    parser.add_argument(
        "--landmark-model",
        type=Path,
        default=None,
        help=(
            "Path to shape_predictor_68_face_landmarks.dat. "
            "If omitted, the app checks MORPHACE_LANDMARK_MODEL "
            "and then the default user data directory."
        ),
    )
    parser.add_argument(
        "--output",
        default="morph.mp4",
        help="Path to save the output video (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration of the morph in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate (FPS) (default: %(default)s)",
    )
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        default=False,
        help="Show the triangle mesh overlay in the output video.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + version("morphace"),
    )
    return parser.parse_args(argv)


def _read_input_images(args: argparse.Namespace) -> tuple[Any, Any] | None:
    """Read both input images, returning None if either cannot be read."""
    image1 = cv2.imread(args.img1)
    image2 = cv2.imread(args.img2)

    if image1 is None:
        logger.error("Could not read image: %s", args.img1)
        return None
    if image2 is None:
        logger.error("Could not read image: %s", args.img2)
        return None

    return image1, image2


def main(argv: list[str] | None = None) -> int:
    """CLI for morphace."""
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Input validation
    if args.duration <= 0 or args.fps <= 0:
        logger.error("Duration and frame rate must be positive integers.")
        return 1

    images = _read_input_images(args)
    if images is None:
        return 1
    image1, image2 = images

    try:
        landmark_model_path = resolve_landmark_model_path(args.landmark_model)
    except LandmarkModelNotFoundError as error:
        # Intentionally use error() to avoid a scary stack trace for a user
        # input issue. We suppress the linter warning because we don't need
        # a traceback here.
        logger.error("%s", error)  # noqa: TRY400
        return 1

    config = MorphConfig(
        duration=args.duration,
        frame_rate=args.fps,
        output=args.output,
        landmark_model_path=landmark_model_path,
    )

    try:
        morph_faces(image1, image2, config, args.show_mesh)
        logger.info("Morphing complete. Video saved to %s", args.output)
    except NoFaceFoundError:
        logger.error(  # noqa: TRY400
            "Error: Could not detect a face in one or both images."
        )
        return 1
    except RuntimeError:
        # Use exception() to log the stack trace for unexpected system errors.
        logger.exception("An unexpected runtime error occurred.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
