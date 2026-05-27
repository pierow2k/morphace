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
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

import cv2

from morphace.landmarks import (
    LandmarkModelNotFoundError,
    NoFaceFoundError,
    resolve_landmark_model_path,
)
from morphace.morphing import MorphConfig, morph_faces

logger = logging.getLogger(__name__)


class ValidatedInputs(NamedTuple):
    """Container for validated inputs required for the morphing workflow.

    Attributes:
        image1: The first input image array.
        image2: The second input image array.
        landmark_model_path: Path to the resolved landmark model file.
        ffmpeg_loglevel: The log level string for ffmpeg execution.
    """

    image1: Any
    image2: Any
    landmark_model_path: Path
    ffmpeg_loglevel: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a face morphing video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "img1",
        help="Path to the first face image",
    )
    parser.add_argument(
        "img2",
        help="Path to the second face image",
    )
    parser.add_argument(
        "-l",
        "--landmark-model",
        default=None,
        help=(
            "Path to `{MODEL_FILENAME}` model file. "
            "If omitted, morphace checks MORPHACE_LANDMARK_MODEL "
            "and then the default user data directory."
        ),
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        default="morph.mp4",
        help="Path for the output MP4 video",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5,
        help="Duration of the morph video in seconds",
    )
    parser.add_argument(
        "-r",
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output video",
    )
    parser.add_argument(
        "-m",
        "--show-mesh",
        action="store_true",
        default=False,
        help="Draw triangle mesh overlay to visualize face geometry warping",
    )
    parser.add_argument(
        "-v",
        "--show-ffmpeg-output",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Show FFmpeg informational output while encoding. "
            "(default: only show FFmpeg errors)"
        ),
    )
    parser.add_argument(
        "-V",
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


def _configure_logging(debug: bool) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def _validate_inputs(args: argparse.Namespace) -> ValidatedInputs | int:
    """Validate command line arguments and system requirements.

    Checks that duration and FPS are positive, ffmpeg is available, input
    images are readable, and the landmark model exists.

    Args:
        args: Parsed command line arguments.

    Returns:
        A ValidatedInputs object containing the processed inputs if
        validation succeeds, or an integer error code (1) if validation fails.
    """
    if args.duration <= 0 or args.fps <= 0:
        logger.error("Duration and frame rate must be positive integers.")
        return 1

    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg is not installed or not in PATH")
        return 1

    images = _read_input_images(args)
    if images is None:
        return 1

    try:
        landmark_model_path = resolve_landmark_model_path(args.landmark_model)
    except LandmarkModelNotFoundError as error:
        # Intentionally use error() to avoid a scary stack trace for a user
        # input issue. We suppress the linter warning because we don't need
        # a traceback here.
        logger.error("%s", error)  # noqa: TRY400
        return 1

    ffmpeg_loglevel = (
        "info" if getattr(args, "show_ffmpeg_output", False) else "error"
    )

    return ValidatedInputs(
        image1=images[0],
        image2=images[1],
        landmark_model_path=landmark_model_path,
        ffmpeg_loglevel=ffmpeg_loglevel,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI for morphace."""
    args = _parse_args(argv)
    _configure_logging(args.debug)

    validated = _validate_inputs(args)
    if isinstance(validated, int):
        return validated

    config = MorphConfig(
        duration=args.duration,
        frame_rate=args.fps,
        output=args.output,
        landmark_model_path=validated.landmark_model_path,
        ffmpeg_loglevel=validated.ffmpeg_loglevel,
    )

    try:
        morph_faces(validated.image1, validated.image2, config, args.show_mesh)
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
