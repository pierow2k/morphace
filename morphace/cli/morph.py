"""Command-line interface for the morphace application.

Parses command-line arguments, validates inputs, and delegates
to the morphace workflow to generate a face morphing video.

Usage:
    morphace morph <image1> <image2> [options]

Example:
    morphace morph source1.jpg source2.jpg --output morph.mp4 --duration 5
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
    MODEL_FILENAME,
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
        output_filename: The validated output filename for the morph video.
        ffmpeg_loglevel: The log level string for ffmpeg execution.
    """

    image1: Any
    image2: Any
    landmark_model_path: Path
    output_filename: str
    ffmpeg_loglevel: str


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add morph command arguments to an argument parser."""
    parser.add_argument(
        "img1",
        help="Path to the first face image",
    )
    parser.add_argument(
        "img2",
        help="Path to the second face image",
    )

    output_options = parser.add_argument_group("output options")
    output_options.add_argument(
        "-o",
        "--output",
        default="morph.mp4",
        help="Path for the output MP4 video",
    )
    output_options.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        dest="overwrite",
        help="Overwrite existing MP4 video",
    )

    morph_options = parser.add_argument_group("morph options")
    morph_options.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5,
        help="Duration of the morph video in seconds",
    )
    morph_options.add_argument(
        "-r",
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output video",
    )
    morph_options.add_argument(
        "-m",
        "--show-mesh",
        action="store_true",
        default=False,
        help="Draw triangle mesh overlay to visualize face geometry warping",
    )

    model_options = parser.add_argument_group("model options")
    model_options.add_argument(
        "-l",
        "--landmark-model",
        default=None,
        help=(
            f"Path to {MODEL_FILENAME} model file. "
            "If omitted, morphace checks MORPHACE_LANDMARK_MODEL "
            "and then the default user data directory."
        ),
        type=Path,
    )

    diagnostic_options = parser.add_argument_group("diagnostic options")
    diagnostic_options.add_argument(
        "-v",
        "--show-ffmpeg-output",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Show FFmpeg informational output while encoding. "
            "(default: only show FFmpeg errors)"
        ),
    )
    diagnostic_options.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )

    general_options = parser.add_argument_group("general options")
    general_options.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit",
    )
    general_options.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s " + version("morphace"),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a face morphing video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    add_arguments(parser)
    return parser.parse_args(argv)


def _read_input_images(args: argparse.Namespace) -> tuple[Any, Any] | None:
    """Read both input images, returning None if either cannot be read."""
    if not Path(args.img1).exists():
        logger.error("Input image %s does not exist", args.img1)
        return None
    if not Path(args.img2).exists():
        logger.error("Input image %s does not exist", args.img2)
        return None

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

    Checks that the output file does not already exist, that the
    duration and FPS are positive, ffmpeg is available, input
    images are readable, and the landmark model exists.

    Args:
        args: Parsed command line arguments.

    Returns:
        A ValidatedInputs object containing the processed inputs if
        validation succeeds, or an integer error code (1) if validation fails.
    """
    if not args.overwrite and Path(args.output).exists():
        logger.error("Output file already exists. Use --force to overwrite.")
        return 1

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

    # Output filename must end with .mp4 for ffmpeg to recognize the format.
    if not args.output.lower().endswith(".mp4"):
        output_filename = args.output + ".mp4"
    else:
        output_filename = args.output

    return ValidatedInputs(
        image1=images[0],
        image2=images[1],
        landmark_model_path=landmark_model_path,
        ffmpeg_loglevel=ffmpeg_loglevel,
        output_filename=output_filename,
    )


def run(args: argparse.Namespace) -> int:
    """Run the morph command with parsed arguments."""
    _configure_logging(args.debug)

    validated = _validate_inputs(args)
    if isinstance(validated, int):
        return validated

    config = MorphConfig(
        duration=args.duration,
        frame_rate=args.fps,
        output=validated.output_filename,
        landmark_model_path=validated.landmark_model_path,
        ffmpeg_loglevel=validated.ffmpeg_loglevel,
    )

    try:
        morph_faces(validated.image1, validated.image2, config, args.show_mesh)
        logger.info(
            "Morphing complete. Video saved to %s", validated.output_filename
        )
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


def main(argv: list[str] | None = None) -> int:
    """CLI for the morph subcommand."""
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
