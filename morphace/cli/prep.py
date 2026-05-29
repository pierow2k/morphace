"""CLI entrypoint for aligning faces and producing cropped images."""

import argparse
import logging
import sys
from importlib.metadata import version
from pathlib import Path

from morphace.alignment import (
    AlignmentConfig,
    FaceAlignmentOptions,
    align_faces,
)
from morphace.landmarks import (
    MODEL_FILENAME,
    LandmarkModelNotFoundError,
    resolve_landmark_model_path,
)

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add prep command arguments to an argument parser."""
    parser.add_argument(
        "source", help="Path to a directory of images or a single image file"
    )
    parser.add_argument(
        "aligned_dir",
        nargs="?",
        default=argparse.SUPPRESS,
        help="Directory for aligned images (default: <source>/cropped for "
        "directories, same directory as source for files)",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        action="store_true",
        default=False,
        help=(
            "Use alpha channel for padded regions instead of reflected padding"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "-e",
        "--em-scale",
        default=0.1,
        help="Shift crop center from eyes toward mouth",
        type=float,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        dest="overwrite",
        help="Overwrite existing aligned images",
    )
    parser.add_argument(
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
    parser.add_argument(
        "-s",
        "--output-size",
        default=1024,
        help="Pixel dimension of the output square crop",
        type=int,
    )
    parser.add_argument(
        "-x",
        "--x-scale",
        default=1.0,
        help=(
            "Horizontal crop extent around face; >1.0 includes more "
            "context, <1.0 crops tighter"
        ),
        type=float,
    )
    parser.add_argument(
        "-y",
        "--y-scale",
        default=1.0,
        help=(
            "Vertical crop extent around face; >1.0 includes more "
            "context, <1.0 crops tighter"
        ),
        type=float,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s " + version("morphace"),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect faces in raw images and prepare aligned square PNG crops "
            "for morphing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    return parser.parse_args(argv)


def _configure_logging(debug: bool) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def run(args: argparse.Namespace) -> int:
    """Run the prep command with parsed arguments."""
    _configure_logging(args.debug)

    source = Path(args.source)

    if not source.exists():
        logger.error("Source path does not exist: %s", source)
        return 1

    if not hasattr(args, "aligned_dir"):
        args.aligned_dir = (
            source / "cropped" if source.is_dir() else source.parent
        )

    try:
        landmark_model_path = resolve_landmark_model_path(args.landmark_model)
    except LandmarkModelNotFoundError:
        logger.exception("error resolving landmark model path")
        return 1

    config = AlignmentConfig(
        source=source,
        aligned_dir=Path(args.aligned_dir),
        landmark_model_path=landmark_model_path,
        face_alignment=FaceAlignmentOptions(
            output_size=args.output_size,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            em_scale=args.em_scale,
            alpha=args.alpha,
        ),
    )

    align_faces(config, overwrite=args.overwrite)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the prep subcommand."""
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
