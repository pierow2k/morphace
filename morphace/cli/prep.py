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
    LandmarkModelNotFoundError,
    resolve_landmark_model_path,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect faces in raw images and prepare aligned square PNG crops "
            "for morphing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        "-l",
        "--landmark-model",
        type=Path,
        default=None,
        help=(
            "Path to shape_predictor_68_face_landmarks.dat. "
            "If omitted, morphace checks MORPHACE_LANDMARK_MODEL "
            "and then the default user data directory."
        ),
    )
    parser.add_argument(
        "-s",
        "--output-size",
        default=1024,
        help="Pixel dimension of the output square crop",
        type=int,
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        default=False,
        help="Overwrite existing aligned images",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--em-scale",
        default=0.1,
        help="Shift crop center from eyes toward mouth",
        type=float,
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
        "-a",
        "--alpha",
        default=False,
        help=(
            "Use alpha channel for padded regions instead of reflected padding"
        ),
        action="store_true",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s " + version("morphace"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    source = Path(args.source)

    if not hasattr(args, "aligned_dir"):
        args.aligned_dir = (
            source / "cropped" if source.is_dir() else source.parent
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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


if __name__ == "__main__":
    sys.exit(main())
