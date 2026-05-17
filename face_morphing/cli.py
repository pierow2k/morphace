"""Command-line interface for face morphing."""

import argparse
import logging
import sys

import cv2

from . import MorphConfig, NoFaceFoundError, morph_faces

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI for face-morphing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate a face morphing video."
    )
    parser.add_argument(
        "--img1", required=True, help="Path to the first source image"
    )
    parser.add_argument(
        "--img2", required=True, help="Path to the second source image"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output video (e.g., output.mp4)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration of the morph in seconds",
    )
    parser.add_argument(
        "--frame", type=int, default=20, help="Frame rate (FPS)"
    )
    parser.add_argument(
        "--show-triangles",
        action="store_true",  # Use store_true for simple boolean flags
        default=False,
        help="Show triangulation lines in the video",
    )
    args = parser.parse_args()

    # Input validation
    if args.duration <= 0 or args.frame <= 0:
        logger.error("Duration and frame rate must be positive integers.")
        sys.exit(1)

    image1 = cv2.imread(args.img1)
    image2 = cv2.imread(args.img2)

    if image1 is None:
        logger.error("Could not read image: %s", args.img1)
        sys.exit(1)
    if image2 is None:
        logger.error("Could not read image: %s", args.img2)
        sys.exit(1)

    config = MorphConfig(
        duration=args.duration,
        frame_rate=args.frame,
        output=args.output,
    )

    try:
        morph_faces(image1, image2, config, args.show_triangles)
        logger.info("Morphing complete. Video saved to %s", args.output)
    except NoFaceFoundError:
        # Intentionally use error() to avoid a scary stack trace for a user
        # input issue. We suppress the linter warning because we don't need
        # a traceback here.
        logger.error("Error: Could not detect a face in one or both images.")  # noqa: TRY400
        sys.exit(1)
    except RuntimeError:
        # Use exception() to log the stack trace for unexpected system errors.
        logger.exception("An unexpected runtime error occurred.")
        sys.exit(1)


if __name__ == "__main__":
    main()
