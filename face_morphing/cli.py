"""Command-line interface for face morphing."""

import argparse
import logging

import cv2

from . import MorphConfig, morph_faces

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the face-morphing CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", required=True, help="The First Image")
    parser.add_argument("--img2", required=True, help="The Second Image")
    parser.add_argument("--duration", type=int, default=5, help="The duration")
    parser.add_argument("--frame", type=int, default=20, help="The frame Rate")
    parser.add_argument("--output", help="Output Video Path")
    parser.add_argument(
        "--show-triangles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show triangulation lines",
    )
    args = parser.parse_args()

    image1 = cv2.imread(args.img1)
    image2 = cv2.imread(args.img2)

    if image1 is None:
        logger.error("Error: Could not read image %s", args.img1)
        return
    if image2 is None:
        logger.error("Error: Could not read image %s", args.img2)
        return

    config = MorphConfig(
        duration=args.duration,
        frame_rate=args.frame,
        output=args.output,
    )

    morph_faces(image1, image2, config, args.show_triangles)
