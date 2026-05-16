"""Face morphing package initialization."""

import argparse
import logging

import cv2

from .delaunay_triangulation import make_delaunay
from .face_landmark_detection import generate_face_correspondences
from .face_morph import generate_morph_sequence

logger = logging.getLogger(__name__)

def do_morphing(img1, img2, duration, frame_rate, output):
    """Perform face morphing between two images.

    Args:
        img1: The first input image.
        img2: The second input image.
        duration: Duration of the morphing sequence in seconds.
        frame_rate: Number of frames per second.
        output: Path to save the output video.
    """
    [size, img1, img2, points1, points2, list3] = \
        generate_face_correspondences(img1, img2)

    tri = make_delaunay(size[1], size[0], list3)

    generate_morph_sequence(
        duration, frame_rate, img1, img2, points1, points2, tri, size, output
    )


def main():
    """Main entry point for the face-morphing CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", required=True, help="The First Image")
    parser.add_argument("--img2", required=True, help="The Second Image")
    parser.add_argument(
        "--duration", type=int, default=5, help="The duration"
    )
    parser.add_argument(
        "--frame", type=int, default=20, help="The frame Rate"
    )
    parser.add_argument("--output", help="Output Video Path")
    args = parser.parse_args()

    image1 = cv2.imread(args.img1)
    image2 = cv2.imread(args.img2)

    if image1 is None:
        logger.error("Error: Could not read image %s", args.img1)
        return
    if image2 is None:
        logger.error("Error: Could not read image %s", args.img2)
        return

    do_morphing(image1, image2, args.duration, args.frame, args.output)


if __name__ == "__main__":
    main()
