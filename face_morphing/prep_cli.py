"""CLI entrypoint for aligning faces and producing cropped images."""

import argparse
import logging
import sys
from pathlib import Path

from .prep_images import AlignmentConfig, align_faces


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Align faces from input images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "raw_dir", help="Directory with raw images for face alignment"
    )
    parser.add_argument(
        "aligned_dir", help="Directory for storing aligned images"
    )
    parser.add_argument(
        "--landmark-model",
        default=None,
        help="Path to shape_predictor_68_face_landmarks.dat.",
    )
    parser.add_argument(
        "--output_size",
        default=1024,
        help="The dimension of images for input to the model",
        type=int,
    )
    parser.add_argument(
        "--x_scale",
        default=1.0,
        help="Scaling factor for x dimension",
        type=float,
    )
    parser.add_argument(
        "--y_scale",
        default=1.0,
        help="Scaling factor for y dimension",
        type=float,
    )
    parser.add_argument(
        "--em_scale",
        default=0.1,
        help="Scaling factor for eye-mouth distance",
        type=float,
    )
    parser.add_argument(
        "--use_alpha",
        default=False,
        help="Add an alpha channel for masking",
        type=bool,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = AlignmentConfig(
        raw_dir=Path(args.raw_dir),
        aligned_dir=Path(args.aligned_dir),
        landmark_model_path=Path(args.landmark_model),
        output_size=args.output_size,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        em_scale=args.em_scale,
        use_alpha=args.use_alpha,
    )

    align_faces(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
