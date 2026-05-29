"""Download and decompress the dlib facial landmark model file.

This module provides a command-line interface for downloading the dlib
shape predictor model from the official dlib-models GitHub repository.
The downloaded .bz2 archive is automatically decompressed and saved to
the user's data directory or a specified path.
"""

import argparse
import bz2
import logging
import sys
import tempfile
from pathlib import Path

import requests

from morphace.landmarks import MODEL_FILENAME, default_landmark_model_path

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add download command arguments to an argument parser."""
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        dest="overwrite",
        help="Overwrite existing model file.",
    )
    parser.add_argument(
        "-s",
        "--save-to",
        type=Path,
        default=None,
        help=(
            "Path to save the model file. "
            "If omitted, the default user data directory will be used."
        ),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Optional list of command line arguments. If None, defaults
            to sys.argv[1:].

    Returns:
        Namespace containing parsed arguments with attributes:
            debug (bool): Whether debug logging is enabled.
            overwrite (bool): Whether to overwrite existing files.
            save_to (Path | None): Custom save directory, or None for default.
    """
    parser = argparse.ArgumentParser(
        description=(f"Download dlib {MODEL_FILENAME} landmark model file."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    return parser.parse_args(argv)


def _configure_logging(debug: bool) -> None:
    """Configure the root logger for the download utility.

    Args:
        debug: If True, sets logging level to DEBUG for verbose output.
            Otherwise, uses INFO level for standard operation messages.
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def show_license_info() -> None:
    """Log license information for the dlib model."""
    logger.info(
        "The dlib model is trained on the ibug 300-W dataset. The "
        "license for this dataset excludes commercial use the trained "
        "model therefore cannot be used in a commercial product. Contact "
        "Imperial College London to for more details before using the "
        "model in a commercial product.\n"
        "https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/"
    )


def download_dlib_model(model_path: Path, overwrite: bool) -> None:
    """Download and decompress the dlib model file to the specified path.

    Downloads the compressed model from GitHub, streams it to a temporary
    file to minimize memory usage, then decompresses it to the final
    destination. Uses atomic write semantics to prevent partial files.

    Args:
        model_path: Full path (including filename) where the decompressed
            model file will be saved. Parent directories are created if
            they do not exist.
        overwrite: If True, overwrites any existing file at model_path.
            If False and the file exists, logs a message and returns early.

    Raises:
        requests.exceptions.RequestException: If a network error occurs
            during download (connection failure, timeout, HTTP error, etc.).
        OSError: If a file system error occurs (permission denied, disk full,
            etc.) or if decompression fails.
    """
    url = (
        "https://github.com/davisking/dlib-models/raw/refs/heads/master/"
        + MODEL_FILENAME
        + ".bz2"
    )

    logger.debug("URL is: %s", url)

    if not overwrite and model_path.exists():
        logger.info(
            "File '%s' already exists. Use --force to overwrite.", model_path
        )
        return

    show_license_info()

    # Ensure parent directory exists before attempting download
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dlib model from: %s", url)

    try:
        # Write to temporary file for atomic replacement on success
        # delete=False allows manual rename after close; we clean up explicitly
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent, suffix=".tmp", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

            # Stream response to avoid loading entire file into memory
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)

        # Decompress the temporary file to the final destination
        logger.info("Decompressing model...")
        try:
            with tmp_path.open("rb") as src, model_path.open("wb") as dst:
                decompressor = bz2.BZ2Decompressor()
                # Process in chunks to maintain low memory footprint
                while True:
                    chunk = src.read(8192)
                    if not chunk:
                        break
                    dst.write(decompressor.decompress(chunk))
        except Exception:
            # Clean up partial output file if decompression fails
            if model_path.exists():
                model_path.unlink()
            raise

        # Remove the temporary compressed file after successful decompression
        tmp_path.unlink()

        logger.info("Model file saved to: %s", model_path)

    except requests.exceptions.RequestException as e:
        # Log without traceback to avoid intimidating output for expected errors
        logger.error("Network error downloading model: %s", e)  # noqa: TRY400
        raise
    except OSError as e:
        logger.error(  # noqa: TRY400
            "File system error saving the model: %s", e
        )
        raise


def run(args: argparse.Namespace) -> int:
    """Run the dlib model download command.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code: 0 on successful download or if file already exists,
            1 if an error occurred during download or decompression.
    """
    _configure_logging(args.debug)

    if args.save_to is None:
        landmark_model_path = default_landmark_model_path()
    else:
        # Join directory path with filename using pathlib operator
        landmark_model_path = args.save_to / MODEL_FILENAME

    try:
        download_dlib_model(landmark_model_path, args.overwrite)
    except Exception:  # pylint: disable=broad-exception-caught
        # Error details are logged within download_dlib_model
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the dlib model download utility."""
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
