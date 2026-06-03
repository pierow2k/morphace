"""Utilities for video stream handling and FFmpeg integration."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from subprocess import PIPE, Popen
from typing import IO

from .config import MorphVideoConfig

logger = logging.getLogger(__name__)


@contextmanager
def video_writer_context(
    config: MorphVideoConfig,
) -> Iterator[tuple[IO[bytes], int]]:
    """Context manager to handle the FFmpeg process lifecycle.

    This manager configures and invokes FFmpeg as a subprocess, yielding a
    writeable stream for raw video frames. It ensures proper cleanup of the
    subprocess upon exit.

    Args:
        config: Video stream configuration object.

    Yields:
        A tuple containing (stdin_stream, num_frames).

    Raises:
        RuntimeError: If FFmpeg fails to start, crashes during execution,
                      or exits with a non-zero status code.
    """
    # config.size is stored as (height, width). FFmpeg expects 'width x height'.
    height, width = config.size
    num_images = max(int(config.duration * config.frame_rate), 1)

    # Calculate output resolution. H.264 requires even dimensions.
    # This logic replicates the FFmpeg filter: trunc(iw/2)*2.
    final_width = width // 2 * 2
    final_height = height // 2 * 2

    logger.debug("Input dimensions: %dx%d", width, height)
    logger.debug("Video resolution will be %dx%d", final_width, final_height)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        config.ffmpeg_loglevel,
        "-y",  # Overwrite output files without asking.
        # --- Input Configuration ---
        "-f",
        "rawvideo",  # Format: raw video data
        "-r",
        str(config.frame_rate),  # Frame rate
        "-s",
        f"{width}x{height}",  # Frame size
        "-pix_fmt",
        "bgr24",  # Input pixel format (OpenCV default)
        "-i",
        "-",  # Read from stdin
        # --- Output Configuration ---
        "-c:v",
        "libx264",  # Codec: H.264
        "-crf",
        "25",  # Quality: Constant Rate Factor
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Force even dimensions
        "-pix_fmt",
        "yuv420p",  # Output pixel format (broad compatibility)
        config.output,
    ]

    # We trust the command construction and are not using shell=True.
    process = Popen(cmd, stdin=PIPE)  # noqa: S603

    if process.stdin is None:
        raise RuntimeError("Unable to open FFmpeg input stream.")

    try:
        yield process.stdin, num_images
    except BrokenPipeError:
        raise RuntimeError("FFmpeg process ended unexpectedly.") from None
    except Exception:
        # Safety net: ensure subprocess is terminated on other exceptions.
        process.terminate()
        raise
    finally:
        # Close stdin to signal EOF to FFmpeg.
        if not process.stdin.closed:
            process.stdin.close()

        # Wait for FFmpeg to finalize the file.
        return_code = process.wait()

        if return_code != 0:
            cmd_str = " ".join(cmd)
            raise RuntimeError(
                f"FFmpeg failed with return code {return_code}.\n"
                f"Command: {cmd_str}",
            )
