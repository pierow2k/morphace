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

    Args:
        config: Video stream configuration.

    Yields:
        A tuple containing (stdin_stream, num_frames).

    Raises:
        RuntimeError: If FFmpeg fails to start or closes unexpectedly.
    """
    # config.size is stored as height x width. FFmpeg expects width x height.
    height, width = config.size
    num_images = max(int(config.duration * config.frame_rate), 1)
    # Replicate the FFmpeg scale logic: round down to the nearest even number.
    # FFmpeg: trunc(iw/2)*2 is equivalent to integer division by 2,
    # multiplied by 2.
    final_width = width // 2 * 2
    final_height = height // 2 * 2

    logger.debug("Input dimensions: %dx%d", width, height)
    logger.debug("Video resolution will be %dx%d", final_width, final_height)

    # Use a list for the command; avoids shell injection issues
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        config.ffmpeg_loglevel,
        "-y",  # Overwrite output
        "-f",
        "rawvideo",  # Read raw bytes
        "-r",
        str(config.frame_rate),
        "-s",
        # Pass the original sizes, FFmpeg scales it internally
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",  # Use BGR24 as the input format
        "-i",
        "-",  # stdin
        "-c:v",
        "libx264",
        "-crf",  # Constant Rate Factor
        "25",
        # Ensure dimensions are divisible by 2 for H.264
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",  # Compatibility for players
        config.output,
    ]

    process = Popen(
        cmd,
        stdin=PIPE,
    )

    if process.stdin is None:
        raise RuntimeError("Unable to open ffmpeg input stream.")

    try:
        yield process.stdin, num_images
    except BrokenPipeError:
        raise RuntimeError("FFmpeg process ended unexpectedly.") from None
    except Exception:
        # Safety net for other exceptions
        process.terminate()
        raise
    finally:
        # Close stdin to signal EOF to ffmpeg
        if not process.stdin.closed:
            process.stdin.close()

        # Wait for ffmpeg to finish writing the file
        return_code = process.wait()

        # If there was an error not caught by BrokenPipeError
        if return_code != 0:
            cmd_str = " ".join(cmd)
            raise RuntimeError(
                f"FFmpeg failed with return code {return_code}.\n"
                f"Command: {cmd_str}"
            )
