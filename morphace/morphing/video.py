"""Utilities for video stream handling and FFmpeg integration."""

from collections.abc import Iterator
from contextlib import contextmanager
from subprocess import PIPE, Popen
from typing import IO

from .config import MorphVideoConfig


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
    width, height = config.size
    num_images = max(int(config.duration * config.frame_rate), 1)
    size_str = f"{width}x{height}"

    process = Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "image2pipe",
            "-r",
            str(config.frame_rate),
            "-s",
            size_str,
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "25",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            config.output,
        ],
        stdin=PIPE,
    )

    if process.stdin is None:
        raise RuntimeError("Unable to open ffmpeg input stream.")

    try:
        yield process.stdin, num_images
    except BrokenPipeError:
        raise RuntimeError("FFmpeg process ended unexpectedly.") from None
    finally:
        process.stdin.close()
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed with return code {process.returncode}."
            )
