"""Utilities for video stream handling and FFmpeg integration."""

from collections.abc import Iterator
from contextlib import contextmanager
from subprocess import PIPE, Popen
from typing import IO

from ._typing import Size


@contextmanager
def video_writer_context(
    config: tuple[int, int, Size, str],
) -> Iterator[tuple[IO[bytes], int]]:
    """Context manager to handle the FFmpeg process lifecycle.

    Args:
        config: A tuple of (duration, frame_rate, size, output_path).

    Yields:
        A tuple containing (stdin_stream, num_frames).

    Raises:
        RuntimeError: If FFmpeg fails to start or closes unexpectedly.
    """
    duration, frame_rate, size, output = config
    width, height = size
    num_images = max(int(duration * frame_rate), 1)
    size_str = f"{width}x{height}"

    process = Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-r",
            str(frame_rate),
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
            output,
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
