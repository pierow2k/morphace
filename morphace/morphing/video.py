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

    # Use a list for the command; avoids shell injection issues
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        config.ffmpeg_loglevel,
        "-y",  # Overwrite output
        "-f",
        "image2pipe",  # Reading images from pipe
        "-r",
        str(config.frame_rate),
        "-s",
        size_str,
        "-i",
        "-",  # stdin
        "-c:v",
        "libx264",
        "-crf",
        "25",
        # Ensure dimensions are divisible by 2 for H.264
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",  # Compatibility for players
        config.output,
    ]

    # stderr=PIPE allows capturing error messages if the process fails
    process = Popen(
        cmd,
        stdin=PIPE,
        stderr=PIPE,  # Capture stderr to diagnose failures
    )

    if process.stdin is None:
        raise RuntimeError("Unable to open ffmpeg input stream.")

    try:
        yield process.stdin, num_images
    except BrokenPipeError:
        # If the pipe breaks, ffmpeg likely crashed; check stderr
        stderr_output = process.stderr.read().decode() if process.stderr else ""
        raise RuntimeError(
            f"FFmpeg process ended unexpectedly.\n"
            f"FFmpeg Error:\n{stderr_output}"
        ) from None
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
            stderr_output = ""
            if process.stderr:
                stderr_output = process.stderr.read().decode()
            cmd_str = " ".join(cmd)
            raise RuntimeError(
                f"FFmpeg failed with return code {return_code}.\n"
                f"Command: {cmd_str}\n"
                f"Error:\n{stderr_output}"
            )
