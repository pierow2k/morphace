"""Tests for FFmpeg video writer lifecycle management."""

from __future__ import annotations

import io
from typing import Any

import pytest

from morphace.morphing import video
from morphace.morphing.config import MorphVideoConfig


class _FakeProcess:
    """Small stand-in for an FFmpeg subprocess."""

    def __init__(
        self,
        stdin: io.BytesIO | None = None,
        stderr: io.BytesIO | None = None,
        returncode: int = 0,
    ) -> None:
        """Initialize the fake process."""
        self.stdin = stdin
        self.stderr = stderr
        self.returncode = returncode
        self.wait_called = False

    def wait(self) -> int:
        """Record process waiting and return the configured code."""
        self.wait_called = True
        return self.returncode


def test_video_writer_context_builds_ffmpeg_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the video writer opens ffmpeg with expected stream settings."""
    process = _FakeProcess(stdin=io.BytesIO())
    captured: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        """Capture the ffmpeg command."""
        captured["command"] = command
        captured.update(kwargs)
        return process

    monkeypatch.setattr(video, "Popen", fake_popen)

    config = MorphVideoConfig(
        duration=2,
        frame_rate=3,
        size=(8, 10),
        output="out.mp4",
        ffmpeg_loglevel="info",
    )
    with video.video_writer_context(config) as (stdin, num_images):
        assert stdin is process.stdin
        assert num_images == 6  # noqa:PLR2004

    assert captured["stdin"] == video.PIPE
    assert captured["stderr"] == video.PIPE
    assert captured["command"] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-f",
        "image2pipe",
        "-r",
        "3",
        "-s",
        "10x8",
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
        "out.mp4",
    ]
    assert process.stdin is not None
    assert process.stdin.closed
    assert process.wait_called


def test_video_writer_context_requires_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify missing ffmpeg stdin is reported as a runtime failure."""
    process = _FakeProcess(stdin=None)

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        """Return a process without stdin."""
        del command, kwargs
        return process

    monkeypatch.setattr(video, "Popen", fake_popen)

    with (
        pytest.raises(RuntimeError, match="input stream"),
        video.video_writer_context(MorphVideoConfig(1, 30, (4, 4), "out.mp4")),
    ):
        pass


def test_video_writer_context_converts_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ffmpeg pipe failures are converted to RuntimeError."""
    process = _FakeProcess(stdin=io.BytesIO())

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        """Return a writable fake process."""
        del command, kwargs
        return process

    monkeypatch.setattr(video, "Popen", fake_popen)

    with (
        pytest.raises(RuntimeError, match="ended unexpectedly"),
        video.video_writer_context(MorphVideoConfig(1, 30, (4, 4), "out.mp4")),
    ):
        raise BrokenPipeError

    assert process.stdin is not None
    assert process.stdin.closed
    assert process.wait_called


def test_video_writer_context_reports_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ffmpeg nonzero exits fail after stream cleanup."""
    process = _FakeProcess(stdin=io.BytesIO(), returncode=2)

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        """Return a process configured to fail on wait."""
        del command, kwargs
        return process

    monkeypatch.setattr(video, "Popen", fake_popen)

    with (
        pytest.raises(RuntimeError, match="return code 2"),
        video.video_writer_context(MorphVideoConfig(1, 30, (4, 4), "out.mp4")),
    ):
        pass

    assert process.stdin is not None
    assert process.stdin.closed
    assert process.wait_called
