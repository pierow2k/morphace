"""Tests for video stream handling and FFmpeg integration."""

from subprocess import PIPE
from unittest.mock import MagicMock

import pytest

from morphace.morphing.config import MorphVideoConfig
from morphace.morphing.video import video_writer_context

VIDEO_HEIGHT = 481
VIDEO_WIDTH = 641
VIDEO_DURATION_SECONDS = 0
VIDEO_FRAME_RATE = 30


def test_video_writer_context_terminates_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that exception inside context manager terminates process."""
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdin.closed = False
    mock_process.wait.return_value = 0

    monkeypatch.setattr(
        "morphace.morphing.video.Popen",
        lambda *_, **__: mock_process,
    )

    config = MorphVideoConfig(
        duration=5,
        frame_rate=30,
        size=(480, 640),
        output="output.mp4",
    )

    message = "Inside context error"

    with (
        pytest.raises(
            ValueError,
            match=message,
        ),
        video_writer_context(config),
    ):
        raise ValueError(message)

    mock_process.terminate.assert_called_once()
    # verify finally block closed stdin
    mock_process.stdin.close.assert_called_once()


def test_video_writer_context_closes_stdin_if_not_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify stdin is closed by context manager if caller didn't do it."""
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdin.closed = False
    mock_process.wait.return_value = 0

    monkeypatch.setattr(
        "morphace.morphing.video.Popen",
        lambda *_, **__: mock_process,
    )

    config = MorphVideoConfig(
        duration=5,
        frame_rate=30,
        size=(480, 640),
        output="output.mp4",
    )

    with video_writer_context(config):
        pass

    mock_process.stdin.close.assert_called_once()


def test_video_writer_context_skips_closing_if_already_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that context manager does not call close() if stdin is closed."""
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdin.closed = True
    mock_process.wait.return_value = 0

    monkeypatch.setattr(
        "morphace.morphing.video.Popen",
        lambda *_, **__: mock_process,
    )

    config = MorphVideoConfig(
        duration=5,
        frame_rate=30,
        size=(480, 640),
        output="output.mp4",
    )

    with video_writer_context(config):
        pass

    mock_process.stdin.close.assert_not_called()


def test_video_writer_context_yields_stream_and_minimum_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the context yields stdin and at least one frame."""
    captured: dict[str, object] = {}
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdin.closed = False
    mock_process.wait.return_value = 0

    def fake_popen(cmd: list[str], *, stdin: int) -> MagicMock:
        """Capture the FFmpeg command used to start the process."""
        captured["cmd"] = cmd
        captured["stdin"] = stdin
        return mock_process

    monkeypatch.setattr("morphace.morphing.video.Popen", fake_popen)

    config = MorphVideoConfig(
        duration=VIDEO_DURATION_SECONDS,
        frame_rate=VIDEO_FRAME_RATE,
        size=(VIDEO_HEIGHT, VIDEO_WIDTH),
        output="output.mp4",
        ffmpeg_loglevel="debug",
    )

    with video_writer_context(config) as (stdin_stream, num_frames):
        assert stdin_stream is mock_process.stdin
        assert num_frames == 1

    assert captured["stdin"] == PIPE
    assert captured["cmd"] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "debug",
        "-y",
        "-f",
        "rawvideo",
        "-r",
        str(VIDEO_FRAME_RATE),
        "-s",
        f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-pix_fmt",
        "bgr24",
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
        "output.mp4",
    ]


def test_video_writer_context_errors_when_stdin_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a missing FFmpeg input stream raises a clear error."""
    mock_process = MagicMock()
    mock_process.stdin = None

    monkeypatch.setattr(
        "morphace.morphing.video.Popen",
        lambda *_, **__: mock_process,
    )

    config = MorphVideoConfig(
        duration=5,
        frame_rate=30,
        size=(480, 640),
        output="output.mp4",
    )

    with (
        pytest.raises(RuntimeError, match="Unable to open FFmpeg"),
        video_writer_context(config),
    ):
        pass

    mock_process.wait.assert_not_called()


def test_video_writer_context_converts_broken_pipe_to_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify broken FFmpeg pipes are reported as runtime errors."""
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdin.closed = False
    mock_process.wait.return_value = 0

    monkeypatch.setattr(
        "morphace.morphing.video.Popen",
        lambda *_, **__: mock_process,
    )

    config = MorphVideoConfig(
        duration=5,
        frame_rate=30,
        size=(480, 640),
        output="output.mp4",
    )

    with (
        pytest.raises(RuntimeError, match="ended unexpectedly"),
        video_writer_context(config),
    ):
        raise BrokenPipeError

    mock_process.terminate.assert_not_called()
    mock_process.stdin.close.assert_called_once()
    mock_process.wait.assert_called_once()


def test_video_writer_context_raises_when_ffmpeg_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-zero FFmpeg exits include the command in the error."""
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdin.closed = False
    mock_process.wait.return_value = 7

    monkeypatch.setattr(
        "morphace.morphing.video.Popen",
        lambda *_, **__: mock_process,
    )

    config = MorphVideoConfig(
        duration=5,
        frame_rate=30,
        size=(480, 640),
        output="output.mp4",
    )

    with (
        pytest.raises(RuntimeError) as error,
        video_writer_context(config),
    ):
        pass

    assert "FFmpeg failed with return code 7" in str(error.value)
    assert "Command: ffmpeg" in str(error.value)
    assert "output.mp4" in str(error.value)
    mock_process.stdin.close.assert_called_once()
