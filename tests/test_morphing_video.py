"""Tests for video stream handling and FFmpeg integration."""

from unittest.mock import MagicMock

import pytest

from morphace.morphing.config import MorphVideoConfig
from morphace.morphing.video import video_writer_context


def test_video_writer_context_terminates_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that exception inside context manager terminates process."""
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdin.closed = False
    mock_process.wait.return_value = 0

    monkeypatch.setattr(
        "morphace.morphing.video.Popen", lambda *_, **__: mock_process
    )

    config = MorphVideoConfig(
        duration=5, frame_rate=30, size=(480, 640), output="output.mp4"
    )

    with (
        pytest.raises(
            ValueError,
            match="Inside context error",
        ),
        video_writer_context(config),
    ):
        raise ValueError("Inside context error")

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
        "morphace.morphing.video.Popen", lambda *_, **__: mock_process
    )

    config = MorphVideoConfig(
        duration=5, frame_rate=30, size=(480, 640), output="output.mp4"
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
        "morphace.morphing.video.Popen", lambda *_, **__: mock_process
    )

    config = MorphVideoConfig(
        duration=5, frame_rate=30, size=(480, 640), output="output.mp4"
    )

    with video_writer_context(config):
        pass

    mock_process.stdin.close.assert_not_called()
