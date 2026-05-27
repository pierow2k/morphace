"""Tests for the landmark model download command-line interface."""

from pathlib import Path

import pytest

from morphace.cli import download
from morphace.landmarks import MODEL_FILENAME


def test_main_downloads_to_default_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the default model path is used when save-to is omitted."""
    captured: dict[str, object] = {}

    def fake_default_landmark_model_path() -> Path:
        """Return a fake default model path."""
        return Path("default") / MODEL_FILENAME

    def fake_download_dlib_model(model_path: Path, overwrite: bool) -> None:
        """Capture the download call."""
        captured["model_path"] = model_path
        captured["overwrite"] = overwrite

    monkeypatch.setattr(
        download,
        "default_landmark_model_path",
        fake_default_landmark_model_path,
    )
    monkeypatch.setattr(
        download,
        "download_dlib_model",
        fake_download_dlib_model,
    )

    result = download.main([])

    assert result == 0
    assert captured == {
        "model_path": Path("default") / MODEL_FILENAME,
        "overwrite": False,
    }


def test_main_downloads_to_save_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify save-to is treated as the target directory."""
    captured: dict[str, object] = {}

    def fake_download_dlib_model(model_path: Path, overwrite: bool) -> None:
        """Capture the download call."""
        captured["model_path"] = model_path
        captured["overwrite"] = overwrite

    monkeypatch.setattr(
        download,
        "download_dlib_model",
        fake_download_dlib_model,
    )

    result = download.main(["--save-to", "model", "--force"])

    assert result == 0
    assert captured == {
        "model_path": Path("model") / MODEL_FILENAME,
        "overwrite": True,
    }


def test_main_returns_error_when_download_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify download failures become CLI errors."""

    def fake_default_landmark_model_path() -> Path:
        """Return a fake default model path."""
        return Path("default") / MODEL_FILENAME

    def fake_download_dlib_model(model_path: Path, overwrite: bool) -> None:
        """Raise an expected download failure."""
        del model_path, overwrite
        raise RuntimeError("download failed")

    monkeypatch.setattr(
        download,
        "default_landmark_model_path",
        fake_default_landmark_model_path,
    )
    monkeypatch.setattr(
        download,
        "download_dlib_model",
        fake_download_dlib_model,
    )

    result = download.main([])

    assert result == 1
