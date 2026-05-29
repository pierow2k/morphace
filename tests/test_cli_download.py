"""Tests for the landmark model download command-line interface."""

import bz2
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

from morphace.cli import download
from morphace.landmarks import MODEL_FILENAME

DOWNLOAD_TIMEOUT_SECONDS = 30
STREAM_CHUNK_SIZE = 8192


class FakeResponse:
    """Minimal requests response double for download tests."""

    def __init__(
        self,
        chunks: tuple[bytes, ...],
        error: requests.exceptions.RequestException | None = None,
    ) -> None:
        """Store fake response chunks and an optional HTTP error."""
        self._chunks = chunks
        self._error = error

    def raise_for_status(self) -> None:
        """Raise the configured HTTP error, if one exists."""
        if self._error is not None:
            raise self._error

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yield fake response chunks using the expected stream size."""
        assert chunk_size == STREAM_CHUNK_SIZE
        yield from self._chunks


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


def test_show_license_info_logs_research_use_notice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify the model license warning mentions research-only use."""
    caplog.set_level(logging.WARNING, logger=download.logger.name)

    download.show_license_info()

    assert "iBUG" in caplog.text
    assert "300-W" in caplog.text
    assert "research purposes only" in caplog.text
    assert "commercial product" in caplog.text


def test_download_dlib_model_skips_existing_file(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify existing models are left untouched unless overwrite is set."""
    model_path = tmp_path / MODEL_FILENAME
    model_path.write_bytes(b"existing model")

    def fake_show_license_info() -> None:
        """Fail if the skip path unexpectedly shows the license notice."""
        raise AssertionError("license notice should not be shown")

    def fake_get(
        url: str,
        *,
        stream: bool,
        timeout: int,
    ) -> FakeResponse:
        """Fail if the skip path unexpectedly downloads the model."""
        del url, stream, timeout
        raise AssertionError("download should not be attempted")

    monkeypatch.setattr(download, "show_license_info", fake_show_license_info)
    monkeypatch.setattr(download.requests, "get", fake_get)
    caplog.set_level(logging.INFO, logger=download.logger.name)

    download.download_dlib_model(model_path, overwrite=False)

    assert model_path.read_bytes() == b"existing model"
    assert "already exists" in caplog.text


def test_download_dlib_model_streams_and_decompresses_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify streamed bz2 content is decompressed to the target path."""
    model_path = tmp_path / "nested" / MODEL_FILENAME
    model_path.parent.mkdir()
    model_path.write_bytes(b"stale model")
    compressed_model = bz2.compress(b"downloaded model")
    captured_call: dict[str, object] = {}

    def fake_get(
        url: str,
        *,
        stream: bool,
        timeout: int,
    ) -> FakeResponse:
        """Return compressed model bytes with an empty chunk mixed in."""
        captured_call["url"] = url
        captured_call["stream"] = stream
        captured_call["timeout"] = timeout
        return FakeResponse((compressed_model[:5], b"", compressed_model[5:]))

    monkeypatch.setattr(download.requests, "get", fake_get)

    download.download_dlib_model(model_path, overwrite=True)

    assert model_path.read_bytes() == b"downloaded model"
    assert captured_call["url"] == (
        "https://github.com/davisking/dlib-models/raw/refs/heads/master/"
        f"{MODEL_FILENAME}.bz2"
    )
    assert captured_call["stream"] is True
    assert captured_call["timeout"] == DOWNLOAD_TIMEOUT_SECONDS
    assert not list(model_path.parent.glob("*.tmp"))


def test_download_dlib_model_reraises_request_errors(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify HTTP failures are logged and re-raised."""
    model_path = tmp_path / MODEL_FILENAME
    request_error = requests.exceptions.RequestException("offline")

    def fake_get(
        url: str,
        *,
        stream: bool,
        timeout: int,
    ) -> FakeResponse:
        """Return a response that fails status validation."""
        del url, stream, timeout
        return FakeResponse((), error=request_error)

    monkeypatch.setattr(download.requests, "get", fake_get)
    caplog.set_level(logging.ERROR, logger=download.logger.name)

    with pytest.raises(requests.exceptions.RequestException, match="offline"):
        download.download_dlib_model(model_path, overwrite=False)

    assert "Network error downloading model: offline" in caplog.text


def test_download_dlib_model_removes_partial_output_on_decompress_error(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify invalid compressed data removes the partial model file."""
    model_path = tmp_path / MODEL_FILENAME

    def fake_get(
        url: str,
        *,
        stream: bool,
        timeout: int,
    ) -> FakeResponse:
        """Return bytes that cannot be decompressed as bz2 data."""
        del url, stream, timeout
        return FakeResponse((b"not a bz2 stream",))

    monkeypatch.setattr(download.requests, "get", fake_get)
    caplog.set_level(logging.ERROR, logger=download.logger.name)

    with pytest.raises(OSError, match="Invalid data stream"):
        download.download_dlib_model(model_path, overwrite=False)

    assert not model_path.exists()
    assert "File system error saving the model" in caplog.text


def test_download_dlib_model_handles_decompress_error_before_output_exists(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify decompression read failures are handled before output exists."""
    model_path = tmp_path / MODEL_FILENAME
    original_open = Path.open

    def fake_get(
        url: str,
        *,
        stream: bool,
        timeout: int,
    ) -> FakeResponse:
        """Return valid compressed bytes so the read failure is isolated."""
        del url, stream, timeout
        return FakeResponse((bz2.compress(b"downloaded model"),))

    def fake_open(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Raise before the model output file is opened."""
        mode = str(args[0]) if args else "r"
        if self.suffix == ".tmp" and "r" in mode:
            raise OSError("cannot read temp file")
        del args, kwargs
        return original_open(self, mode)

    monkeypatch.setattr(download.requests, "get", fake_get)
    monkeypatch.setattr(Path, "open", fake_open)
    caplog.set_level(logging.ERROR, logger=download.logger.name)

    with pytest.raises(OSError, match="cannot read temp file"):
        download.download_dlib_model(model_path, overwrite=False)

    assert not model_path.exists()
    assert "File system error saving the model" in caplog.text
