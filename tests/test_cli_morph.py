"""Tests for the face morphing command-line interface."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from morphace.cli import morph
from morphace.landmarks import LandmarkModelNotFoundError, NoFaceFoundError
from morphace.morphing import MorphConfig


def _image() -> np.ndarray:
    """Return a small fake OpenCV image."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_main_builds_morph_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify CLI arguments are converted into morph workflow config."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")
    captured: dict[str, Any] = {}

    def fake_imread(path: str) -> np.ndarray:
        """Return a fake image for any path."""
        captured.setdefault("paths", []).append(path)
        return _image()

    def fake_resolve_landmark_model_path(model_path: Path | None) -> Path:
        """Return a fake resolved landmark model path."""
        assert model_path == Path("shape_predictor.dat")
        return Path("resolved_shape_predictor.dat")

    def fake_morph_faces(
        image1: np.ndarray,
        image2: np.ndarray,
        config: MorphConfig,
        show_triangles: bool = False,
    ) -> Path:
        """Capture the workflow call."""
        captured["workflow"] = (image1, image2, config, show_triangles)
        return Path(config.output)

    monkeypatch.setattr(morph.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(morph.cv2, "imread", fake_imread)
    monkeypatch.setattr(
        morph,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )
    monkeypatch.setattr(morph, "morph_faces", fake_morph_faces)

    result = morph.main(
        [
            str(img1),
            str(img2),
            "--landmark-model",
            "shape_predictor.dat",
            "--output",
            "out.mp4",
            "--duration",
            "7",
            "--fps",
            "24",
            "--show-mesh",
        ]
    )

    assert result == 0
    assert captured["paths"] == [str(img1), str(img2)]
    assert captured["workflow"][2] == MorphConfig(
        duration=7,
        frame_rate=24,
        output="out.mp4",
        landmark_model_path=Path("resolved_shape_predictor.dat"),
    )
    assert captured["workflow"][3] is True


def test_main_sets_verbose_ffmpeg_loglevel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the CLI can request FFmpeg informational output."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")
    captured: dict[str, Any] = {}

    def fake_imread(path: str) -> np.ndarray:
        """Return a fake image."""
        del path
        return _image()

    def fake_resolve_landmark_model_path(model_path: Path | None) -> Path:
        """Return a fake resolved landmark model path."""
        del model_path
        return Path("resolved_shape_predictor.dat")

    def fake_morph_faces(
        image1: np.ndarray,
        image2: np.ndarray,
        config: MorphConfig,
        show_triangles: bool = False,
    ) -> Path:
        """Capture the workflow call."""
        del image1, image2, show_triangles
        captured["config"] = config
        return Path(config.output)

    monkeypatch.setattr(morph.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(morph.cv2, "imread", fake_imread)
    monkeypatch.setattr(
        morph,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )
    monkeypatch.setattr(morph, "morph_faces", fake_morph_faces)

    result = morph.main([str(img1), str(img2), "--show-ffmpeg-output"])

    assert result == 0
    assert captured["config"].ffmpeg_loglevel == "info"


@pytest.mark.parametrize("args", [["--duration", "0"], ["--fps", "0"]])
def test_main_rejects_non_positive_timing(
    args: list[str],
    tmp_path: Path,
) -> None:
    """Verify invalid duration or FPS exits before image processing."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")

    result = morph.main([str(img1), str(img2), *args])

    assert result == 1


def test_main_returns_error_for_unreadable_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify unreadable input images return an error."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")

    def fake_imread(path: str) -> np.ndarray | None:
        """Fail to read the second image."""
        if path == str(img2):
            return None
        return _image()

    monkeypatch.setattr(morph.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(morph.cv2, "imread", fake_imread)

    result = morph.main([str(img1), str(img2)])

    assert result == 1


def test_main_returns_error_when_ffmpeg_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify morph CLI returns error if ffmpeg is not found."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")

    monkeypatch.setattr(morph.shutil, "which", lambda _: None)

    result = morph.main([str(img1), str(img2)])

    assert result == 1


def test_main_returns_error_when_source_images_do_not_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify morph CLI returns error if input images do not exist."""
    img1 = tmp_path / "nonexistent1.png"
    img2 = tmp_path / "nonexistent2.png"

    monkeypatch.setattr(morph.shutil, "which", lambda _: "ffmpeg")

    result = morph.main([str(img1), str(img2)])

    assert result == 1


def test_main_returns_error_when_second_source_image_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify morph CLI returns error if the second image does not exist."""
    img1 = tmp_path / "first.png"
    img1.write_bytes(b"")
    img2 = tmp_path / "nonexistent2.png"

    monkeypatch.setattr(morph.shutil, "which", lambda _: "ffmpeg")

    result = morph.main([str(img1), str(img2)])

    assert result == 1


def test_main_returns_error_for_unreadable_first_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the first unreadable input image returns an error."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")

    def fake_imread(path: str) -> np.ndarray | None:
        """Fail to read the first image."""
        if path == str(img1):
            return None
        return _image()

    monkeypatch.setattr(morph.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(morph.cv2, "imread", fake_imread)

    result = morph.main([str(img1), str(img2)])

    assert result == 1


def test_main_returns_error_when_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify missing model resolution stops before morphing."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")
    called_morph_faces = False

    def fake_resolve_landmark_model_path(model_path: Path | None) -> Path:
        """Raise the model resolution error."""
        del model_path
        raise LandmarkModelNotFoundError("missing model")

    def fake_morph_faces(*args: object, **kwargs: object) -> Path:
        """Fail if the workflow is unexpectedly called."""
        nonlocal called_morph_faces
        called_morph_faces = True
        del args, kwargs
        return Path("out.mp4")

    def fake_imread(path: str) -> np.ndarray:
        """Return a fake image."""
        del path
        return _image()

    monkeypatch.setattr(morph.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(morph.cv2, "imread", fake_imread)
    monkeypatch.setattr(
        morph,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )
    monkeypatch.setattr(morph, "morph_faces", fake_morph_faces)

    result = morph.main([str(img1), str(img2)])

    assert result == 1
    assert not called_morph_faces


def test_main_returns_error_when_no_face_is_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify no-face workflow errors are converted into CLI errors."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")

    def fake_morph_faces(*args: object, **kwargs: object) -> Path:
        """Raise the public no-face error."""
        del args, kwargs
        raise NoFaceFoundError

    def fake_imread(path: str) -> np.ndarray:
        """Return a fake image."""
        del path
        return _image()

    def fake_resolve_landmark_model_path(model_path: Path | None) -> Path:
        """Return a resolved model path."""
        del model_path
        return Path("resolved_shape_predictor.dat")

    monkeypatch.setattr(morph.cv2, "imread", fake_imread)
    monkeypatch.setattr(
        morph,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )
    monkeypatch.setattr(morph, "morph_faces", fake_morph_faces)

    result = morph.main([str(img1), str(img2)])

    assert result == 1


def test_main_returns_error_for_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify unexpected runtime workflow failures become CLI errors."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")

    def fake_morph_faces(*args: object, **kwargs: object) -> Path:
        """Raise an unexpected runtime error."""
        del args, kwargs
        raise RuntimeError("ffmpeg failed")

    def fake_imread(path: str) -> np.ndarray:
        """Return a fake image."""
        del path
        return _image()

    def fake_resolve_landmark_model_path(model_path: Path | None) -> Path:
        """Return a resolved model path."""
        del model_path
        return Path("resolved_shape_predictor.dat")

    monkeypatch.setattr(morph.cv2, "imread", fake_imread)
    monkeypatch.setattr(
        morph,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )
    monkeypatch.setattr(morph, "morph_faces", fake_morph_faces)

    result = morph.main([str(img1), str(img2)])

    assert result == 1


def test_main_appends_mp4_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the CLI appends .mp4 to the output if missing."""
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"")
    img2.write_bytes(b"")
    captured_output: str = ""

    def fake_morph_faces(  # pylint: disable=invalid-name
        _img1: np.ndarray,
        _img2: np.ndarray,
        config: MorphConfig,
        _show: bool = False,
    ) -> Path:
        nonlocal captured_output
        captured_output = config.output
        return Path(config.output)

    monkeypatch.setattr(morph.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(morph.cv2, "imread", lambda _: _image())
    monkeypatch.setattr(
        morph, "resolve_landmark_model_path", lambda _: Path("r.dat")
    )
    monkeypatch.setattr(morph, "morph_faces", fake_morph_faces)

    morph.main([str(img1), str(img2), "--output", "video"])
    assert captured_output == "video.mp4"

    morph.main([str(img1), str(img2), "--output", "movie.mp4"])
    assert captured_output == "movie.mp4"
