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


def test_main_builds_morph_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI arguments are converted into morph workflow config."""
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

    monkeypatch.setattr(morph.cv2, "imread", fake_imread)
    monkeypatch.setattr(
        morph,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )
    monkeypatch.setattr(morph, "morph_faces", fake_morph_faces)

    result = morph.main(
        [
            "first.png",
            "second.png",
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
    assert captured["paths"] == ["first.png", "second.png"]
    assert captured["workflow"][2] == MorphConfig(
        duration=7,
        frame_rate=24,
        output="out.mp4",
        landmark_model_path=Path("resolved_shape_predictor.dat"),
    )
    assert captured["workflow"][3] is True


@pytest.mark.parametrize("args", [["--duration", "0"], ["--fps", "0"]])
def test_main_rejects_non_positive_timing(args: list[str]) -> None:
    """Verify invalid duration or FPS exits before image processing."""
    result = morph.main(["first.png", "second.png", *args])

    assert result == 1


def test_main_returns_error_for_unreadable_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unreadable input images return an error."""

    def fake_imread(path: str) -> np.ndarray | None:
        """Fail to read the second image."""
        if path == "second.png":
            return None
        return _image()

    monkeypatch.setattr(morph.cv2, "imread", fake_imread)

    result = morph.main(["first.png", "second.png"])

    assert result == 1


def test_main_returns_error_when_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify missing model resolution stops before morphing."""
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

    monkeypatch.setattr(morph.cv2, "imread", fake_imread)
    monkeypatch.setattr(
        morph,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )
    monkeypatch.setattr(morph, "morph_faces", fake_morph_faces)

    result = morph.main(["first.png", "second.png"])

    assert result == 1
    assert not called_morph_faces


def test_main_returns_error_when_no_face_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify no-face workflow errors are converted into CLI errors."""

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

    result = morph.main(["first.png", "second.png"])

    assert result == 1
