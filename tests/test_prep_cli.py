"""Tests for the image prep command-line interface."""

from pathlib import Path

import pytest

from morphace import prep_cli
from morphace.models import LandmarkModelNotFoundError
from morphace.prep_images import AlignmentConfig


def test_main_builds_alignment_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prep CLI passes parsed arguments to align_faces."""
    captured_configs: list[AlignmentConfig] = []

    def fake_align_faces(config: AlignmentConfig) -> None:
        """Capture the parsed prep configuration."""
        captured_configs.append(config)

    def fake_resolve_landmark_model_path(
        model_path: Path | None,
    ) -> Path:
        """Return a resolved model path without touching the filesystem."""
        assert model_path == Path("shape_predictor.dat")
        return Path("resolved_shape_predictor.dat")

    monkeypatch.setattr(prep_cli, "align_faces", fake_align_faces)
    monkeypatch.setattr(
        prep_cli,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )

    result = prep_cli.main(
        [
            "raw",
            "aligned",
            "--landmark-model",
            "shape_predictor.dat",
            "--output_size",
            "512",
            "--x_scale",
            "1.2",
            "--y_scale",
            "0.9",
            "--em_scale",
            "0.2",
            "--use_alpha",
            "True",
        ]
    )

    assert result == 0
    assert captured_configs == [
        AlignmentConfig(
            raw_dir=Path("raw"),
            aligned_dir=Path("aligned"),
            landmark_model_path=Path("resolved_shape_predictor.dat"),
            output_size=512,
            x_scale=1.2,
            y_scale=0.9,
            em_scale=0.2,
            use_alpha=True,
        )
    ]


def test_main_returns_error_when_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prep CLI returns error before alignment if model is absent."""
    called_align_faces = False

    def fake_align_faces(config: AlignmentConfig) -> None:
        """Fail if the prep pipeline is unexpectedly called."""
        nonlocal called_align_faces
        called_align_faces = True
        del config

    def fake_resolve_landmark_model_path(
        model_path: Path | None,
    ) -> Path:
        """Raise the same model-resolution error used by the morph CLI."""
        del model_path
        raise LandmarkModelNotFoundError("missing model")

    monkeypatch.setattr(prep_cli, "align_faces", fake_align_faces)
    monkeypatch.setattr(
        prep_cli,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )

    result = prep_cli.main(["raw", "aligned"])

    assert result == 1
    assert not called_align_faces
