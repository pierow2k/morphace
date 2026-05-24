"""Tests for the image prep command-line interface."""

from pathlib import Path

import pytest

from morphace.alignment import AlignmentConfig, FaceAlignmentOptions
from morphace.cli import prep
from morphace.landmarks import LandmarkModelNotFoundError


def test_main_builds_alignment_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prep CLI passes parsed arguments to align_faces."""
    captured_configs: list[AlignmentConfig] = []

    def fake_align_faces(
        config: AlignmentConfig,
        overwrite: bool = False,
    ) -> None:
        """Capture the parsed prep configuration."""
        assert overwrite is False
        captured_configs.append(config)

    def fake_resolve_landmark_model_path(
        model_path: Path | None,
    ) -> Path:
        """Return a resolved model path without touching the filesystem."""
        assert model_path == Path("shape_predictor.dat")
        return Path("resolved_shape_predictor.dat")

    monkeypatch.setattr(prep, "align_faces", fake_align_faces)
    monkeypatch.setattr(
        prep,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )

    result = prep.main(
        [
            "raw",
            "aligned",
            "--landmark-model",
            "shape_predictor.dat",
            "--output-size",
            "512",
            "--x-scale",
            "1.2",
            "--y-scale",
            "0.9",
            "--em-scale",
            "0.2",
            "--alpha",
        ]
    )

    assert result == 0
    assert captured_configs == [
        AlignmentConfig(
            source=Path("raw"),
            aligned_dir=Path("aligned"),
            landmark_model_path=Path("resolved_shape_predictor.dat"),
            face_alignment=FaceAlignmentOptions(
                output_size=512,
                x_scale=1.2,
                y_scale=0.9,
                em_scale=0.2,
                alpha=True,
            ),
        )
    ]


def test_main_defaults_aligned_dir_to_raw_cropped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify prep CLI defaults directory source output to cropped."""
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    captured_configs: list[AlignmentConfig] = []

    def fake_align_faces(
        config: AlignmentConfig,
        overwrite: bool = False,
    ) -> None:
        """Capture the defaulted prep configuration."""
        assert overwrite is True
        captured_configs.append(config)

    def fake_resolve_landmark_model_path(
        model_path: Path | None,
    ) -> Path:
        """Return a resolved model path without filesystem access."""
        assert model_path is None
        return Path("resolved_shape_predictor.dat")

    monkeypatch.setattr(prep, "align_faces", fake_align_faces)
    monkeypatch.setattr(
        prep,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )

    result = prep.main([str(source_dir), "--overwrite"])

    assert result == 0
    assert captured_configs[0].aligned_dir == source_dir / "cropped"


def test_main_defaults_aligned_dir_to_image_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify prep CLI defaults file source output to its parent."""
    source_file = tmp_path / "face.png"
    source_file.write_bytes(b"not a real image")
    captured_configs: list[AlignmentConfig] = []

    def fake_align_faces(
        config: AlignmentConfig,
        overwrite: bool = False,
    ) -> None:
        """Capture the defaulted prep configuration."""
        assert overwrite is False
        captured_configs.append(config)

    def fake_resolve_landmark_model_path(
        model_path: Path | None,
    ) -> Path:
        """Return a resolved model path without filesystem access."""
        assert model_path is None
        return Path("resolved_shape_predictor.dat")

    monkeypatch.setattr(prep, "align_faces", fake_align_faces)
    monkeypatch.setattr(
        prep,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )

    result = prep.main([str(source_file)])

    assert result == 0
    assert captured_configs[0].source == source_file
    assert captured_configs[0].aligned_dir == tmp_path


def test_main_uses_explicit_aligned_dir_for_file_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify explicit prep output wins for a file source."""
    source_file = tmp_path / "face.png"
    aligned_dir = tmp_path / "aligned"
    source_file.write_bytes(b"not a real image")
    captured_configs: list[AlignmentConfig] = []

    def fake_align_faces(
        config: AlignmentConfig,
        overwrite: bool = False,
    ) -> None:
        """Capture the parsed prep configuration."""
        assert overwrite is False
        captured_configs.append(config)

    def fake_resolve_landmark_model_path(
        model_path: Path | None,
    ) -> Path:
        """Return a resolved model path without filesystem access."""
        assert model_path is None
        return Path("resolved_shape_predictor.dat")

    monkeypatch.setattr(prep, "align_faces", fake_align_faces)
    monkeypatch.setattr(
        prep,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )

    result = prep.main([str(source_file), str(aligned_dir)])

    assert result == 0
    assert captured_configs[0].source == source_file
    assert captured_configs[0].aligned_dir == aligned_dir


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

    monkeypatch.setattr(prep, "align_faces", fake_align_faces)
    monkeypatch.setattr(
        prep,
        "resolve_landmark_model_path",
        fake_resolve_landmark_model_path,
    )

    result = prep.main(["raw", "aligned"])

    assert result == 1
    assert not called_align_faces
