"""Tests for the image prep command-line interface."""

from pathlib import Path

import pytest

from face_morphing import prep_cli
from face_morphing.prep_images import AlignmentConfig


def test_main_builds_alignment_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prep CLI passes parsed arguments to align_faces."""
    captured_configs: list[AlignmentConfig] = []

    def fake_align_faces(config: AlignmentConfig) -> None:
        """Capture the parsed prep configuration."""
        captured_configs.append(config)

    monkeypatch.setattr(prep_cli, "align_faces", fake_align_faces)

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
            landmark_model_path=Path("shape_predictor.dat"),
            output_size=512,
            x_scale=1.2,
            y_scale=0.9,
            em_scale=0.2,
            use_alpha=True,
        )
    ]
