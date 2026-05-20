"""Tests for the image prep orchestration logic."""

from pathlib import Path
from typing import Any

import pytest

from morphace import prep_images
from morphace.prep_face_alignment import FaceAlignmentOptions


def test_align_faces_uses_shared_model_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify prep alignment uses shared detector and predictor helpers."""
    raw_dir = tmp_path / "raw"
    aligned_dir = tmp_path / "aligned"
    raw_dir.mkdir()
    aligned_dir.mkdir()
    raw_img_path = raw_dir / "face.png"
    raw_img_path.write_bytes(b"not a real image")

    detector = object()
    predictor = object()
    face_landmarks = [(float(i), float(i)) for i in range(68)]
    calls: dict[str, Any] = {}

    def fake_get_detector() -> object:
        """Return a fake shared detector."""
        calls["detector_loaded"] = True
        return detector

    def fake_get_predictor(landmark_model_path: Path) -> object:
        """Return a fake shared predictor."""
        calls["landmark_model_path"] = landmark_model_path
        return predictor

    def fake_get_landmarks(
        image: Path,
        detector: object,
        predictor: object,
    ) -> list[list[tuple[float, float]]]:
        """Return one fake face landmark set."""
        calls["landmark_args"] = (image, detector, predictor)
        return [face_landmarks]

    def fake_align_face_image(
        src_file: Path,
        dst_file: Path,
        face_landmarks: list[tuple[float, float]],
        options: FaceAlignmentOptions,
    ) -> None:
        """Capture the image alignment call."""
        calls["align_face_image_args"] = (
            src_file,
            dst_file,
            face_landmarks,
            options,
        )

    monkeypatch.setattr(prep_images, "get_detector", fake_get_detector)
    monkeypatch.setattr(prep_images, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(prep_images, "get_landmarks", fake_get_landmarks)
    monkeypatch.setattr(
        prep_images,
        "align_face_image",
        fake_align_face_image,
    )

    prep_images.align_faces(
        prep_images.AlignmentConfig(
            raw_dir=raw_dir,
            aligned_dir=aligned_dir,
            landmark_model_path=Path("resolved_model.dat"),
            face_alignment=FaceAlignmentOptions(
                output_size=512,
                x_scale=1.2,
                y_scale=0.9,
                em_scale=0.2,
                alpha=True,
            ),
        )
    )

    assert calls["detector_loaded"] is True
    assert calls["landmark_model_path"] == Path("resolved_model.dat")
    assert calls["landmark_args"] == (raw_img_path, detector, predictor)
    assert calls["align_face_image_args"] == (
        raw_img_path,
        aligned_dir / "face_face01.png",
        face_landmarks,
        FaceAlignmentOptions(
            output_size=512,
            x_scale=1.2,
            y_scale=0.9,
            em_scale=0.2,
            alpha=True,
        ),
    )


def test_alignment_config_uses_default_face_alignment_options(
    tmp_path: Path,
) -> None:
    """Verify image prep defaults to standard face alignment options."""
    config = prep_images.AlignmentConfig(
        raw_dir=tmp_path / "raw",
        aligned_dir=tmp_path / "aligned",
        landmark_model_path=Path("resolved_model.dat"),
    )

    assert config.face_alignment == FaceAlignmentOptions()
