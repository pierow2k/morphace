"""Tests for the image prep orchestration logic."""

from pathlib import Path
from typing import Any

import pytest

from morphace.alignment import FaceAlignmentOptions, batch


def test_align_faces_uses_shared_model_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify prep alignment uses shared detector and predictor helpers."""
    source_dir = tmp_path / "raw"
    aligned_dir = tmp_path / "aligned"
    source_dir.mkdir()
    aligned_dir.mkdir()
    raw_img_path = source_dir / "face.png"
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

    monkeypatch.setattr(batch, "get_detector", fake_get_detector)
    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(batch, "get_landmarks", fake_get_landmarks)
    monkeypatch.setattr(
        batch,
        "align_face_image",
        fake_align_face_image,
    )

    batch.align_faces(
        batch.AlignmentConfig(
            source=source_dir,
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
    config = batch.AlignmentConfig(
        source=tmp_path / "raw",
        aligned_dir=tmp_path / "aligned",
        landmark_model_path=Path("resolved_model.dat"),
    )

    assert config.face_alignment == FaceAlignmentOptions()


def test_align_faces_skips_existing_output_without_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify existing first-face output skips landmark detection."""
    source_dir = tmp_path / "raw"
    aligned_dir = tmp_path / "aligned"
    source_dir.mkdir()
    aligned_dir.mkdir()
    raw_img_path = source_dir / "face.png"
    raw_img_path.write_bytes(b"not a real image")
    (aligned_dir / "face_face01.png").write_bytes(b"existing")
    called_get_landmarks = False

    def fake_get_landmarks(*args: object, **kwargs: object) -> list[object]:
        """Fail if landmark detection is unexpectedly called."""
        nonlocal called_get_landmarks
        called_get_landmarks = True
        del args, kwargs
        return []

    def fake_get_detector_for_skip() -> object:
        """Return a fake detector."""
        return object()

    def fake_get_predictor_for_skip(model_path: Path) -> object:
        """Return a fake predictor."""
        del model_path
        return object()

    monkeypatch.setattr(batch, "get_detector", fake_get_detector_for_skip)
    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor_for_skip)
    monkeypatch.setattr(batch, "get_landmarks", fake_get_landmarks)

    batch.align_faces(
        batch.AlignmentConfig(
            source=source_dir,
            aligned_dir=aligned_dir,
            landmark_model_path=Path("resolved_model.dat"),
        )
    )

    assert not called_get_landmarks


def test_align_faces_ignores_directories_and_non_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify only recognized image files are sent to detection."""
    source_dir = tmp_path / "raw"
    aligned_dir = tmp_path / "aligned"
    source_dir.mkdir()
    (source_dir / "nested").mkdir()
    (source_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    image_path = source_dir / "face.JPEG"
    image_path.write_bytes(b"not a real image")
    processed: list[Path] = []

    def fake_get_detector() -> object:
        """Return a fake detector."""
        return object()

    def fake_get_predictor(model_path: Path) -> object:
        """Return a fake predictor."""
        del model_path
        return object()

    def fake_align_detected_faces(
        config: batch.AlignmentConfig,
        raw_img_path: Path,
        detector: object,
        predictor: object,
    ) -> None:
        """Capture files selected for alignment."""
        del config, detector, predictor
        processed.append(raw_img_path)

    monkeypatch.setattr(batch, "get_detector", fake_get_detector)
    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(
        batch,
        "_align_detected_faces",
        fake_align_detected_faces,
    )

    batch.align_faces(
        batch.AlignmentConfig(
            source=source_dir,
            aligned_dir=aligned_dir,
            landmark_model_path=Path("resolved_model.dat"),
        )
    )

    assert processed == [image_path]


def test_align_faces_processes_single_image_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify a file source is sent directly to alignment."""
    image_path = tmp_path / "face.png"
    aligned_dir = tmp_path / "aligned"
    image_path.write_bytes(b"not a real image")
    processed: list[Path] = []

    def fake_get_detector() -> object:
        """Return a fake detector."""
        return object()

    def fake_get_predictor(model_path: Path) -> object:
        """Return a fake predictor."""
        del model_path
        return object()

    def fake_align_detected_faces(
        config: batch.AlignmentConfig,
        raw_img_path: Path,
        detector: object,
        predictor: object,
    ) -> None:
        """Capture files selected for alignment."""
        del config, detector, predictor
        processed.append(raw_img_path)

    monkeypatch.setattr(batch, "get_detector", fake_get_detector)
    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(
        batch,
        "_align_detected_faces",
        fake_align_detected_faces,
    )

    batch.align_faces(
        batch.AlignmentConfig(
            source=image_path,
            aligned_dir=aligned_dir,
            landmark_model_path=Path("resolved_model.dat"),
        )
    )

    assert processed == [image_path]


def test_align_faces_ignores_single_non_image_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify a non-image file source is not sent to alignment."""
    source_file = tmp_path / "notes.txt"
    aligned_dir = tmp_path / "aligned"
    source_file.write_text("not an image", encoding="utf-8")
    processed: list[Path] = []

    def fake_get_detector() -> object:
        """Return a fake detector."""
        return object()

    def fake_get_predictor(model_path: Path) -> object:
        """Return a fake predictor."""
        del model_path
        return object()

    def fake_align_detected_faces(
        config: batch.AlignmentConfig,
        raw_img_path: Path,
        detector: object,
        predictor: object,
    ) -> None:
        """Capture files selected for alignment."""
        del config, detector, predictor
        processed.append(raw_img_path)

    monkeypatch.setattr(batch, "get_detector", fake_get_detector)
    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(
        batch,
        "_align_detected_faces",
        fake_align_detected_faces,
    )

    batch.align_faces(
        batch.AlignmentConfig(
            source=source_file,
            aligned_dir=aligned_dir,
            landmark_model_path=Path("resolved_model.dat"),
        )
    )

    assert not processed


def test_align_faces_rejects_invalid_source_before_loading_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify invalid source paths fail before model loading."""
    missing_source = tmp_path / "missing"
    aligned_dir = tmp_path / "aligned"
    called_get_detector = False

    def fake_get_detector() -> object:
        """Fail if model helpers are unexpectedly called."""
        nonlocal called_get_detector
        called_get_detector = True
        return object()

    monkeypatch.setattr(batch, "get_detector", fake_get_detector)

    with pytest.raises(
        ValueError,
        match="Source path does not exist or is not accessible",
    ):
        batch.align_faces(
            batch.AlignmentConfig(
                source=missing_source,
                aligned_dir=aligned_dir,
                landmark_model_path=Path("resolved_model.dat"),
            )
        )

    assert not called_get_detector
    assert not aligned_dir.exists()


def test_align_faces_skips_existing_single_image_without_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify skip behavior applies to a single file source."""
    image_path = tmp_path / "face.png"
    aligned_dir = tmp_path / "aligned"
    image_path.write_bytes(b"not a real image")
    aligned_dir.mkdir()
    (aligned_dir / "face_face01.png").write_bytes(b"existing")
    called_get_landmarks = False

    def fake_get_landmarks(*args: object, **kwargs: object) -> list[object]:
        """Fail if landmark detection is unexpectedly called."""
        nonlocal called_get_landmarks
        called_get_landmarks = True
        del args, kwargs
        return []

    def fake_get_detector() -> object:
        """Return a fake detector."""
        return object()

    def fake_get_predictor(model_path: Path) -> object:
        """Return a fake predictor."""
        del model_path
        return object()

    monkeypatch.setattr(batch, "get_detector", fake_get_detector)
    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(batch, "get_landmarks", fake_get_landmarks)

    batch.align_faces(
        batch.AlignmentConfig(
            source=image_path,
            aligned_dir=aligned_dir,
            landmark_model_path=Path("resolved_model.dat"),
        )
    )

    assert not called_get_landmarks


def test_align_detected_faces_logs_alignment_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify a single face alignment failure does not stop the loop."""
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    raw_img_path = source_dir / "face.png"
    raw_img_path.write_bytes(b"not a real image")
    aligned_dir = tmp_path / "aligned"
    face_landmarks = [(float(index), float(index)) for index in range(68)]
    attempts: list[Path] = []

    def fake_get_detector() -> object:
        """Return a fake detector."""
        return object()

    def fake_get_predictor(model_path: Path) -> object:
        """Return a fake predictor."""
        del model_path
        return object()

    def fake_get_landmarks(
        image: Path,
        detector: object,
        predictor: object,
    ) -> list[list[tuple[float, float]]]:
        """Return two fake faces for alignment."""
        del image, detector, predictor
        return [face_landmarks, face_landmarks]

    def fake_align_face_image(
        src_file: Path,
        dst_file: Path,
        landmarks: list[tuple[float, float]],
        options: FaceAlignmentOptions,
    ) -> None:
        """Raise for one face, then allow the next one."""
        del src_file, landmarks, options
        attempts.append(dst_file)
        if len(attempts) == 1:
            raise ValueError("cannot align")

    monkeypatch.setattr(batch, "get_detector", fake_get_detector)
    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(batch, "get_landmarks", fake_get_landmarks)
    monkeypatch.setattr(batch, "align_face_image", fake_align_face_image)

    batch.align_faces(
        batch.AlignmentConfig(
            source=source_dir,
            aligned_dir=aligned_dir,
            landmark_model_path=Path("resolved_model.dat"),
        )
    )

    assert attempts == [
        aligned_dir / "face_face01.png",
        aligned_dir / "face_face02.png",
    ]


def test_align_faces_logs_landmark_detection_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify one image-level failure does not stop batch processing."""
    source_dir = tmp_path / "raw"
    aligned_dir = tmp_path / "aligned"
    source_dir.mkdir()
    first_image = source_dir / "first.png"
    second_image = source_dir / "second.png"
    first_image.write_bytes(b"not a real image")
    second_image.write_bytes(b"not a real image")
    processed: list[Path] = []

    def fake_get_detector() -> object:
        """Return a fake detector."""
        return object()

    def fake_get_predictor(model_path: Path) -> object:
        """Return a fake predictor."""
        del model_path
        return object()

    def fake_align_detected_faces(
        config: batch.AlignmentConfig,
        raw_img_path: Path,
        detector: object,
        predictor: object,
    ) -> None:
        """Raise for the first image and capture the second."""
        del config, detector, predictor
        if raw_img_path == first_image:
            raise RuntimeError("detection failed")
        processed.append(raw_img_path)

    monkeypatch.setattr(batch, "get_detector", fake_get_detector)
    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(
        batch,
        "_align_detected_faces",
        fake_align_detected_faces,
    )

    batch.align_faces(
        batch.AlignmentConfig(
            source=source_dir,
            aligned_dir=aligned_dir,
            landmark_model_path=Path("resolved_model.dat"),
        )
    )

    assert processed == [second_image]
