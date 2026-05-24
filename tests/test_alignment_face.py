"""Tests for single-image face alignment orchestration."""

from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import pytest

from morphace.alignment import FaceAlignmentOptions, face


def test_align_face_image_returns_when_source_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify missing source images stop before geometry calculation."""
    called_calculate_quad = False

    def fake_calculate_alignment_quad(
        face_landmarks: list[tuple[float, float]],
        options: FaceAlignmentOptions,
    ) -> tuple[np.ndarray, float]:
        """Fail if a missing image still triggers geometry work."""
        nonlocal called_calculate_quad
        called_calculate_quad = True
        del face_landmarks, options
        return np.zeros((4, 2)), 0.0

    monkeypatch.setattr(
        face,
        "calculate_alignment_quad",
        fake_calculate_alignment_quad,
    )

    face.align_face_image(
        tmp_path / "missing.png",
        tmp_path / "aligned.png",
        [],
    )

    assert not called_calculate_quad
    assert not (tmp_path / "aligned.png").exists()


def test_align_face_image_saves_warped_face(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the alignment pipeline passes images through each stage."""
    src_file = tmp_path / "source.png"
    dst_file = tmp_path / "aligned.png"
    PIL.Image.new("RGBA", (8, 8), color=(1, 2, 3, 255)).save(src_file)
    options = FaceAlignmentOptions(output_size=4, transform_size=4)
    face_landmarks = [(float(index), float(index)) for index in range(68)]
    quad = np.array(
        [[0.0, 0.0], [0.0, 7.0], [7.0, 7.0], [7.0, 0.0]],
        dtype=np.float32,
    )
    prepared_quad = quad + 1.0
    prepared_image = PIL.Image.new("RGB", (8, 8), color=(4, 5, 6))
    warped_image = PIL.Image.new("RGB", (4, 4), color=(7, 8, 9))
    captured: dict[str, Any] = {}

    def fake_calculate_alignment_quad(
        landmarks: list[tuple[float, float]],
        alignment_options: FaceAlignmentOptions,
    ) -> tuple[np.ndarray, float]:
        """Capture geometry inputs and return a simple quad."""
        captured["geometry"] = (landmarks, alignment_options)
        return quad, 8.0

    def fake_prepare_alignment_canvas(
        image: PIL.Image.Image,
        input_quad: np.ndarray,
        crop_size: float,
        alignment_options: FaceAlignmentOptions,
    ) -> tuple[PIL.Image.Image, np.ndarray]:
        """Capture canvas preparation inputs."""
        captured["canvas"] = (
            image.mode,
            input_quad.copy(),
            crop_size,
            alignment_options,
        )
        return prepared_image, prepared_quad

    def fake_warp_aligned_face(
        image: PIL.Image.Image,
        input_quad: np.ndarray,
        alignment_options: FaceAlignmentOptions,
    ) -> PIL.Image.Image:
        """Capture warp inputs and return a saveable image."""
        captured["warp"] = (image, input_quad.copy(), alignment_options)
        return warped_image

    monkeypatch.setattr(
        face,
        "calculate_alignment_quad",
        fake_calculate_alignment_quad,
    )
    monkeypatch.setattr(
        face,
        "prepare_alignment_canvas",
        fake_prepare_alignment_canvas,
    )
    monkeypatch.setattr(face, "warp_aligned_face", fake_warp_aligned_face)

    face.align_face_image(src_file, dst_file, face_landmarks, options)

    assert captured["geometry"] == (face_landmarks, options)
    assert captured["canvas"][0] == "RGB"
    np.testing.assert_array_equal(captured["canvas"][1], quad)
    assert captured["canvas"][2:] == (8.0, options)
    assert captured["warp"][0] is prepared_image
    np.testing.assert_array_equal(captured["warp"][1], prepared_quad)
    assert captured["warp"][2] is options
    assert PIL.Image.open(dst_file).size == (4, 4)
