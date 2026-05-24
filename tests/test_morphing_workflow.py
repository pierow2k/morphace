"""Tests for high-level morphing workflow orchestration."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from morphace.morphing import FaceCorrespondences, MorphConfig, workflow
from morphace.morphing.config import MorphVideoConfig


def test_morph_faces_uses_named_correspondence_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify workflow passes named correspondence data to later stages."""
    image1 = np.zeros((4, 4, 3), dtype=np.uint8)
    image2 = np.ones((4, 4, 3), dtype=np.uint8)
    average_landmarks = np.array([(0, 0), (3, 0), (0, 3)], dtype=np.float32)
    correspondences = FaceCorrespondences(
        size=(4, 4),
        image1=image1,
        image2=image2,
        points1=[(0, 0), (3, 0), (0, 3)],
        points2=[(0, 0), (3, 0), (0, 3)],
        average_landmarks=average_landmarks,
    )
    captured: dict[str, Any] = {}

    def fake_get_detector() -> object:
        """Return a fake detector."""
        return object()

    def fake_get_predictor(model_path: Path) -> object:
        """Return a fake predictor."""
        del model_path
        return object()

    def fake_align_faces(*args: object) -> FaceCorrespondences:
        """Return fake correspondences."""
        del args
        return correspondences

    monkeypatch.setattr(workflow, "get_detector", fake_get_detector)
    monkeypatch.setattr(workflow, "get_predictor", fake_get_predictor)
    monkeypatch.setattr(
        workflow,
        "align_faces",
        fake_align_faces,
    )

    def fake_compute_delaunay_triangles(
        width: int,
        height: int,
        landmarks: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        """Capture triangulation inputs."""
        captured["triangulation"] = (width, height, landmarks)
        return [(0, 1, 2)]

    def fake_generate_morph_sequence(
        img_pair: tuple[np.ndarray, np.ndarray],
        points_pair: tuple[list[tuple[int, int]], list[tuple[int, int]]],
        tri_list: list[tuple[int, int, int]],
        video_config: MorphVideoConfig,
        show_triangles: bool,
    ) -> None:
        """Capture frame generation inputs."""
        captured["sequence"] = (
            img_pair,
            points_pair,
            tri_list,
            video_config,
            show_triangles,
        )

    monkeypatch.setattr(
        workflow,
        "compute_delaunay_triangles",
        fake_compute_delaunay_triangles,
    )
    monkeypatch.setattr(
        workflow,
        "generate_morph_sequence",
        fake_generate_morph_sequence,
    )

    output = workflow.morph_faces(
        image1,
        image2,
        MorphConfig(
            duration=2,
            frame_rate=10,
            output="out.mp4",
            landmark_model_path=Path("model.dat"),
            ffmpeg_loglevel="info",
        ),
        show_triangles=True,
    )

    assert output == Path("out.mp4")
    triangulation = captured["triangulation"]
    assert triangulation[:2] == (4, 4)
    assert triangulation[2] is average_landmarks
    sequence = captured["sequence"]
    assert sequence[0][0] is image1
    assert sequence[0][1] is image2
    assert sequence[1] == (correspondences.points1, correspondences.points2)
    assert sequence[2] == [(0, 1, 2)]
    assert sequence[3] == MorphVideoConfig(
        duration=2,
        frame_rate=10,
        size=(4, 4),
        output="out.mp4",
        ffmpeg_loglevel="info",
    )
    assert sequence[4] is True
