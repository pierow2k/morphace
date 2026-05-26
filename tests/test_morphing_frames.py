"""Tests for morph frame generation."""

import io
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from morphace.morphing import frames
from morphace.morphing.config import MorphVideoConfig
from morphace.morphing.frames import (
    _generate_morph_frame,
    _triangle_mask,
    generate_morph_sequence,
)

BLENDED_PIXEL_VALUE = 20


def test_generate_morph_frame_blends_triangle_pixels() -> None:
    """Verify a simple triangle is warped and blended into the output."""
    image1 = np.full((4, 4, 3), 10, dtype=np.float32)
    image2 = np.full((4, 4, 3), 30, dtype=np.float32)
    points = np.array([(0, 0), (3, 0), (0, 3)], dtype=np.float32)

    frame = _generate_morph_frame(
        image1,
        image2,
        points,
        points,
        [(0, 1, 2)],
        alpha=0.5,
        show_triangles=False,
    )

    assert frame.shape == image1.shape
    assert frame[0, 0, 0] == BLENDED_PIXEL_VALUE
    assert frame[3, 3, 0] == 0


def test_triangle_mask_handles_grayscale_images() -> None:
    """Verify triangle masks match grayscale image dimensions."""
    image = np.zeros((4, 4), dtype=np.float32)

    mask = _triangle_mask(image, (0, 0, 4, 4), [(0, 0), (3, 0), (0, 3)])

    assert mask.shape == (4, 4)
    assert mask.dtype == np.float32
    assert mask[0, 0] == 1.0


def test_generate_morph_frame_draws_triangle_overlay() -> None:
    """Verify triangle outlines are drawn when requested."""
    image1 = np.full((4, 4, 3), 10, dtype=np.float32)
    image2 = np.full((4, 4, 3), 30, dtype=np.float32)
    points = np.array([(0, 0), (3, 0), (0, 3)], dtype=np.float32)

    frame = _generate_morph_frame(
        image1,
        image2,
        points,
        points,
        [(0, 1, 2)],
        alpha=0.5,
        show_triangles=True,
    )

    assert np.max(frame) == 255  # noqa:PLR2004


def test_generate_morph_sequence_writes_interpolated_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sequence generation writes every alpha step to the stream."""
    output = io.BytesIO()
    captured: dict[str, object] = {"alphas": [], "show_triangles": []}
    image1 = np.zeros((2, 2, 3), dtype=np.uint8)
    image2 = np.full((2, 2, 3), 100, dtype=np.uint8)
    points = [(0, 0), (1, 0), (0, 1)]
    video_config = MorphVideoConfig(
        duration=1,
        frame_rate=3,
        size=(2, 2),
        output="out.mp4",
    )

    @contextmanager
    def fake_video_writer_context(
        config: MorphVideoConfig,
    ) -> Iterator[tuple[io.BytesIO, int]]:
        """Yield an in-memory output stream."""
        captured["config"] = config
        yield output, 3

    def fake_generate_morph_frame(  # noqa:PLR0913
        img1: np.ndarray,
        img2: np.ndarray,
        p1_arr: np.ndarray,
        p2_arr: np.ndarray,
        tri_list: list[tuple[int, int, int]],
        alpha: float,
        show_triangles: bool,
    ) -> np.ndarray:
        """Capture frame-generation inputs and return a small BGR frame."""
        assert img1.dtype == np.float32
        assert img2.dtype == np.float32
        np.testing.assert_array_equal(p1_arr, np.array(points))
        np.testing.assert_array_equal(p2_arr, np.array(points))
        assert tri_list == [(0, 1, 2)]
        cast_alphas = captured["alphas"]
        assert isinstance(cast_alphas, list)
        cast_alphas.append(alpha)
        cast_show_triangles = captured["show_triangles"]
        assert isinstance(cast_show_triangles, list)
        cast_show_triangles.append(show_triangles)
        return np.full((2, 2, 3), int(alpha * 100), dtype=np.float32)

    monkeypatch.setattr(
        frames,
        "video_writer_context",
        fake_video_writer_context,
    )
    monkeypatch.setattr(
        frames,
        "_generate_morph_frame",
        fake_generate_morph_frame,
    )

    generate_morph_sequence(
        (image1, image2),
        (points, points),
        [(0, 1, 2)],
        video_config,
        show_triangles=True,
    )

    assert captured["config"] == video_config
    assert captured["alphas"] == [0.0, 0.5, 1.0]
    assert captured["show_triangles"] == [True, True, True]
    assert output.getvalue()
