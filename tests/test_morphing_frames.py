"""Tests for morph frame generation."""

import numpy as np

from morphace.morphing.frames import generate_morph_frame

BLENDED_PIXEL_VALUE = 20


def test_generate_morph_frame_blends_triangle_pixels() -> None:
    """Verify a simple triangle is warped and blended into the output."""
    image1 = np.full((4, 4, 3), 10, dtype=np.float32)
    image2 = np.full((4, 4, 3), 30, dtype=np.float32)
    points = np.array([(0, 0), (3, 0), (0, 3)], dtype=np.float32)

    frame = generate_morph_frame(
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
