"""Tests for triangulation geometry helpers."""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from morphace.morphing.triangulation import (
    Bounds,
    Point,
    _extract_triangle_indices,
    _is_within_bounds,
    compute_delaunay_triangles,
)

TRIANGLES_IN_SQUARE = 2
POINTS_IN_TRIANGLE = 3


@pytest.mark.parametrize(
    ("bounds", "point", "expected"),
    [
        ((0, 0, 100, 100), (50, 50), True),  # Center
        ((0, 0, 100, 100), (0, 0), True),  # Top-left corner
        ((0, 0, 100, 100), (100, 100), True),  # Bottom-right corner
        ((0, 0, 100, 100), (0, 50), True),  # Left edge
        ((0, 0, 100, 100), (100, 50), True),  # Right edge
        ((0, 0, 100, 100), (-1, 50), False),  # Outside left
        ((0, 0, 100, 100), (101, 50), False),  # Outside right
        ((0, 0, 100, 100), (50, -1), False),  # Outside top
        ((0, 0, 100, 100), (50, 101), False),  # Outside bottom
    ],
)
def test_is_within_bounds(
    bounds: Bounds,
    point: Point,
    expected: bool,
) -> None:
    """Verify that _is_within_bounds correctly identifies points in range."""
    assert _is_within_bounds(bounds, point) is expected


def test_extract_triangle_indices_filters_invalid_triangles() -> None:
    """Verify extraction filters out-of-bounds and virtual points."""
    subdiv = MagicMock(spec=cv2.Subdiv2D)

    # Mock three triangles:
    # 1. Valid triangle
    # 2. Out of bounds triangle
    # 3. Triangle with points missing from the index map (virtual points)
    subdiv.getTriangleList.return_value = np.array(
        [
            [10, 10, 20, 10, 10, 20],  # Valid
            [10, 10, 110, 10, 10, 20],  # Out of bounds (width 100)
            [5, 5, 15, 5, 5, 15],  # Points missing from map
        ],
        dtype=np.float32,
    )

    point_indices = {
        (10, 10): 0,
        (20, 10): 1,
        (10, 20): 2,
        (110, 10): 3,  # Point exists, but triangle will fail bounds check
    }

    result = _extract_triangle_indices(100, 100, subdiv, point_indices)

    assert result == [(0, 1, 2)]


def test_compute_delaunay_triangles_basic() -> None:
    """Verify full triangulation pipeline with a simple square."""
    width, height = 100, 100
    # 4 points forming a square (as floats to test sanitization)
    landmarks = np.array(
        [
            [10.1, 10.9],  # -> (10, 10) idx 0
            [90.0, 10.0],  # -> (90, 10) idx 1
            [10.0, 90.0],  # -> (10, 90) idx 2
            [90.0, 90.0],  # -> (90, 90) idx 3
        ]
    )

    triangles = compute_delaunay_triangles(width, height, landmarks)

    # A square split by a diagonal yields 2 triangles
    assert len(triangles) == TRIANGLES_IN_SQUARE
    for tri in triangles:
        assert len(tri) == POINTS_IN_TRIANGLE
        max_index = len(landmarks) - 1
        assert all(0 <= idx <= max_index for idx in tri)


def test_compute_delaunay_triangles_handles_duplicates() -> None:
    """Verify triangulation ignores duplicate points but keeps first index."""
    width, height = 100, 100
    landmarks = np.array([[10, 10], [90, 10], [10, 90], [10, 10]])
    triangles = compute_delaunay_triangles(width, height, landmarks)
    assert len(triangles) == 1
    assert sorted(triangles[0]) == [0, 1, 2]
