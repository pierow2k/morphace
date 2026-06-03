"""Tests for face correspondence and image alignment."""

import pytest

from morphace.morphing.correspondence import _get_boundary_points

BOUNDARY_POINTS_COUNT = 8


@pytest.mark.parametrize(
    ("h", "w", "expected"),
    [
        (
            100,
            200,
            [
                (1, 1),  # Top-left
                (199, 1),  # Top-right
                (199, 99),  # Bottom-right
                (1, 99),  # Bottom-left
                (99, 1),  # Top-mid: (200-1)//2
                (199, 49),  # Right-mid: (100-1)//2
                (99, 99),  # Bottom-mid
                (1, 49),  # Left-mid
            ],
        ),
        (
            11,
            11,
            [
                (1, 1),
                (10, 1),
                (10, 10),
                (1, 10),
                (5, 1),
                (10, 5),
                (5, 10),
                (1, 5),
            ],
        ),
    ],
)
def test_get_boundary_points(
    h: int,
    w: int,
    expected: list[tuple[int, int]],
) -> None:
    """Verify the calculation of boundary points using various dimensions."""
    result = _get_boundary_points(h, w)
    assert len(result) == BOUNDARY_POINTS_COUNT
    assert result == expected
