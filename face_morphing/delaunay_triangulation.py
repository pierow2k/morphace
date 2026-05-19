"""Module for performing Delaunay triangulation on facial landmarks."""

import cv2
import numpy as np

from ._typing import Bounds, Point, LandmarkArray, TriangleList


def _is_within_bounds(bounds: Bounds, point: Point) -> bool:
    """Check if a point is inside a rectangular boundary.

    Args:
        bounds: A tuple of (x_min, y_min, x_max, y_max).
        point: A tuple of (x, y) coordinates.

    Returns:
        True if the point is inside or on the edge of the bounds.
    """
    x_min, y_min, x_max, y_max = bounds
    px, py = point

    return (x_min <= px <= x_max) and (y_min <= py <= y_max)


def _extract_triangle_indices(
    width: int,
    height: int,
    subdiv: cv2.Subdiv2D,
    point_indices: dict[Point, int],
) -> TriangleList:
    """Extracts triangle vertex indices from a Subdiv2D object.

    Args:
        width: Width of the image frame.
        height: Height of the image frame.
        subdiv: The subdivision object containing the points.
        point_indices: Mapping from point coordinates to original indices.

    Returns:
        Triangle vertex indices from the subdivision.
    """
    triangles: TriangleList = []
    triangle_list = subdiv.getTriangleList()
    bounds = (0, 0, width, height)

    for t in triangle_list:
        # Extract points using tuple unpacking for clarity
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        # Check geometric bounds first (optional optimization)
        if not (
            _is_within_bounds(bounds, pt1)
            and _is_within_bounds(bounds, pt2)
            and _is_within_bounds(bounds, pt3)
        ):
            continue

        # Check existence in dictionary to handle virtual points generated
        # by Subdiv2D (e.g., corners) that are not in the original set.
        if (
            pt1 in point_indices
            and pt2 in point_indices
            and pt3 in point_indices
        ):
            triangles.append(
                (point_indices[pt1], point_indices[pt2], point_indices[pt3])
            )

    return triangles


def compute_delaunay_triangles(
    width: int,
    height: int,
    landmarks: LandmarkArray,
) -> TriangleList:
    """Creates a Delaunay triangulation from a provided list of points."""
    points_array = np.asarray(landmarks)

    # Sanitize points: convert to int tuples
    points: list[Point] = [(int(x), int(y)) for x, y in points_array]

    # Create a mapping of coordinates to indices.
    # We keep the first occurrence if duplicates exist.
    index_map: dict[Point, int] = {}
    for idx, pt in enumerate(points):
        if pt not in index_map:
            index_map[pt] = idx

    # Initialize Subdiv2D with a padded rect to ensure all points are inside.
    padding = 1
    # Note: OpenCV Rect is (x, y, width, height)
    rect = (-padding, -padding, width + 2 * padding, height + 2 * padding)
    subdiv = cv2.Subdiv2D(rect)

    # Bulk insert unique points.
    # We trust this will work because we sanitized inputs (padded bounds
    # + unique points). If it fails, we want the error to propagate up.
    subdiv.insert(list(index_map.keys()))

    return _extract_triangle_indices(width, height, subdiv, index_map)
