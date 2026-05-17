"""Module for performing Delaunay triangulation on facial landmarks."""

import cv2

from ._typing import Point, PointArray, Rect, TriangleList


# Check if a point is inside a rectangle
def rect_contains(rect: Rect, point: Point) -> bool:
    """Check if a point is inside a rectangle using chained comparisons."""
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


# Write the delaunay triangles into a file
def draw_delaunay(
    f_w: int,
    f_h: int,
    subdiv: cv2.Subdiv2D,
    dictionary1: dict[Point, int],
) -> TriangleList:
    """Extracts triangle vertex indices from a Subdiv2D object.

    Args:
        f_w: Width of the image frame.
        f_h: Height of the image frame.
        subdiv: The subdivision object containing the points.
        dictionary1: Mapping from point coordinates to original indices.

    Returns:
        Triangle vertex indices from the subdivision.
    """
    list4: TriangleList = []

    triangle_list = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if (
            rect_contains(r, pt1)
            and rect_contains(r, pt2)
            and rect_contains(r, pt3)
        ):
            list4.append((dictionary1[pt1], dictionary1[pt2], dictionary1[pt3]))

    return list4


def make_delaunay(f_w: int, f_h: int, the_list: PointArray) -> TriangleList:
    """Creates a Delaunay triangulation from a provided list of points.

    Args:
        f_w: Width of the image frame.
        f_h: Height of the image frame.
        the_list: Landmark points to triangulate.

    Returns:
        Triangle vertex indices.
    """
    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary.
    the_list = the_list.tolist()
    points: list[Point] = [(int(x[0]), int(x[1])) for x in the_list]
    dictionary = dict(zip(points, range(len(points)), strict=True))

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    # Make and return delaunay triangulation list.
    return draw_delaunay(f_w, f_h, subdiv, dictionary)
