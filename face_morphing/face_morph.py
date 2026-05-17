"""Module for performing image warping and morphing between two faces."""

from subprocess import PIPE, Popen
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from ._typing import (
    FloatPoint,
    ImageArray,
    ImagePair,
    LandmarkList,
    Point,
    Size,
    TriangleList,
)

_CV2 = cast("Any", cv2)
_NP = cast("Any", np)

TrianglePoints = tuple[list[FloatPoint], list[FloatPoint], list[FloatPoint]]


def apply_affine_transform(
    src: ImageArray,
    src_rri: list[FloatPoint],
    dst_tri: list[FloatPoint],
    size: Size,
) -> ImageArray:
    """Apply affine transform.

    Apply affine transform calculated using srcTri and dstTri calculated using
    srcTri and dstTri to src and output an image of size.

    Args:
        src (np.ndarray): The source image patch.
        src_rri (list): Triangle coordinates in the source image.
        dst_tri (list): Triangle coordinates in the destination image.
        size (tuple): The size (width, height) of the output image.

    Returns:
        np.ndarray: The warped image patch.
    """
    # Given a pair of triangles, find the affine transform.
    warp_mat = _CV2.getAffineTransform(
        _NP.float32(src_rri), _NP.float32(dst_tri)
    )

    # Return the Affine Transform just found to the src image
    return _CV2.warpAffine(
        src,
        warp_mat,
        (size[0], size[1]),
        None,
        flags=_CV2.INTER_LINEAR,
        borderMode=_CV2.BORDER_REFLECT_101,
    )


def _morph_triangle(
    img1: ImageArray,
    img2: ImageArray,
    img: ImageArray,
    triangles: TrianglePoints,
    alpha: float,
) -> None:
    """Warps and alpha blends triangular regions from img1 and img2 to img.

    Args:
        img1 (np.ndarray): The first source image.
        img2 (np.ndarray): The second source image.
        img (np.ndarray): The output image to be updated.
        triangles (tuple): A tuple containing (t1, t2, t) triangle coordinates.
        alpha (float): Blending factor.
    """
    t1, t2, t = triangles

    # Find bounding rectangle for each triangle
    r1 = _CV2.boundingRect(_NP.float32([t1]))
    r2 = _CV2.boundingRect(_NP.float32([t2]))
    r = _CV2.boundingRect(_NP.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1_rect: list[FloatPoint] = []
    t2_rect: list[FloatPoint] = []
    t_rect: list[FloatPoint] = []

    for i in range(3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    _CV2.fillConvexPoly(mask, _NP.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    img2_rect = img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    # Alpha blend rectangular patches
    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1] : r[1] + r[3], r[0] : r[0] + r[2]] = (
        img[r[1] : r[1] + r[3], r[0] : r[0] + r[2]] * (1 - mask)
        + img_rect * mask
    )


def generate_morph_sequence(
    img_pair: ImagePair,
    points_pair: tuple[LandmarkList, LandmarkList],
    tri_list: TriangleList,
    video_config: tuple[int, int, Size, str],
) -> None:
    """Generates a face morphing sequence and saves it as a video.

    Args:
        img_pair (tuple): A tuple containing (img1, img2) images.
        points_pair (tuple): A tuple containing (points1, points2) landmarks.
        tri_list (list): A list of triangle vertex indices.
        video_config (tuple): Video parameters (duration, frame_rate, size,
        output).
    """
    img1, img2 = img_pair
    points1, points2 = points_pair
    duration, frame_rate, size, output = video_config

    num_images = int(duration * frame_rate)
    p = Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-r",
            str(frame_rate),
            "-s",
            str(size[1]) + "x" + str(size[0]),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "25",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        stdin=PIPE,
    )
    if p.stdin is None:
        raise RuntimeError("Unable to open ffmpeg input stream.")
    stdin = p.stdin

    for j in range(num_images):
        # Convert Mat to float data type
        img1 = cast("ImageArray", _NP.float32(img1))
        img2 = cast("ImageArray", _NP.float32(img2))

        # Read array of corresponding points
        points: list[FloatPoint] = []
        alpha = j / (num_images - 1)

        # Compute weighted average point coordinates
        for i in range(len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))

        # Allocate space for final output
        morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)

        for i in range(len(tri_list)):
            x = int(tri_list[i][0])
            y = int(tri_list[i][1])
            z = int(tri_list[i][2])

            t1 = [
                _as_float_point(points1[x]),
                _as_float_point(points1[y]),
                _as_float_point(points1[z]),
            ]
            t2 = [
                _as_float_point(points2[x]),
                _as_float_point(points2[y]),
                _as_float_point(points2[z]),
            ]
            t = [points[x], points[y], points[z]]

            # Morph one triangle at a time.
            _morph_triangle(img1, img2, morphed_frame, (t1, t2, t), alpha)

            pt1 = (int(t[0][0]), int(t[0][1]))
            pt2 = (int(t[1][0]), int(t[1][1]))
            pt3 = (int(t[2][0]), int(t[2][1]))

            _CV2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
            _CV2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
            _CV2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)

        res = Image.fromarray(
            _CV2.cvtColor(np.uint8(morphed_frame), _CV2.COLOR_BGR2RGB)
        )
        res.save(stdin, "JPEG")

    stdin.close()
    p.wait()


def _as_float_point(point: Point) -> FloatPoint:
    return (float(point[0]), float(point[1]))
