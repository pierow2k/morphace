"""Module for performing image warping and morphing between two faces."""

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ._typing import (
    FloatPoint,
    ImageArray,
    ImagePair,
    LandmarkList,
    Size,
    TriangleList,
)
from .morph_video import video_writer_context

_CV2: Any = cv2
_NP: Any = np
TrianglePoints = tuple[
    np.ndarray | list[FloatPoint],
    np.ndarray | list[FloatPoint],
    np.ndarray | list[FloatPoint],
]


def _apply_affine_transform(
    src: ImageArray,
    src_tri: list[FloatPoint] | np.ndarray,
    dst_tri: list[FloatPoint] | np.ndarray,
    size: Size,
) -> ImageArray:
    """Applies an affine transform to warp a source image patch.

    Calculates the transform matrix mapping src_tri to dst_tri and applies it
    to the source image.

    Args:
        src: The source image patch.
        src_tri: Triangle coordinates (3 points) in the source image.
        dst_tri: Triangle coordinates (3 points) in the destination image.
        size: Output image size (width, height).

    Returns:
        The warped image patch.
    """
    # Ensure inputs are float32 arrays for OpenCV
    src_tri_arr = _NP.asarray(src_tri, dtype=_NP.float32)
    dst_tri_arr = _NP.asarray(dst_tri, dtype=_NP.float32)

    # Given a pair of triangles, find the affine transform.
    warp_mat = _CV2.getAffineTransform(src_tri_arr, dst_tri_arr)

    # Apply the Affine Transform
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
        img1: The first source image.
        img2: The second source image.
        img: The output image to be updated.
        triangles: Source and destination triangle coordinates.
        alpha: Blending factor.
    """
    tri_src1, tri_src2, tri_dest = triangles

    # Find bounding rectangles for each triangle
    rect_src1 = _CV2.boundingRect(_NP.float32([tri_src1]))
    rect_src2 = _CV2.boundingRect(_NP.float32([tri_src2]))
    rect_dest = _CV2.boundingRect(_NP.float32([tri_dest]))

    # Calculate offset points relative to top-left corner of bounding
    # rectangles using list comprehensions for brevity and readability.
    tri_src1_offset = [
        (tri_src1[i][0] - rect_src1[0], tri_src1[i][1] - rect_src1[1])
        for i in range(3)
    ]
    tri_src2_offset = [
        (tri_src2[i][0] - rect_src2[0], tri_src2[i][1] - rect_src2[1])
        for i in range(3)
    ]
    tri_dest_offset = [
        (tri_dest[i][0] - rect_dest[0], tri_dest[i][1] - rect_dest[1])
        for i in range(3)
    ]

    # Create mask for the destination triangle
    # Dynamically determine channels to support grayscale or RGBA
    # the rect_dest is (x, y, w, h).
    rect_dest_width, rect_dest_height = rect_dest[2], rect_dest[3]

    # Handle both grayscale (2D) and color (3D) images
    grayscale_dims = 2
    if img.ndim == grayscale_dims:
        mask_shape = (rect_dest_height, rect_dest_width)
        fill_color = 1.0
    else:
        mask_shape = (rect_dest_height, rect_dest_width, img.shape[2])
        fill_color = tuple([1.0] * img.shape[2])

    mask = np.zeros(mask_shape, dtype=np.float32)
    _CV2.fillConvexPoly(mask, _NP.int32([tri_dest_offset]), fill_color, 16, 0)

    # Crop source image patches
    img1_rect = img1[
        rect_src1[1] : rect_src1[1] + rect_src1[3],
        rect_src1[0] : rect_src1[0] + rect_src1[2],
    ]
    img2_rect = img2[
        rect_src2[1] : rect_src2[1] + rect_src2[3],
        rect_src2[0] : rect_src2[0] + rect_src2[2],
    ]

    # Warp patches
    size = (rect_dest_width, rect_dest_height)
    warp_img1 = _apply_affine_transform(
        img1_rect, tri_src1_offset, tri_dest_offset, size
    )
    warp_img2 = _apply_affine_transform(
        img2_rect, tri_src2_offset, tri_dest_offset, size
    )

    # Alpha blend rectangular patches
    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

    # Copy triangular region to output image using the mask
    # Slicing coordinates for the destination image
    y_start, x_start = rect_dest[1], rect_dest[0]
    y_end, x_end = y_start + rect_dest_height, x_start + rect_dest_width

    # Ensure we don't go out of bounds (safety check)
    img_roi = img[y_start:y_end, x_start:x_end]

    # Blend using mask (handle broadcasting if mask is 2D and img_roi is 3D)
    img[y_start:y_end, x_start:x_end] = img_roi * (1 - mask) + img_rect * mask


def generate_morph_frame(  # noqa: PLR0913
    img1: ImageArray,
    img2: ImageArray,
    p1_arr: np.ndarray,
    p2_arr: np.ndarray,
    tri_list: TriangleList,
    alpha: float,
    show_triangles: bool,
) -> ImageArray:
    """Generates a single morphed frame for a given alpha value.

    Args:
        img1: Float32 source image 1.
        img2: Float32 source image 2.
        p1_arr: Source points for image 1.
        p2_arr: Source points for image 2.
        tri_list: List of triangles.
        alpha: Interpolation factor (0.0 to 1.0).
        show_triangles: Whether to draw triangle outlines.

    Returns:
        The generated morphed frame (float32 BGR array).
    """
    # Vectorized calculation of intermediate points
    points_arr = (1 - alpha) * p1_arr + alpha * p2_arr

    # Allocate space for final output
    morphed_frame = _NP.zeros(img1.shape, dtype=_NP.float32)

    for i in range(len(tri_list)):
        x, y, z = map(int, tri_list[i])

        # Use NumPy fancy indexing to get triangle vertices
        t1 = p1_arr[[x, y, z]]
        t2 = p2_arr[[x, y, z]]
        t = points_arr[[x, y, z]]

        _morph_triangle(img1, img2, morphed_frame, (t1, t2, t), alpha)

        if show_triangles:
            pts = t.reshape((-1, 1, 2)).astype(_NP.int32)
            _CV2.polylines(morphed_frame, [pts], True, (255, 255, 255), 1)

    return _NP.clip(morphed_frame, 0, 255)


def generate_morph_sequence(
    img_pair: ImagePair,
    points_pair: Sequence[LandmarkList],
    tri_list: TriangleList,
    video_config: tuple[int, int, Size, str],
    show_triangles: bool,
) -> None:
    """Generates a face morphing sequence and saves it as a video."""
    img1, img2 = img_pair
    points1, points2 = points_pair

    # Pre-processing
    img1_float = _NP.float32(img1)
    img2_float = _NP.float32(img2)
    p1_arr = _NP.array(points1, dtype=_NP.float32)
    p2_arr = _NP.array(points2, dtype=_NP.float32)

    with video_writer_context(video_config) as (stdin, num_images):
        for j in range(num_images):
            alpha = j / max(1, num_images - 1)

            # Logic is delegated to the helper
            frame = generate_morph_frame(
                img1_float,
                img2_float,
                p1_arr,
                p2_arr,
                tri_list,
                alpha,
                show_triangles,
            )

            # Conversion and writing
            res = Image.fromarray(
                _CV2.cvtColor(_NP.uint8(frame), _CV2.COLOR_BGR2RGB)
            )
            res.save(stdin, "JPEG")
