"""Module for performing image warping and morphing between two faces."""

import logging
from collections.abc import Sequence

import cv2
import numpy as np
from tqdm import tqdm

from morphace._typing import (
    FloatPoint,
    ImageArray,
    ImagePair,
    LandmarkList,
    Size,
    TriangleList,
)

from .config import MorphVideoConfig
from .video import video_writer_context

logger = logging.getLogger(__name__)

TrianglePoints = tuple[
    np.ndarray | list[FloatPoint],
    np.ndarray | list[FloatPoint],
    np.ndarray | list[FloatPoint],
]
type CvRect = tuple[int, int, int, int]

_GRAYSCALE_DIMS = 2
_PROGRESS_LOG_INTERVAL_SECONDS = 5.0


class _TqdmLogStream:
    """File-like stream that forwards tqdm output through module logging."""

    def write(self, message: str) -> None:
        """Log non-empty tqdm progress messages."""
        progress_message = message.strip()
        if progress_message:
            logger.info(progress_message)

    def flush(self) -> None:
        """Satisfy tqdm's file-like stream interface."""


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
    src_tri_arr = np.asarray(src_tri, dtype=np.float32)
    dst_tri_arr = np.asarray(dst_tri, dtype=np.float32)

    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(src_tri_arr, dst_tri_arr)

    # Apply the Affine Transform
    return cv2.warpAffine(
        src,
        warp_mat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _bounding_rect(triangle: np.ndarray | list[FloatPoint]) -> CvRect:
    """Return the OpenCV bounding rectangle for a triangle."""
    x, y, width, height = cv2.boundingRect(
        np.asarray([triangle], dtype=np.float32)
    )
    return (x, y, width, height)


def _offset_triangle(
    triangle: np.ndarray | list[FloatPoint],
    rect: CvRect,
) -> list[FloatPoint]:
    """Return triangle points relative to a rectangle origin."""
    return [
        (triangle[index][0] - rect[0], triangle[index][1] - rect[1])
        for index in range(3)
    ]


def _triangle_mask(
    image: ImageArray,
    rect: CvRect,
    triangle: list[FloatPoint],
) -> ImageArray:
    """Create a mask for a triangle inside a bounding rectangle."""
    width, height = rect[2], rect[3]

    if image.ndim == _GRAYSCALE_DIMS:
        mask_shape = (height, width)
        fill_color = 1.0
    else:
        mask_shape = (height, width, image.shape[2])
        fill_color = tuple([1.0] * image.shape[2])

    mask = np.zeros(mask_shape, dtype=np.float32)
    cv2.fillConvexPoly(
        mask,
        np.asarray([triangle], dtype=np.int32),
        fill_color,
        16,
        0,
    )
    return mask


def _crop_rect(image: ImageArray, rect: CvRect) -> ImageArray:
    """Crop an image array to an OpenCV rectangle."""
    x, y, width, height = rect
    return image[y : y + height, x : x + width]


def _blend_triangle_patch(
    image: ImageArray,
    rect: CvRect,
    mask: ImageArray,
    patch: ImageArray,
) -> None:
    """Blend a warped triangular patch into the destination image."""
    x, y, width, height = rect
    destination = image[y : y + height, x : x + width]
    image[y : y + height, x : x + width] = (
        destination * (1 - mask) + patch * mask
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

    rect_src1 = _bounding_rect(tri_src1)
    rect_src2 = _bounding_rect(tri_src2)
    rect_dest = _bounding_rect(tri_dest)

    tri_src1_offset = _offset_triangle(tri_src1, rect_src1)
    tri_src2_offset = _offset_triangle(tri_src2, rect_src2)
    tri_dest_offset = _offset_triangle(tri_dest, rect_dest)

    mask = _triangle_mask(img, rect_dest, tri_dest_offset)
    img1_rect = _crop_rect(img1, rect_src1)
    img2_rect = _crop_rect(img2, rect_src2)

    size = (rect_dest[2], rect_dest[3])
    warp_img1 = _apply_affine_transform(
        img1_rect, tri_src1_offset, tri_dest_offset, size
    )
    warp_img2 = _apply_affine_transform(
        img2_rect, tri_src2_offset, tri_dest_offset, size
    )

    # Alpha blend rectangular patches
    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2
    _blend_triangle_patch(img, rect_dest, mask, img_rect)


def _generate_morph_frame(  # noqa: PLR0913
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
    morphed_frame = np.zeros(img1.shape, dtype=np.float32)

    for tri in tri_list:
        x, y, z = map(int, tri)

        # Use NumPy fancy indexing to get triangle vertices
        t1 = p1_arr[[x, y, z]]
        t2 = p2_arr[[x, y, z]]
        t = points_arr[[x, y, z]]

        _morph_triangle(img1, img2, morphed_frame, (t1, t2, t), alpha)

        if show_triangles:
            pts = t.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(morphed_frame, [pts], True, (255, 255, 255), 1)

    return np.clip(morphed_frame, 0, 255)


def generate_morph_sequence(
    img_pair: ImagePair,
    points_pair: Sequence[LandmarkList],
    tri_list: TriangleList,
    video_config: MorphVideoConfig,
    show_triangles: bool,
) -> None:
    """Generates a face morphing sequence and writes it to a video file.

    This function orchestrates the morphing workflow by iterating through the
    required number of frames, calculating intermediate morph states, and
    streaming raw video frames to an FFmpeg subprocess managed by
    `video_writer_context`.

    Args:
        img_pair: A tuple containing the source and destination images.
        points_pair: A tuple containing facial landmarks for both images.
        tri_list: The Delaunay triangulation list used for warping.
        video_config: Configuration object defining video output settings.
        show_triangles: If True, renders the triangulation mesh over the
            frames.

    Returns:
        None. The output is written to a file via the video writer context.
    """
    img1, img2 = img_pair
    points1, points2 = points_pair

    # Convert images and landmarks to float32 for high-precision arithmetic
    # during the warping and interpolation process.
    img1_float = np.asarray(img1, dtype=np.float32)
    img2_float = np.asarray(img2, dtype=np.float32)
    p1_arr = np.array(points1, dtype=np.float32)
    p2_arr = np.array(points2, dtype=np.float32)

    with video_writer_context(video_config) as (stdin, num_images):
        frame_range = range(num_images)
        progress_frames = tqdm(
            frame_range,
            bar_format=(
                "{desc}: {n_fmt}/{total_fmt} frames ({percentage:3.0f}%)"
            ),
            desc="Generating morph video",
            disable=not logger.isEnabledFor(logging.INFO),
            file=_TqdmLogStream(),
            mininterval=_PROGRESS_LOG_INTERVAL_SECONDS,
            total=num_images,
        )

        for frame_index in progress_frames:
            # Calculate the interpolation factor (0.0 to 1.0).
            # alpha=0.0 yields the source image; alpha=1.0 yields
            # the destination.
            alpha = frame_index / max(1, num_images - 1)

            # Generate the intermediate morph frame.
            frame = _generate_morph_frame(
                img1_float,
                img2_float,
                p1_arr,
                p2_arr,
                tri_list,
                alpha,
                show_triangles,
            )

            # Ensure array is uint8
            frame_uint8 = np.asarray(frame, dtype=np.uint8)

            # Convert frame to uint8 (standard byte format for images).
            # The resulting bytes (BGR order) are streamed directly to FFmpeg.
            stdin.write(frame_uint8.tobytes())
