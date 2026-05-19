"""Align detected faces into canonical square crops using facial landmarks.

This module provides a pipeline for geometrically normalizing face images based
on 68-point facial landmark detection. The alignment process stabilizes facial
orientation by deriving a transformation quadrilateral from eye and mouth
landmarks, then warping the image to produce a standardized square crop.

The pipeline supports optional reflective padding with seamless blending for
faces near image boundaries, configurable output resolutions, and scaling
factors to control crop composition.

Typical usage:

    image_align("input.jpg", "output.png", landmarks, AlignmentOptions())
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast  # pylint: disable=unused-import

import numpy as np
import PIL.Image
import scipy.ndimage
from numpy.typing import NDArray

from ._typing import FloatPoint, PathInput, Point

type AlignmentQuad = NDArray[Any]

logger = logging.getLogger(__name__)
_SCIPY_NDIMAGE = cast("Any", scipy.ndimage)


@dataclass(frozen=True)
class AlignmentOptions:
    """Configuration for aligning a detected face.

    Attributes:
        output_size: Final square image dimension in pixels.
        transform_size: Intermediate square transform dimension in pixels.
        enable_padding: Whether to synthesize reflected image padding.
        x_scale: Horizontal scale factor for the aligned crop.
        y_scale: Vertical scale factor for the aligned crop.
        em_scale: Offset factor from the eyes toward the mouth.
        alpha: Whether to include an alpha mask for padded regions.
    """

    output_size: int = 1024
    transform_size: int = 4096
    enable_padding: bool = True
    x_scale: float = 1.0
    y_scale: float = 1.0
    em_scale: float = 0.1
    alpha: bool = False


# Constants for 68-point facial landmark model
LM_LEFT_EYE = slice(36, 42)
LM_RIGHT_EYE = slice(42, 48)
LM_MOUTH_OUTER = slice(48, 60)
# Represents a tolerance slightly above single-precision float limits
FLOAT_TOLERANCE = 1e-7
# Constants for the standard 68-point facial landmark model
EXPECTED_LANDMARK_COUNT = 68
# Minimum resolution buffer for shrinking.
# A value of 0.5 ensures the face geometry remains at least 2x the target size
# after shrinking, preventing upscaling artifacts in the final transform.
SHRINK_THRESHOLD = 0.5
# Constants for blending heuristics
MIN_PAD_SCALE = 0.3  # Minimum padding relative to quad size
BLUR_SCALE = 0.02  # Gaussian blur sigma relative to quad size
EDGE_BLEND_THRESHOLD = 3.0
SOLID_BLEND_THRESHOLD = 1.0


def _rotate_90_ccw(vec: np.ndarray) -> np.ndarray:
    """Rotates a 2D vector 90 degrees counter-clockwise."""
    return np.array([-vec[1], vec[0]])


def _calculate_alignment_quad(
    face_landmarks: Sequence[FloatPoint | Point], options: AlignmentOptions
) -> tuple[AlignmentQuad, float]:
    """Calculate the alignment quadrilateral and crop size for a face image.

    This function computes a bounding quadrilateral (quad) based on facial
    landmarks to align the face. It stabilizes the roll angle by combining
    the eye-line vector and the facial vertical axis.

    Args:
        face_landmarks: Sequence of (x, y) landmark coordinate pairs. Must
            conform to the 68-point landmark model format.
        options: Configuration object containing scaling factors (x_scale,
            y_scale, em_scale) for the alignment.

    Raises:
        ValueError: If the number of landmarks provided does not match the
            expected 68-point model format.
        ValueError: If the calculated alignment vector has zero magnitude,
            indicating degenerate face geometry (e.g., corrupt landmarks).

    Returns:
        tuple[Array, float]: A tuple containing:
            - **crop_corners**: A 4x2 array representing the corners of the
              crop quadrilateral (top-left, top-right, bottom-right,
              bottom-left).
            - **crop_size**: The scalar size (width/height) of the square crop.
    """
    landmarks = np.array(face_landmarks)

    if landmarks.shape[0] != EXPECTED_LANDMARK_COUNT:
        raise ValueError(
            f"This function requires the 68-point landmark model. "
            f"Received {landmarks.shape[0]} points."
        )

    # Extract landmark groups using standard 68-point model indices
    lm_eye_left = landmarks[LM_LEFT_EYE]
    lm_eye_right = landmarks[LM_RIGHT_EYE]
    lm_mouth_outer = landmarks[LM_MOUTH_OUTER]

    # Calculate geometric centers of facial features
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eyes_midpoint = (eye_left + eye_right) * 0.5
    eye_line_vec = eye_right - eye_left

    left_mouth_corner = lm_mouth_outer[0]
    mouth_right_corner = lm_mouth_outer[6]
    mouth_avg = (left_mouth_corner + mouth_right_corner) * 0.5
    face_axis_vec = mouth_avg - eyes_midpoint

    # Determine the orientation of the crop rectangle.
    # We combine the eye-line vector with a rotated version of the face axis
    # to stabilize the alignment against head rotation (roll).
    crop_vec_x = eye_line_vec - _rotate_90_ccw(face_axis_vec)

    # Normalize the vector to unit length
    x_norm = np.hypot(*crop_vec_x)
    if x_norm < FLOAT_TOLERANCE:
        raise ValueError("Degenerate face geometry: alignment vector is zero.")

    crop_vec_x /= x_norm

    # Scale the vector based on facial dimensions.
    # The crop size is determined by the larger of the eye width or face height,
    # ensuring the entire face is captured.
    crop_vec_x *= max(
        np.hypot(*eye_line_vec) * 2.0, np.hypot(*face_axis_vec) * 1.8
    )
    crop_vec_x *= options.x_scale

    # Calculate the perpendicular Y vector for the crop height
    # Note: This rotates the X vector 90 degrees CCW.
    crop_vec_y = _rotate_90_ccw(crop_vec_x) * options.y_scale

    # Calculate the center of the crop rectangle.
    # The center is offset from the eyes towards the mouth based on em_scale.
    crop_center = eyes_midpoint + face_axis_vec * options.em_scale

    # Compute the four corners of the quadrilateral.
    # PIL expects corners in order: top-left, bottom-left, bottom-right,
    # top-right.
    crop_corners = np.stack(
        [
            crop_center - crop_vec_x - crop_vec_y,  # Top-left
            crop_center - crop_vec_x + crop_vec_y,  # Bottom-left
            crop_center + crop_vec_x + crop_vec_y,  # Bottom-right
            crop_center + crop_vec_x - crop_vec_y,  # Top-right
        ]
    )

    # Calculate the final scalar size of the crop
    crop_size = float(np.hypot(*crop_vec_x) * 2)

    return crop_corners, crop_size


def _shrink_image(
    img: PIL.Image.Image,
    crop_corners: AlignmentQuad,
    crop_size: float,
    options: AlignmentOptions,
) -> tuple[PIL.Image.Image, AlignmentQuad, float]:
    """Conditionally downscale the image and adjust the crop geometry.

    If the detected face crop is significantly larger than the required output
    resolution, this function shrinks the image. It ensures the image
    remains large enough for the transformation pipeline, preventing
    upscaling artifacts later.

    Args:
        img: The source PIL Image to be potentially downscaled.
        crop_corners: A 4x2 array representing the corners of the
            crop quadrilateral (top-left, top-right, bottom-right,
            bottom-left).
        crop_size: The scalar size (width/height) of the square crop area.
        options: Alignment configuration containing target `output_size`
            and `transform_size`.

    Returns:
        A tuple containing:
            - The potentially resized PIL Image.
            - The updated 4x2 array of crop corners.
            - The updated scalar crop size.
    """
    # Determine the minimum resolution needed for the pipeline. We must not
    # shrink below transform_size, or the subsequent transform step will
    # require upscaling, which introduces blur.
    min_required_size = max(options.output_size, options.transform_size)

    # Calculate the integer shrink divisor. SHRINK_THRESHOLD ensures we only
    # downscale if the crop is significantly larger than required, preserving
    # a 2x resolution buffer for high-quality interpolation.
    shrink = int(np.floor(crop_size / min_required_size * SHRINK_THRESHOLD))

    if shrink > 1:
        # Calculate new dimensions, enforcing a minimum of 1 pixel to prevent
        # zero-dimension errors in PIL if shrink is extremely large.
        rsize = (
            max(1, int(np.rint(img.size[0] / shrink))),
            max(1, int(np.rint(img.size[1] / shrink))),
        )

        img = img.resize(rsize, PIL.Image.Resampling.LANCZOS)

        # Update geometry by creating a new array, avoiding in-place mutation
        # of the caller's original data.
        crop_corners = crop_corners / shrink
        crop_size = crop_size / shrink

    return img, crop_corners, crop_size


def _crop_image(
    img: PIL.Image.Image, quad: AlignmentQuad, border: int
) -> tuple[PIL.Image.Image, AlignmentQuad]:
    """Crops the image to the bounding box of the alignment quad plus a border.

    Args:
        img: The source PIL Image.
        quad: The 4x2 array of alignment quad coordinates.
        border: Padding to add around the bounding box.

    Returns:
        A tuple of the cropped image and the adjusted quad coordinates.
    """
    # Ensure quad is a numpy array for advanced indexing.
    quad = np.asarray(quad)

    # Calculate bounding box of the quad.
    # Use float precision for bounds calculation before flooring.
    left = int(np.floor(np.min(quad[:, 0])))
    top = int(np.floor(np.min(quad[:, 1])))
    right = int(np.ceil(np.max(quad[:, 0])))
    bottom = int(np.ceil(np.max(quad[:, 1])))

    # Add border and clamp to image boundaries.
    width, height = img.size
    left = max(left - border, 0)
    top = max(top - border, 0)
    right = min(right + border, width)
    bottom = min(bottom + border, height)

    # Validate crop dimensions.
    if right <= left or bottom <= top:
        # If the quad is outside the image, skip cropping.
        return img, quad

    # Perform crop only if necessary.
    crop_box = (left, top, right, bottom)

    # Check if the crop box is strictly smaller than the image.
    if crop_box != (0, 0, width, height):
        img = img.crop(crop_box)
        # Adjust quad coordinates relative to the new image origin (0,0).
        quad = quad - np.array([left, top], dtype=quad.dtype)

    return img, quad


def _compute_blend_gradient(img: np.ndarray, padding: np.ndarray) -> np.ndarray:
    """Computes a distance-based gradient map for blending padded regions.

    Values are based on the normalized distance to the nearest padding boundary.
    Pixels in the original image region have values >= 1.0, while pixels in the
    padded region have values < 1.0, approaching 0.0 at the extreme edges.

    Args:
        img: The image array, expected to be 3D (H, W, C).
        padding: A 4-element sequence of padding sizes in the format
            (left, top, right, bottom).

    Returns:
        A 2D float array of shape (H, W) representing the blend gradient.
    """
    height, width, _ = img.shape
    # 2D grids are sufficient; broadcasting will handle the color channel.
    y, x = np.ogrid[:height, :width]

    # Normalized distance to the left/right and top/bottom borders.
    # FLOAT_TOLERANCE prevents division by zero if a padding dimension is 0.
    x_dist = np.minimum(
        np.float32(x) / max(padding[0], FLOAT_TOLERANCE),
        np.float32(width - 1 - x) / max(padding[2], FLOAT_TOLERANCE),
    )
    y_dist = np.minimum(
        np.float32(y) / max(padding[1], FLOAT_TOLERANCE),
        np.float32(height - 1 - y) / max(padding[3], FLOAT_TOLERANCE),
    )

    return np.maximum(1.0 - x_dist, 1.0 - y_dist)


def _extend_image_canvas(
    img: PIL.Image.Image,
    quad: np.ndarray,
    qsize: float,
    border: int,
    options: AlignmentOptions,
) -> tuple[PIL.Image.Image, np.ndarray]:
    """Extends the image canvas using reflective padding and seamless blending.

    If the alignment quad extends beyond the image boundaries, this function
    pads the image to accommodate it. It uses a technique of reflective padding
    combined with Gaussian blurring and median color blending to create a
    smooth, artifact-free transition at the edges.

    Args:
        img: The source PIL Image.
        quad: The 4x2 array of alignment quad coordinates.
        qsize: The scalar crop size, used to scale blending parameters.
        border: The pixel margin required around the quad.
        options: Configuration object containing alignment parameters.

    Returns:
        A tuple containing the extended PIL Image and the updated quad
        coordinates shifted by the applied padding.
    """
    # Calculate bounding box of the quad.
    quad_bbox = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )

    # Calculate required padding margins (left, top, right, bottom).
    padding = (
        max(-quad_bbox[0] + border, 0),
        max(-quad_bbox[1] + border, 0),
        max(quad_bbox[2] - img.size[0] + border, 0),
        max(quad_bbox[3] - img.size[1] + border, 0),
    )

    # Skip if padding is disabled or practically zero.
    if not options.enable_padding or max(padding) < 1:
        return img, quad

    # Ensure minimum padding size for effective blending.
    min_pad = int(np.rint(qsize * MIN_PAD_SCALE))
    padding = np.maximum(padding, min_pad)

    # Apply reflective padding.
    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.pad(
        img_array,
        ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)),
        "reflect",
    )

    # Generate gradient map for blending the padded edges.
    mask = _compute_blend_gradient(img_array, padding)
    blur = qsize * BLUR_SCALE

    # Blend with blurred version to smooth out reflection artifacts.
    blurred = _SCIPY_NDIMAGE.gaussian_filter(img_array, [blur, blur, 0])
    blend_mask_blur = np.clip(
        mask * EDGE_BLEND_THRESHOLD + SOLID_BLEND_THRESHOLD, 0.0, 1.0
    )
    img_array += (blurred - img_array) * blend_mask_blur

    # Blend with median color to create a solid fade-out at the extreme edges.
    median_color = np.median(img_array, axis=(0, 1))
    img_array += (median_color - img_array) * np.clip(mask, 0.0, 1.0)

    # Convert back to uint8.
    img_array = np.clip(np.rint(img_array), 0, 255).astype(np.uint8)

    # Handle Alpha channel generation.
    if options.alpha:
        alpha_mask = 1 - np.clip(EDGE_BLEND_THRESHOLD * mask, 0.0, 1.0)
        alpha_mask = np.clip(np.rint(alpha_mask * 255), 0, 255).astype(np.uint8)
        # Expand dimensions to concatenate: (H, W) -> (H, W, 1)
        img_array = np.concatenate(
            (img_array, alpha_mask[..., np.newaxis]), axis=2
        )
        img = PIL.Image.fromarray(img_array, "RGBA")
    else:
        img = PIL.Image.fromarray(img_array, "RGB")

    # Offset quad coordinates to account for the new padding.
    quad += padding[:2]

    return img, quad


def _warp_image(
    img: PIL.Image.Image, quad: AlignmentQuad, options: AlignmentOptions
) -> PIL.Image.Image:
    """Warps the input image to align the face based on the quad and resizes.

    This function performs a quadrilateral transformation (warp) to align the
    face geometry defined by `quad` to a standard square canvas. It then
    resizes the result to the final desired output dimensions.

    Args:
        img: The source PIL Image (typically cropped and padded).
        quad: A NumPy array of shape (4, 2) containing the (x, y) coordinates
            of the four corners of the alignment quadrilateral.
        options: Configuration object containing `transform_size` and
            `output_size` parameters.

    Returns:
        The warped and resized PIL Image.

    Raises:
        ValueError: If `quad` does not have the expected shape (4, 2).
    """
    # Validate input shape to prevent cryptic errors from PIL during
    # transformation.
    if quad.shape != (4, 2):
        raise ValueError(
            f"Invalid quad shape: expected (4, 2), got {quad.shape}"
        )

    # PIL Image.transform maps the *center* of the source pixels.
    # The input quad coordinates are calculated based on pixel corners
    # (standard NumPy indexing). We add 0.5 to shift the coordinates
    # to the pixel center, ensuring geometric alignment.
    quad_coords = (quad + 0.5).flatten().tolist()

    # Perform the quadrilateral warp to the intermediate transform size.
    logger.info("Warping image to transform size...")
    img = img.transform(
        (options.transform_size, options.transform_size),
        PIL.Image.Transform.QUAD,
        quad_coords,
        PIL.Image.Resampling.BICUBIC,
    )

    # Resize to the final output size if it differs from the transform size.
    if options.output_size != options.transform_size:
        # Select the optimal resampling filter based on scaling direction.
        if options.output_size < options.transform_size:
            # Downscaling: LANCZOS provides the best anti-aliasing quality.
            resample = PIL.Image.Resampling.LANCZOS
        else:
            # Upscaling: BICUBIC is preferred over LANCZOS for enlarging
            # as it avoids some ringing artifacts.
            resample = PIL.Image.Resampling.BICUBIC
            logger.warning(
                "Output size is larger than transform size; upscaling image."
            )

        img = img.resize(
            (options.output_size, options.output_size), resample=resample
        )

    return img


def image_align(
    src_file: PathInput,
    dst_file: PathInput,
    face_landmarks: Sequence[FloatPoint | Point],
    options: AlignmentOptions | None = None,
) -> None:
    """Align and save a face crop from a source image.

    Args:
        src_file: Path to the source image.
        dst_file: Path where the aligned PNG should be written.
        face_landmarks: Sequence of 68 ``(x, y)`` facial landmark points.
        options: Optional alignment configuration.
    """
    options = options or AlignmentOptions()

    if not Path(src_file).is_file():
        logger.error("Cannot find source image.")
        return

    quad, qsize = _calculate_alignment_quad(face_landmarks, options)
    img = PIL.Image.open(src_file).convert("RGBA").convert("RGB")
    img, quad, qsize = _shrink_image(img, quad, qsize, options)
    border = max(int(np.rint(qsize * 0.1)), 3)
    img, quad = _crop_image(img, quad, border)
    img, quad = _extend_image_canvas(img, quad, qsize, border, options)
    img = _warp_image(img, quad, options)
    img.save(dst_file, "PNG")
