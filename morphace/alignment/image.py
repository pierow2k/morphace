"""Image-processing helpers for face alignment.

This module owns the raster side of alignment: downscaling large source
images, cropping around the alignment quad, synthesizing reflected padding for
faces near image boundaries, and warping the prepared canvas into a square
output image.
"""

import logging

import numpy as np
import PIL.Image
import scipy.ndimage

from .geometry import AlignmentQuad
from .options import FaceAlignmentOptions

logger = logging.getLogger(__name__)


# A value of 0.5 ensures the face geometry remains at least 2x the target size
# after shrinking, preventing upscaling artifacts in the final transform.
_SHRINK_THRESHOLD = 0.5
_FLOAT_TOLERANCE = 1e-7

# Constants for padding and blending heuristics.
_MIN_PAD_SCALE = 0.3
_BLUR_SCALE = 0.02
_EDGE_BLEND_THRESHOLD = 3.0
_SOLID_BLEND_THRESHOLD = 1.0


def _shrink_image(
    image: PIL.Image.Image,
    crop_corners: AlignmentQuad,
    crop_size: float,
    options: FaceAlignmentOptions,
) -> tuple[PIL.Image.Image, AlignmentQuad, float]:
    """Conditionally downscale the image and adjust the crop geometry.

    If the detected face crop is significantly larger than the required output
    resolution, this function shrinks the image while preserving enough
    resolution for the later transform step.

    Args:
        image: Source image to be potentially downscaled.
        crop_corners: Alignment quad coordinates in the current image space.
        crop_size: Scalar size of the square crop area.
        options: Alignment configuration containing output and transform sizes.

    Returns:
        The potentially resized image, adjusted crop corners, and adjusted crop
        size.
    """
    # Determine the minimum resolution needed for the pipeline. We must not
    # shrink below transform_size, or the later transform step will require
    # upscaling and introduce blur.
    min_required_size = max(options.output_size, options.transform_size)

    # _SHRINK_THRESHOLD preserves a 2x resolution buffer before downscaling.
    shrink = int(np.floor(crop_size / min_required_size * _SHRINK_THRESHOLD))

    if shrink > 1:
        # Enforce at least one pixel per dimension to avoid invalid PIL sizes
        # when the shrink factor is very large.
        resized_size = (
            max(1, int(np.rint(image.size[0] / shrink))),
            max(1, int(np.rint(image.size[1] / shrink))),
        )
        image = image.resize(resized_size, PIL.Image.Resampling.LANCZOS)

        # Keep the quad in the same coordinate space as the resized image.
        crop_corners = crop_corners / shrink
        crop_size = crop_size / shrink

    return image, crop_corners, crop_size


def _crop_image(
    image: PIL.Image.Image,
    quad: AlignmentQuad,
    border: int,
) -> tuple[PIL.Image.Image, AlignmentQuad]:
    """Crop the image to the alignment quad bounds plus a border.

    Args:
        image: Source image.
        quad: Alignment quad coordinates.
        border: Padding to add around the quad bounding box.

    Returns:
        The cropped image and quad adjusted to the cropped image origin.
    """
    # Ensure quad is a NumPy array so coordinate slicing is predictable.
    quad = np.asarray(quad)

    # Calculate the bounding box of the quad, using floor/ceil to tightly
    # encompass the face geometry before converting to integer pixel coords.
    left = int(np.floor(np.min(quad[:, 0])))
    top = int(np.floor(np.min(quad[:, 1])))
    right = int(np.ceil(np.max(quad[:, 0])))
    bottom = int(np.ceil(np.max(quad[:, 1])))

    # Add border and clamp the crop to image boundaries.
    width, height = image.size
    left = max(left - border, 0)
    top = max(top - border, 0)
    right = min(right + border, width)
    bottom = min(bottom + border, height)

    if right <= left or bottom <= top:
        # If the quad is outside the image, skip cropping and let padding or
        # the final warp handle the geometry.
        return image, quad

    crop_box = (left, top, right, bottom)
    if crop_box != (0, 0, width, height):
        image = image.crop(crop_box)
        # Rebase quad coordinates to the cropped image's origin.
        quad = quad - np.array([left, top], dtype=quad.dtype)

    return image, quad


def _compute_blend_gradient(
    image_array: np.ndarray,
    padding: np.ndarray,
) -> np.ndarray:
    """Compute a distance-based gradient map for padded-region blending.

    Values are based on the normalized distance to the nearest padding
    boundary. Pixels in the original image region have negative values,
    while pixels in padded regions approach 1.0 at the outermost edges
    and fall to 0.0 at the padding boundary.

    Args:
        image_array: Image array with shape ``(height, width, channels)``.
        padding: Padding sizes in ``(left, top, right, bottom)`` order.

    Returns:
        A 2D float array representing the blend gradient.
    """
    height, width, _ = image_array.shape

    # 2D grids are sufficient; broadcasting handles the color channel later.
    y, x = np.ogrid[:height, :width]

    # _FLOAT_TOLERANCE avoids division by zero when one side has no padding.
    x_dist = np.minimum(
        np.float32(x) / max(padding[0], _FLOAT_TOLERANCE),
        np.float32(width - 1 - x) / max(padding[2], _FLOAT_TOLERANCE),
    )
    y_dist = np.minimum(
        np.float32(y) / max(padding[1], _FLOAT_TOLERANCE),
        np.float32(height - 1 - y) / max(padding[3], _FLOAT_TOLERANCE),
    )

    return np.maximum(1.0 - x_dist, 1.0 - y_dist)


def _required_padding(
    image: PIL.Image.Image,
    quad: AlignmentQuad,
    border: int,
) -> tuple[int, int, int, int]:
    """Calculate missing canvas margins in left, top, right, bottom order."""
    quad_bbox = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )

    return (
        max(-quad_bbox[0] + border, 0),
        max(-quad_bbox[1] + border, 0),
        max(quad_bbox[2] - image.size[0] + border, 0),
        max(quad_bbox[3] - image.size[1] + border, 0),
    )


def _minimum_padding(
    padding: tuple[int, int, int, int],
    crop_size: float,
) -> np.ndarray:
    """Apply the minimum padding needed for smooth edge blending."""
    min_pad = int(np.rint(crop_size * _MIN_PAD_SCALE))
    return np.maximum(padding, min_pad)


def _reflected_canvas_array(
    image: PIL.Image.Image,
    padding: np.ndarray,
) -> np.ndarray:
    """Return a float image array extended by reflective padding."""
    image_array = np.asarray(image, dtype=np.float32)
    return np.pad(
        image_array,
        ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)),
        "reflect",
    )


def _blend_padded_edges(
    image_array: np.ndarray,
    padding: np.ndarray,
    crop_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth reflected padding with blur and median-color blending."""
    mask = _compute_blend_gradient(image_array, padding)
    blur = crop_size * _BLUR_SCALE

    blurred = scipy.ndimage.gaussian_filter(image_array, [blur, blur, 0])
    blend_mask_blur = np.clip(
        mask * _EDGE_BLEND_THRESHOLD + _SOLID_BLEND_THRESHOLD,
        0.0,
        1.0,
    )
    image_array += (blurred - image_array) * blend_mask_blur[..., np.newaxis]

    median_color = np.median(image_array, axis=(0, 1))
    median_mask = np.clip(mask, 0.0, 1.0)
    image_array += (median_color - image_array) * median_mask[..., np.newaxis]

    image_array = np.clip(np.rint(image_array), 0, 255).astype(np.uint8)
    return image_array, mask


def _canvas_image_from_array(
    image_array: np.ndarray,
    mask: np.ndarray,
    include_alpha: bool,
) -> PIL.Image.Image:
    """Build a Pillow image from a padded canvas array."""
    if not include_alpha:
        return PIL.Image.fromarray(image_array, "RGB")

    alpha_mask = 1 - np.clip(_EDGE_BLEND_THRESHOLD * mask, 0.0, 1.0)
    alpha_mask = np.clip(np.rint(alpha_mask * 255), 0, 255).astype(np.uint8)
    image_array = np.concatenate(
        (image_array, alpha_mask[..., np.newaxis]),
        axis=2,
    )
    return PIL.Image.fromarray(image_array, "RGBA")


def _extend_image_canvas(
    image: PIL.Image.Image,
    quad: AlignmentQuad,
    crop_size: float,
    border: int,
    options: FaceAlignmentOptions,
) -> tuple[PIL.Image.Image, AlignmentQuad]:
    """Extend the image canvas with reflected padding and blended edges.

    If the alignment quad extends beyond the image boundaries, this function
    pads the image to accommodate it. It uses reflective padding, Gaussian
    blur, and median-color blending to smooth edge artifacts.

    Args:
        image: Source image.
        quad: Alignment quad coordinates.
        crop_size: Scalar crop size used to scale blend parameters.
        border: Pixel margin required around the quad.
        options: Alignment configuration containing padding options.

    Returns:
        The extended image and quad shifted by the applied padding.
    """
    padding = _required_padding(image, quad, border)

    if not options.enable_padding or max(padding) < 1:
        return image, quad

    padding = _minimum_padding(padding, crop_size)
    image_array = _reflected_canvas_array(image, padding)
    image_array, mask = _blend_padded_edges(image_array, padding, crop_size)
    image = _canvas_image_from_array(image_array, mask, options.alpha)

    # Shift the quad into the padded image's coordinate space.
    quad += padding[:2]

    return image, quad


def prepare_alignment_canvas(
    image: PIL.Image.Image,
    quad: AlignmentQuad,
    crop_size: float,
    options: FaceAlignmentOptions,
) -> tuple[PIL.Image.Image, AlignmentQuad]:
    """Prepare an image and quad for the final alignment warp.

    This groups the pre-warp image operations that must stay in sequence:
    shrink oversized inputs, crop to the useful region around the face, then
    extend the canvas if the requested crop reaches beyond the source image.

    Args:
        image: Source image to shrink, crop, and pad as needed.
        quad: Alignment quad in source image coordinates.
        crop_size: Scalar square crop size.
        options: Configuration for output sizing and padding.

    Returns:
        The prepared image and quad adjusted to the prepared image's
        coordinate space.
    """
    image, quad, crop_size = _shrink_image(image, quad, crop_size, options)
    # Add a 10% border around the crop (minimum 3 pixels) for context.
    border = max(int(np.rint(crop_size * 0.1)), 3)
    image, quad = _crop_image(image, quad, border)
    return _extend_image_canvas(image, quad, crop_size, border, options)


def warp_aligned_face(
    image: PIL.Image.Image,
    quad: AlignmentQuad,
    options: FaceAlignmentOptions,
) -> PIL.Image.Image:
    """Warp the input image to align the face and resize the result.

    This performs a quadrilateral transform that maps the face crop to a
    standard square canvas, then resizes that canvas to the final output size.

    Args:
        image: Source image, typically cropped and padded.
        quad: NumPy array of shape (4, 2) containing source coordinates.
        options: Configuration for transform and output sizing.

    Raises:
        ValueError: If `quad` does not have the expected shape (4, 2).

    Returns:
        The warped and resized face image.
    """
    if quad.shape != (4, 2):
        message = f"Invalid quad shape: expected (4, 2), got {quad.shape}"
        raise ValueError(message)

    # Pillow's transform samples from source pixel centers. The quad is based
    # on NumPy-style pixel-corner coordinates, so shift by 0.5 for alignment.
    quad_coords = (quad + 0.5).flatten().tolist()

    # First warp into the high-resolution intermediate transform canvas.
    logger.info("Warping image to transform size...")
    image = image.transform(
        (options.transform_size, options.transform_size),
        PIL.Image.Transform.QUAD,
        quad_coords,
        PIL.Image.Resampling.BICUBIC,
    )

    # Resize to the final output size if it differs from the transform size.
    if options.output_size != options.transform_size:
        if options.output_size < options.transform_size:
            # Downscaling: LANCZOS provides strong anti-aliasing quality.
            resample = PIL.Image.Resampling.LANCZOS
        else:
            # Upscaling: BICUBIC avoids some LANCZOS ringing artifacts.
            resample = PIL.Image.Resampling.BICUBIC
            logger.warning(
                "Output size is larger than transform size; upscaling image.",
            )

        image = image.resize(
            (options.output_size, options.output_size),
            resample,
        )

    return image
