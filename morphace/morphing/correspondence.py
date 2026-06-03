"""Module for detecting facial landmarks and aligning images for morphing."""

from dataclasses import dataclass
from typing import Any

import cv2
import dlib
import numpy as np

from morphace._typing import (
    ImageArray,
    ImagePair,
    LandmarkArray,
    LandmarkList,
    Size,
)
from morphace.landmarks import NoFaceFoundError, detect_all_landmarks

__all__ = ["FaceCorrespondences", "NoFaceFoundError", "align_faces"]
_EXPECTED_LANDMARK_COUNT = 68


@dataclass(frozen=True)
class FaceCorrespondences:
    """Images and point correspondences used by the morphing pipeline.

    Attributes:
        size: Cropped frame size as ``(height, width)``.
        image1: First image cropped to the shared frame size.
        image2: Second image cropped to the shared frame size.
        points1: Landmark and boundary points for the first image.
        points2: Landmark and boundary points for the second image.
        average_landmarks: Average landmark positions for triangulation.
    """

    size: Size
    image1: ImageArray
    image2: ImageArray
    points1: LandmarkList
    points2: LandmarkList
    average_landmarks: LandmarkArray


def _center_crop(img: ImageArray, target_h: int, target_w: int) -> ImageArray:
    """Crops the center of an image to the target dimensions."""
    h, w = img.shape[:2]
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return img[start_h : start_h + target_h, start_w : start_w + target_w]


def _match_image_sizes(img1: ImageArray, img2: ImageArray) -> ImagePair:
    """Resizes and crops two images so they have matching dimensions.

    Strategy: If one image is smaller in both dimensions, the larger image
    is downscaled to cover the smaller image, and then center-cropped to
    exact dimensions. If images are mixed sizes, both are center-cropped
    to their overlapping minimum dimensions.

    Args:
        img1: The first input image.
        img2: The second input image.

    Returns:
        The two processed images in input order.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 == h2 and w1 == w2:
        return (img1, img2)

    # Case 1: img1 is strictly smaller. Downscale img2 to cover img1.
    if h1 <= h2 and w1 <= w2:
        scale = max(h1 / h2, w1 / w2)
        # Calculate explicit dsize to avoid float rounding issues, ensuring
        # the new dimensions are at least the target dimensions.
        new_w2 = max(int(w2 * scale), w1)
        new_h2 = max(int(h2 * scale), h1)
        res2 = cv2.resize(img2, (new_w2, new_h2), interpolation=cv2.INTER_AREA)
        return (img1, _center_crop(res2, h1, w1))

    # Case 2: img2 is strictly smaller. Downscale img1 to cover img2.
    if h1 >= h2 and w1 >= w2:
        scale = max(h2 / h1, w2 / w1)
        new_w1 = max(int(w1 * scale), w2)
        new_h1 = max(int(h1 * scale), h2)
        res1 = cv2.resize(img1, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
        return (_center_crop(res1, h2, w2), img2)

    # Case 3: Mixed dimensions. Crop both to the minimum shared dimensions.
    target_h = min(h1, h2)
    target_w = min(w1, w2)
    return (
        _center_crop(img1, target_h, target_w),
        _center_crop(img2, target_h, target_w),
    )


def _get_boundary_points(h: int, w: int) -> list:
    """Generates 8 boundary points for image corners and midpoints."""
    return [
        (1, 1),  # Top-left
        (w - 1, 1),  # Top-right
        (w - 1, h - 1),  # Bottom-right
        (1, h - 1),  # Bottom-left
        ((w - 1) // 2, 1),  # Top-mid
        (w - 1, (h - 1) // 2),  # Right-mid
        ((w - 1) // 2, h - 1),  # Bottom-mid
        (1, (h - 1) // 2),  # Left-mid
    ]


def align_faces(
    image1: ImageArray,
    image2: ImageArray,
    detector: Any | None = None,
    predictor: dlib.shape_predictor | None = None,
) -> FaceCorrespondences:
    """Detects facial landmarks and creates correspondence between images.

    Args:
        image1: The first input image.
        image2: The second input image.
        detector: Optional pre-loaded dlib detector. Uses global if None.
        predictor: Optional pre-loaded dlib predictor. Uses global if None.

    Raises:
        NoFaceFoundError: If dlib fails to detect a face in either image.
        RuntimeError: If dlib models are not loaded.

    Returns:
        Image size, cropped images, landmark points, and average landmark
        coordinates for both images.
    """
    if detector is None or predictor is None:
        message = "Dlib models are not loaded. Cannot process faces."
        raise RuntimeError(message)

    # Crop images to matching dimensions.
    img_list = _match_image_sizes(image1, image2)
    img1_cropped, img2_cropped = img_list

    # Initialize storage
    list1: LandmarkList = []
    list2: LandmarkList = []
    corresp = np.zeros((_EXPECTED_LANDMARK_COUNT, 2))

    # Image size is guaranteed identical by _crop_image.
    h, w = img1_cropped.shape[:2]
    size: Size = (h, w)

    # Process both images using a loop.
    # Zip the images with their corresponding output lists
    for img, out_list in zip(
        [img1_cropped, img2_cropped],
        [list1, list2],
        strict=True,
    ):
        # Only process the primary face. Processing multiple faces would
        # break the point correspondence math.
        face_points = next(detect_all_landmarks(img, detector, predictor))

        # Extract landmarks
        for i, (x, y) in enumerate(face_points):
            out_list.append((x, y))
            corresp[i][0] += x
            corresp[i][1] += y

        # Add boundary points to the individual list
        out_list.extend(_get_boundary_points(h, w))

    # Compute average landmarks. corresp currently holds sum of points
    # from img1 + img2
    narray = corresp / 2

    # Append boundary points to the average (they are identical for both
    # images) Note: Boundary points are static, so the average is just the
    # boundary points themselves.
    boundary_points = _get_boundary_points(h, w)

    # Efficiently append multiple rows to numpy array.
    # Note: The order of points added here must match the order in
    # _get_boundary_points
    narray = np.vstack([narray, boundary_points])

    return FaceCorrespondences(
        size=size,
        image1=img1_cropped,
        image2=img2_cropped,
        points1=list1,
        points2=list2,
        average_landmarks=narray,
    )
