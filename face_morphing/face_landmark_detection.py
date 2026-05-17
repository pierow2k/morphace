"""Module for detecting facial landmarks and aligning images for morphing."""

import logging

import cv2
import dlib
import numpy as np

from ._typing import (
    FaceCorrespondences,
    ImageArray,
    ImagePair,
    LandmarkList,
    Size,
)

logger = logging.getLogger(__name__)


class NoFaceFoundError(Exception):
    """Raised when there is no face found."""


def _center_crop(img: ImageArray, target_h: int, target_w: int) -> ImageArray:
    """Crops the center of an image to the target dimensions."""
    h, w = img.shape[:2]
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return img[start_h : start_h + target_h, start_w : start_w + target_w]


def _crop_image(img1: ImageArray, img2: ImageArray) -> ImagePair:
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


def generate_face_correspondences(
    image1: ImageArray, image2: ImageArray
) -> FaceCorrespondences:
    """Detects facial landmarks and creates correspondence between images.

    Args:
        image1: The first input image.
        image2: The second input image.

    Raises:
        NoFaceFoundError: If dlib fails to detect a face in either image.

    Returns:
        Image size, cropped images, landmark points, and average landmark
        coordinates for both images.
    """
    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "face_morphing/utils/shape_predictor_68_face_landmarks.dat"
    )
    corresp = np.zeros((68, 2))

    img_list = _crop_image(image1, image2)
    list1: LandmarkList = []
    list2: LandmarkList = []
    j = 1
    size: Size = (img_list[0].shape[0], img_list[0].shape[1])

    for img in img_list:
        size = (img.shape[0], img.shape[1])
        curr_list = list1 if j == 1 else list2

        dets = detector(img, 1)

        if len(dets) == 0:
            logger.error("Unable to find a face in the image.")
            raise NoFaceFoundError("Unable to find a face in the image.")

        j = j + 1

        for _, rect in enumerate(dets):
            # Get the landmarks/parts for the face in rect.
            shape = predictor(img, rect)

            for i in range(68):
                x = shape.part(i).x
                y = shape.part(i).y
                curr_list.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y

            # Add back the background
            curr_list.append((1, 1))
            curr_list.append((size[1] - 1, 1))
            curr_list.append(((size[1] - 1) // 2, 1))
            curr_list.append((1, size[0] - 1))
            curr_list.append((1, (size[0] - 1) // 2))
            curr_list.append(((size[1] - 1) // 2, size[0] - 1))
            curr_list.append((size[1] - 1, size[0] - 1))
            curr_list.append(((size[1] - 1), (size[0] - 1) // 2))

    # Add back the background
    narray = corresp / 2
    narray = np.append(narray, [[1, 1]], axis=0)
    narray = np.append(narray, [[size[1] - 1, 1]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, 1]], axis=0)
    narray = np.append(narray, [[1, size[0] - 1]], axis=0)
    narray = np.append(narray, [[1, (size[0] - 1) // 2]], axis=0)
    narray = np.append(narray, [[(size[1] - 1) // 2, size[0] - 1]], axis=0)
    narray = np.append(narray, [[size[1] - 1, size[0] - 1]], axis=0)
    narray = np.append(narray, [[(size[1] - 1), (size[0] - 1) // 2]], axis=0)

    return (size, img_list[0], img_list[1], list1, list2, narray)
