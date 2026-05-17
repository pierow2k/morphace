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
    MarginInfo,
    Size,
)

logger = logging.getLogger(__name__)


class NoFaceFoundError(Exception):
    """Raised when there is no face found."""


def calculate_margin_help(img1: ImageArray, img2: ImageArray) -> MarginInfo:
    """Calculates dimensions and offsets required to align two images.

    Args:
        img1 (np.ndarray): The first input image.
        img2 (np.ndarray): The second input image.

    Returns:
        list: A list containing [size1, size2, diff0, diff1, avg0, avg1].
    """
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0] - size2[0]) // 2
    diff1 = abs(size1[1] - size2[1]) // 2
    avg0 = (size1[0] + size2[0]) // 2
    avg1 = (size1[1] + size2[1]) // 2

    return (size1, size2, diff0, diff1, avg0, avg1)


def crop_image(img1: ImageArray, img2: ImageArray) -> ImagePair:
    """Resizes and crops two images so they have matching dimensions.

    Args:
        img1 (np.ndarray): The first input image.
        img2 (np.ndarray): The second input image.

    Returns:
        list: A list containing the two processed images [img1, img2].
    """
    [size1, size2, diff0, diff1, avg0, avg1] = calculate_margin_help(img1, img2)

    if size1[0] == size2[0] and size1[1] == size2[1]:
        return (img1, img2)

    if size1[0] <= size2[0] and size1[1] <= size2[1]:
        scale0 = size1[0] / size2[0]
        scale1 = size1[1] / size2[1]
        if scale0 > scale1:
            res = cv2.resize(
                img2, None, fx=scale0, fy=scale0, interpolation=cv2.INTER_AREA
            )
        else:
            res = cv2.resize(
                img2, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_AREA
            )
        return crop_image_help(img1, res)

    if size1[0] >= size2[0] and size1[1] >= size2[1]:
        scale0 = size2[0] / size1[0]
        scale1 = size2[1] / size1[1]
        if scale0 > scale1:
            res = cv2.resize(
                img1, None, fx=scale0, fy=scale0, interpolation=cv2.INTER_AREA
            )
        else:
            res = cv2.resize(
                img1, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_AREA
            )
        return crop_image_help(res, img2)

    if size1[0] >= size2[0] and size1[1] <= size2[1]:
        return (img1[diff0:avg0, :], img2[:, -diff1:avg1])

    return (img1[:, diff1:avg1], img2[-diff0:avg0, :])


def crop_image_help(img1: ImageArray, img2: ImageArray) -> ImagePair:
    """Helper function to perform cropping of images based on margins.

    Args:
        img1 (np.ndarray): The first input image.
        img2 (np.ndarray): The second input image.

    Returns:
        list: A list containing the two cropped images.
    """
    [size1, size2, diff0, diff1, avg0, avg1] = calculate_margin_help(img1, img2)

    if size1[0] == size2[0] and size1[1] == size2[1]:
        return (img1, img2)

    if size1[0] <= size2[0] and size1[1] <= size2[1]:
        return (img1, img2[-diff0:avg0, -diff1:avg1])

    if size1[0] >= size2[0] and size1[1] >= size2[1]:
        return (img1[diff0:avg0, diff1:avg1], img2)

    if size1[0] >= size2[0] and size1[1] <= size2[1]:
        return (img1[diff0:avg0, :], img2[:, -diff1:avg1])

    return (img1[:, diff1:avg1], img2[diff0:avg0, :])


def generate_face_correspondences(
    image1: ImageArray, image2: ImageArray
) -> FaceCorrespondences:
    """Detects facial landmarks and creates correspondence between images.

    Args:
        image1 (np.ndarray): The first input image.
        image2 (np.ndarray): The second input image.

    Raises:
        NoFaceFoundError: If dlib fails to detect a face in either image.

    Returns:
        list: A list containing [size, img1, img2, points1, points2, narray]
            where narray contains the average coordinates of the landmarks
            for both images.
    """
    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "face_morphing/utils/shape_predictor_68_face_landmarks.dat"
    )
    corresp = np.zeros((68, 2))

    img_list = crop_image(image1, image2)
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

        for _k, rect in enumerate(dets):
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
