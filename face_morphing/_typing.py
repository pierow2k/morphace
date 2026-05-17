"""Shared type aliases for face morphing arrays and geometry."""

from os import PathLike
from typing import Any

from numpy.typing import NDArray

type Array = NDArray[Any]
type ImageArray = NDArray[Any]
type PointArray = NDArray[Any]

type FloatPoint = tuple[float, float]
type Point = tuple[int, int]
type Rect = tuple[int, int, int, int]
type Size = tuple[int, int]
type Triangle = tuple[int, int, int]

type LandmarkList = list[Point]
type TriangleList = list[Triangle]
type PathInput = str | PathLike[str]

type ImagePair = tuple[ImageArray, ImageArray]
type ImageShape = tuple[int, ...]
type FaceCorrespondences = tuple[
    Size,
    ImageArray,
    ImageArray,
    LandmarkList,
    LandmarkList,
    PointArray,
]
