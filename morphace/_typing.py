"""Shared type aliases for face morphing arrays and geometry."""

from os import PathLike
from typing import Any

from numpy.typing import NDArray

type ImageArray = NDArray[Any]
type LandmarkArray = NDArray[Any]

type FloatPoint = tuple[float, float]
type Point = tuple[int, int]
type Bounds = tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
type Size = tuple[int, int]
type Triangle = tuple[int, int, int]

type LandmarkList = list[Point]
type TriangleList = list[Triangle]
type PathInput = str | PathLike[str]

type ImagePair = tuple[ImageArray, ImageArray]
type ImageShape = tuple[int, ...]
