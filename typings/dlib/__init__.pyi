from collections.abc import Sequence
from os import PathLike
from typing import Any

from numpy.typing import NDArray

# ruff: noqa: N801

class point:
    x: int
    y: int

class rectangle: ...

class full_object_detection:
    def part(self, idx: int) -> point: ...
    def parts(self) -> Sequence[point]: ...

class frontal_face_detector:
    def __call__(
        self, image: NDArray[Any], upsample_num_times: int = ...
    ) -> Sequence[rectangle]: ...

class shape_predictor:
    def __init__(self, predictor_model_path: str) -> None: ...
    def __call__(
        self, image: NDArray[Any], box: rectangle
    ) -> full_object_detection: ...

def get_frontal_face_detector() -> frontal_face_detector: ...
def load_rgb_image(path: str | PathLike[str]) -> NDArray[Any]: ...
