"""Configuration objects for face morphing."""

from dataclasses import dataclass
from pathlib import Path

from morphace._typing import Size


@dataclass(frozen=True)
class MorphConfig:
    """Configuration for morph video output.

    Args:
        duration: Duration of the morphing sequence in seconds.
        frame_rate: Number of frames per second.
        output: Path to save the output video.
        landmark_model_path: Path to the dlib landmark model.
    """

    duration: int
    frame_rate: int
    output: str
    landmark_model_path: Path


@dataclass(frozen=True)
class MorphVideoConfig:
    """Configuration for the generated video stream.

    Attributes:
        duration: Duration of the morphing sequence in seconds.
        frame_rate: Number of frames per second.
        size: Frame size as ``(height, width)``.
        output: Path to save the output video.
    """

    duration: int
    frame_rate: int
    size: Size
    output: str
