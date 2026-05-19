"""Configuration objects for face morphing."""

from dataclasses import dataclass
from pathlib import Path


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
