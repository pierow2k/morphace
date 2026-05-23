"""Configuration for face alignment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceAlignmentOptions:
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
