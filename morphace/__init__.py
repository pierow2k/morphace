# __init__.py

"""Public API for the morphace package."""

from .landmarks import NoFaceFoundError
from .morphing import MorphConfig, morph_faces

__all__ = ["MorphConfig", "NoFaceFoundError", "morph_faces"]
