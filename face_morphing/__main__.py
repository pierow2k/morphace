# pylint: disable=invalid-name

"""Allow running the package as a module.

python -m face_morphing
"""

from .cli import main

if __name__ == "__main__":
    main()
