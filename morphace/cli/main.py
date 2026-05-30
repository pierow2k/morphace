"""Canonical command-line interface for Morphace."""

import argparse
import sys
from collections.abc import Callable
from importlib.metadata import version
from typing import cast

from morphace.cli import download, morph, prep

CommandRunner = Callable[[argparse.Namespace], int]


def _build_parser() -> argparse.ArgumentParser:
    """Build the root Morphace command parser."""
    parser = argparse.ArgumentParser(
        prog="morphace",
        description="Create face morph videos and prepare source images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s " + version("morphace"),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        required=True,
    )

    morph_parser = subparsers.add_parser(
        "morph",
        add_help=False,
        help="Generate a face morphing video.",
        description="Generate a face morphing video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    morph.add_arguments(morph_parser)
    morph_parser.set_defaults(command_func=morph.run)

    prep_parser = subparsers.add_parser(
        "prep",
        add_help=False,
        help="Prepare aligned square PNG face crops.",
        description=(
            "Detect faces in raw images and prepare aligned square PNG crops "
            "for morphing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prep.add_arguments(prep_parser)
    prep_parser.set_defaults(command_func=prep.run)

    download_parser = subparsers.add_parser(
        "download",
        add_help=False,
        help="Download the dlib landmark model.",
        description="Download the dlib landmark model file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    download.add_arguments(download_parser)
    download_parser.set_defaults(command_func=download.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the canonical Morphace CLI."""
    args = _build_parser().parse_args(argv)
    command_func = cast("CommandRunner", args.command_func)
    return command_func(args)


if __name__ == "__main__":
    sys.exit(main())
