"""Tests for the canonical Morphace command-line dispatcher."""

import argparse
from pathlib import Path

import pytest

from morphace.cli import main as cli_main

ARGPARSE_USAGE_ERROR = 2
DOWNLOAD_RETURN_CODE = 5
MORPH_DURATION = 9
MORPH_FPS = 24
MORPH_RETURN_CODE = 7
PREP_OUTPUT_SIZE = 512
PREP_RETURN_CODE = 3


def test_main_dispatches_morph_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the morph subcommand parses arguments and dispatches."""
    captured_args: list[argparse.Namespace] = []

    def fake_run(args: argparse.Namespace) -> int:
        """Capture parsed morph arguments."""
        captured_args.append(args)
        return MORPH_RETURN_CODE

    monkeypatch.setattr(cli_main.morph, "run", fake_run)

    result = cli_main.main(
        [
            "morph",
            "first.png",
            "second.png",
            "--duration",
            str(MORPH_DURATION),
            "--fps",
            str(MORPH_FPS),
            "--show-mesh",
        ],
    )

    assert result == MORPH_RETURN_CODE
    assert captured_args[0].command == "morph"
    assert captured_args[0].img1 == "first.png"
    assert captured_args[0].img2 == "second.png"
    assert captured_args[0].duration == MORPH_DURATION
    assert captured_args[0].fps == MORPH_FPS
    assert captured_args[0].show_mesh is True


def test_main_dispatches_prep_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the prep subcommand parses arguments and dispatches."""
    captured_args: list[argparse.Namespace] = []

    def fake_run(args: argparse.Namespace) -> int:
        """Capture parsed prep arguments."""
        captured_args.append(args)
        return PREP_RETURN_CODE

    monkeypatch.setattr(cli_main.prep, "run", fake_run)

    result = cli_main.main(
        [
            "prep",
            "raw",
            "aligned",
            "--output-size",
            str(PREP_OUTPUT_SIZE),
            "--alpha",
        ],
    )

    assert result == PREP_RETURN_CODE
    assert captured_args[0].command == "prep"
    assert captured_args[0].source == "raw"
    assert captured_args[0].aligned_dir == "aligned"
    assert captured_args[0].output_size == PREP_OUTPUT_SIZE
    assert captured_args[0].alpha is True


def test_main_dispatches_download_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the download subcommand parses arguments and dispatches."""
    captured_args: list[argparse.Namespace] = []

    def fake_run(args: argparse.Namespace) -> int:
        """Capture parsed download arguments."""
        captured_args.append(args)
        return DOWNLOAD_RETURN_CODE

    monkeypatch.setattr(cli_main.download, "run", fake_run)

    result = cli_main.main(
        ["download", "--save-to", "model", "--force", "--debug"],
    )

    assert result == DOWNLOAD_RETURN_CODE
    assert captured_args[0].command == "download"
    assert captured_args[0].save_to == Path("model")
    assert captured_args[0].overwrite is True
    assert captured_args[0].debug is True


def test_main_requires_subcommand() -> None:
    """Verify the root command rejects missing subcommands."""
    with pytest.raises(SystemExit) as error:
        cli_main.main([])

    assert error.value.code == ARGPARSE_USAGE_ERROR


def test_main_rejects_unknown_subcommand() -> None:
    """Verify the root command rejects unknown subcommands."""
    with pytest.raises(SystemExit) as error:
        cli_main.main(["unknown"])

    assert error.value.code == ARGPARSE_USAGE_ERROR


def test_main_rejects_legacy_bare_morph_arguments() -> None:
    """Verify bare morph arguments are not treated as a legacy alias."""
    with pytest.raises(SystemExit) as error:
        cli_main.main(["first.png", "second.png"])

    assert error.value.code == ARGPARSE_USAGE_ERROR


def test_main_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the root command exposes package version output."""
    with pytest.raises(SystemExit) as error:
        cli_main.main(["--version"])

    assert error.value.code == 0
    assert capsys.readouterr().out.startswith("morphace ")
