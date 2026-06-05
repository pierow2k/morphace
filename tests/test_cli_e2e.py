"""End-to-end tests for deterministic Morphace command output."""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from morphace.landmarks import MODEL_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = REPO_ROOT / "website" / "public" / "media"
EXAMPLE_YOUNG = MEDIA_DIR / "example-young.jpg"
EXAMPLE_OLD = MEDIA_DIR / "example-old.jpg"

MODEL_SHA256 = (
    "249a69a1d5f2d7c714a92934d35367d46eb52dc308d46717e82d49e8386b3b80"
)
PREP_SHA256 = (
    "c5f5e403407e5d43167c4de59853b8544b90a2f262c89f66a48668088c1d8ead"
)
MORPH_SHA256 = (
    "39c6b41ded9e8d5e17c5f7548882bc8cce3c9a316253240888dfbe41d2f2eef2"
)


def _run_morphace(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the canonical Morphace CLI dispatcher in a subprocess."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "morphace.cli.main", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def _assert_success(
    result: subprocess.CompletedProcess[str],
    command: str,
) -> None:
    """Assert a command succeeded with captured output on failure."""
    assert result.returncode == 0, (
        f"{command} failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    with path.open("rb") as file_obj:
        return hashlib.file_digest(file_obj, "sha256").hexdigest()


@pytest.mark.e2e
def test_morphace_commands_produce_deterministic_outputs(
    tmp_path: Path,
) -> None:
    """Verify download, prep, and morph produce deterministic artifacts."""
    assert shutil.which("ffmpeg") is not None, "ffmpeg is required for e2e"
    assert EXAMPLE_YOUNG.exists(), f"Missing sample image: {EXAMPLE_YOUNG}"
    assert EXAMPLE_OLD.exists(), f"Missing sample image: {EXAMPLE_OLD}"

    model_dir = tmp_path / "model"
    model_path = model_dir / MODEL_FILENAME
    prepared_dir = tmp_path / "prepared"
    prepared_image = prepared_dir / "example-young_face01.png"
    morph_video = tmp_path / "test_morph.mp4"

    download_result = _run_morphace(
        "download",
        "--force",
        "--save-to",
        str(model_dir),
    )
    _assert_success(download_result, "morphace download")
    assert _sha256(model_path) == MODEL_SHA256

    prep_result = _run_morphace(
        "prep",
        "--force",
        "--landmark-model",
        str(model_path),
        str(EXAMPLE_YOUNG),
        str(prepared_dir),
    )
    _assert_success(prep_result, "morphace prep")
    assert _sha256(prepared_image) == PREP_SHA256

    morph_result = _run_morphace(
        "morph",
        "--duration",
        "5",
        "--fps",
        "30",
        "--force",
        "--landmark-model",
        str(model_path),
        "--output",
        str(morph_video),
        str(EXAMPLE_YOUNG),
        str(EXAMPLE_OLD),
    )
    _assert_success(morph_result, "morphace morph")
    assert _sha256(morph_video) == MORPH_SHA256
