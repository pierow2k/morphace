"""Tests for dlib landmark model resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from morphace.landmarks import (
    MODEL_ENV_VAR,
    LandmarkModelNotFoundError,
    resolve_landmark_model_path,
)


def test_resolve_landmark_model_path_from_argument() -> None:
    """Test resolution when an explicit path is provided."""
    custom_path = "/path/to/model.dat"
    with (
        patch.object(Path, "is_file", return_value=True),
        patch.object(Path, "expanduser", return_value=Path(custom_path)),
    ):
        result = resolve_landmark_model_path(custom_path)
        assert result == Path(custom_path)


def test_resolve_landmark_model_path_argument_not_found() -> None:
    """Test error when the provided explicit path does not exist."""
    with (
        patch.object(Path, "is_file", return_value=False),
        pytest.raises(LandmarkModelNotFoundError, match="--landmark-model"),
    ):
        resolve_landmark_model_path("/missing/path.dat")


def test_resolve_landmark_model_path_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test resolution via environment variable."""
    env_path = "/env/path/model.dat"
    monkeypatch.setenv(MODEL_ENV_VAR, env_path)

    with (
        patch.object(Path, "is_file", return_value=True),
        patch.object(Path, "expanduser", return_value=Path(env_path)),
    ):
        result = resolve_landmark_model_path()
        assert result == Path(env_path)


def test_resolve_landmark_model_path_from_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test resolution via default user data directory."""
    # Ensure env var doesn't interfere
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    default_path = Path("/default/user/path/model.dat")

    with (
        patch(
            "morphace.landmarks.default_landmark_model_path",
            return_value=default_path,
        ),
        patch.object(Path, "is_file", return_value=True),
    ):
        result = resolve_landmark_model_path()
        assert result == default_path


def test_resolve_landmark_model_path_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test error when no model can be found anywhere."""
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    # Mock default path to something specific so we can verify the error message
    mock_default = Path("/app/data/model.dat")
    with (
        patch(
            "morphace.landmarks.default_landmark_model_path",
            return_value=mock_default,
        ),
        patch.object(Path, "is_file", return_value=False),
    ):
        with pytest.raises(LandmarkModelNotFoundError) as exc_info:
            resolve_landmark_model_path()
        assert str(mock_default) in str(exc_info.value)
