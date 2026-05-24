"""Tests for dlib landmark model resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from morphace.landmarks import (
    MODEL_ENV_VAR,
    MODEL_FILENAME,
    LandmarkModelNotFoundError,
    default_landmark_model_path,
    get_landmarks,
    get_predictor,
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
        pytest.raises(LandmarkModelNotFoundError) as exc_info,
    ):
        resolve_landmark_model_path()
    assert str(mock_default) in str(exc_info.value)


def test_get_predictor_raises_file_not_found() -> None:
    """Verify get_predictor FileNotFoundError when model file is missing."""
    # Clear the cache to ensure we're testing the logic, not returning a cached
    # result.

    from morphace.landmarks import (  # pylint: disable=import-outside-toplevel # noqa:PLC0415
        _load_predictor,
    )

    _load_predictor.cache_clear()

    with (
        patch.object(Path, "is_file", return_value=False),
        pytest.raises(FileNotFoundError, match="Dlib model file not found at"),
    ):
        get_predictor(Path("/missing/model.dat"))


def test_get_landmarks_requires_models() -> None:
    """Verify get_landmarks raises RuntimeError when models are missing."""
    with pytest.raises(RuntimeError, match="Dlib models are not loaded"):
        # get_landmarks is a generator. We must attempt to iterate to trigger
        # the error check inside the function body.
        next(get_landmarks("dummy.jpg", detector=None, predictor=None))


def test_default_landmark_model_path() -> None:
    """Verify default_landmark_model_path uses platformdirs correctly."""
    fake_base = Path("/fake/user/data")
    with patch(
        "morphace.landmarks.user_data_path", return_value=fake_base
    ) as mock_user_data:
        result = default_landmark_model_path()
        assert result == fake_base / MODEL_FILENAME
        mock_user_data.assert_called_once_with(
            appname="morphace", appauthor=False, ensure_exists=True
        )
