"""Tests for dlib landmark model resolution."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from morphace.landmarks import (
    MODEL_ENV_VAR,
    MODEL_FILENAME,
    LandmarkModelNotFoundError,
    NoFaceFoundError,
    _load_predictor,
    default_landmark_model_path,
    detect_all_landmarks,
    dlib,
    get_detector,
    get_landmarks,
    get_predictor,
    resolve_landmark_model_path,
)


@dataclass
class FakePoint:
    """A simple class to mimic dlib's point object."""

    x: int
    y: int


class FakeShape:
    """A simple class to mimic dlib's shape object with parts()."""

    def parts(self) -> list[FakePoint]:
        """Return a fixed set of points for testing."""
        return [
            FakePoint(10, 20),
            FakePoint(30, 40),
            FakePoint(50, 60),
        ]


@pytest.fixture(autouse=True)
def clear_landmark_model_caches() -> Iterator[None]:
    """Clear cached dlib helpers before and after each test."""
    get_detector.cache_clear()
    _load_predictor.cache_clear()
    yield
    get_detector.cache_clear()
    _load_predictor.cache_clear()


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


def test_get_landmarks_loads_image_and_detects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify get_landmarks loads the image via dlib and calls detection."""
    fake_image_path = tmp_path / "test.jpg"
    fake_image_path.touch()

    mock_img = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_detector = Mock()
    mock_predictor = Mock(spec=dlib.shape_predictor)
    expected_landmarks = [[(10, 20), (30, 40)]]

    # Mock dlib.load_rgb_image
    load_mock = Mock(return_value=mock_img)
    monkeypatch.setattr(dlib, "load_rgb_image", load_mock)

    # Mock detect_all_landmarks to verify delegation
    detect_mock = Mock(return_value=iter(expected_landmarks))
    monkeypatch.setattr("morphace.landmarks.detect_all_landmarks", detect_mock)

    results = list(
        get_landmarks(
            str(fake_image_path),
            detector=mock_detector,
            predictor=mock_predictor,
        )
    )

    assert results == expected_landmarks
    load_mock.assert_called_once_with(str(fake_image_path.resolve()))
    detect_mock.assert_called_once_with(mock_img, mock_detector, mock_predictor)


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


def test_get_detector_loads_dlib_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify get_detector loads the dlib frontal face detector."""
    fake_detector = Mock(name="fake_detector")
    get_detector_mock = Mock(return_value=fake_detector)

    monkeypatch.setattr(
        dlib,
        "get_frontal_face_detector",
        get_detector_mock,
    )

    result = get_detector()

    assert result is fake_detector
    get_detector_mock.assert_called_once_with()


def test_get_detector_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_detector returns a cached instance on subsequent calls."""
    fake_detector = Mock(name="fake_detector")
    get_detector_mock = Mock(return_value=fake_detector)

    monkeypatch.setattr(
        dlib,
        "get_frontal_face_detector",
        get_detector_mock,
    )

    first = get_detector()
    second = get_detector()

    assert first is second
    get_detector_mock.assert_called_once_with()


def test_load_predictor_raises_if_model_missing(
    tmp_path: Path,
) -> None:
    """Verify _load_predictor raises FileNotFoundError when model is missing."""
    missing_path = tmp_path / "missing.dat"

    with pytest.raises(FileNotFoundError, match="Dlib model file not found"):
        _load_predictor(str(missing_path))


def test_load_predictor_loads_existing_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _load_predictor loads the model when the file exists."""
    fake_model = tmp_path / "shape_predictor_68_face_landmarks_GTX.dat"
    fake_model.write_bytes(b"not a real model, but path exists")

    fake_predictor = Mock(name="fake_predictor")
    shape_predictor_mock = Mock(return_value=fake_predictor)

    monkeypatch.setattr(
        dlib,
        "shape_predictor",
        shape_predictor_mock,
    )

    result = _load_predictor(str(fake_model))

    assert result is fake_predictor
    shape_predictor_mock.assert_called_once_with(str(fake_model))


def test_load_predictor_is_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _load_predictor returns a cached instance on subsequent calls."""
    fake_model = tmp_path / "shape_predictor_68_face_landmarks_GTX.dat"
    fake_model.write_bytes(b"not a real model")

    fake_predictor = Mock(name="fake_predictor")
    shape_predictor_mock = Mock(return_value=fake_predictor)

    monkeypatch.setattr(
        dlib,
        "shape_predictor",
        shape_predictor_mock,
    )

    first = _load_predictor(str(fake_model))
    second = _load_predictor(str(fake_model))

    assert first is second
    shape_predictor_mock.assert_called_once_with(str(fake_model))


def test_detect_all_landmarks_returns_landmarks_for_each_face() -> None:
    """Verify detect_all_landmarks yields correct landmarks for faces."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    fake_rect = Mock(name="fake_rect")

    detector = Mock(return_value=[fake_rect])
    predictor = Mock(spec=dlib.shape_predictor)
    predictor.return_value = FakeShape()

    result = list(
        detect_all_landmarks(
            img,
            detector,
            predictor,
        )
    )

    assert result == [
        [
            (10, 20),
            (30, 40),
            (50, 60),
        ]
    ]
    detector.assert_called_once_with(img, 1)
    predictor.assert_called_once_with(img, fake_rect)


def test_detect_all_landmarks_raises_when_no_faces_found() -> None:
    """Verify detect_all_landmarks raises NoFaceFoundError."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    detector = Mock(return_value=[])
    predictor = Mock(spec=dlib.shape_predictor)

    with pytest.raises(NoFaceFoundError, match="Unable to find a face"):
        list(
            detect_all_landmarks(
                img,
                detector,
                predictor,
            )
        )

    detector.assert_called_once_with(img, 1)
    predictor.assert_not_called()
