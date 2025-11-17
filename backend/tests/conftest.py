"""Shared pytest fixtures for backend tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Create a temporary video file path for testing."""
    video_path = tmp_path / "test_video.mp4"
    # Note: Tests should mock actual video operations
    # This just provides a valid path
    return video_path


@pytest.fixture
def sample_frames() -> np.ndarray:
    """Generate sample frame data for testing."""
    # Create 5 frames of 480x640 RGB images
    return np.random.randint(0, 255, size=(5, 480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_video_metadata() -> dict:
    """Sample video metadata for testing."""
    return {
        "filename": "test_video.mp4",
        "duration": 24.5,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "frame_count": 735,
    }


@pytest.fixture
def sample_chunk_data() -> dict:
    """Sample chunk data for testing."""
    return {
        "chunk_id": "video123_chunk_0",
        "video_id": "video123",
        "start_time": 0.0,
        "end_time": 5.2,
        "duration": 5.2,
        "frame_count": 10,
        "sampling_fps": 1.92,
        "complexity_score": 0.65,
    }


@pytest.fixture
def sample_job_data() -> dict:
    """Sample job data for testing."""
    return {
        "job_id": "test-job-123",
        "status": "processing",
        "filename": "test_video.mp4",
        "file_size": 3887170,
        "created_at": "2025-11-16T12:00:00",
    }


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_modal_dict(mocker) -> MagicMock:
    """Mock Modal Dict for job storage testing."""
    mock_dict = MagicMock()
    mock_dict._storage = {}  # Internal storage for testing

    # Mock the dict-like interface
    def getitem(key):
        return mock_dict._storage.get(key)

    def setitem(key, value):
        mock_dict._storage[key] = value

    def contains(key):
        return key in mock_dict._storage

    mock_dict.__getitem__ = getitem
    mock_dict.__setitem__ = setitem
    mock_dict.__contains__ = contains
    mock_dict.get = lambda key, default=None: mock_dict._storage.get(key, default)
    mock_dict.pop = lambda key, default=None: mock_dict._storage.pop(key, default)

    return mock_dict


@pytest.fixture
def mock_pinecone_index(mocker) -> MagicMock:
    """Mock Pinecone index for database testing."""
    mock_index = MagicMock()
    mock_index.upsert = MagicMock(return_value={"upserted_count": 1})
    mock_index.query = MagicMock(return_value={
        "matches": [
            {
                "id": "chunk_0",
                "score": 0.95,
                "metadata": {"video_id": "video123", "start_time": 0.0}
            }
        ]
    })
    return mock_index


@pytest.fixture
def mock_cv2_video_capture(mocker) -> MagicMock:
    """Mock OpenCV VideoCapture for video processing tests."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 30.0,   # CAP_PROP_FPS
        7: 735,    # CAP_PROP_FRAME_COUNT
        3: 1920,   # CAP_PROP_FRAME_WIDTH
        4: 1080,   # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0.0)

    # Mock frame reading
    frame = np.random.randint(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, frame)
    mock_cap.release = MagicMock()

    return mock_cap


@pytest.fixture
def mock_scene_manager(mocker) -> MagicMock:
    """Mock PySceneDetect SceneManager for chunking tests."""
    mock_manager = MagicMock()

    # Mock scene detection results
    # Returns list of tuples: (start_frame, end_frame)
    mock_manager.get_scene_list.return_value = [
        (0, 90),      # 0-3s at 30fps
        (90, 240),    # 3-8s
        (240, 450),   # 8-15s
        (450, 735),   # 15-24.5s
    ]

    return mock_manager


# ============================================================================
# Temporary File/Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars(monkeypatch) -> None:
    """Mock environment variables for testing."""
    monkeypatch.setenv("PINECONE_API_KEY", "test-api-key-123")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-chunks-index")


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
def mock_modal_decorators(mocker):
    """Mock Modal decorators for testing Modal app functions."""
    # Mock the decorators to be pass-through functions
    mocker.patch("modal.app", return_value=lambda cls: cls)
    mocker.patch("modal.method", return_value=lambda func: func)
    mocker.patch("modal.enter", return_value=lambda func: func)
    mocker.patch("modal.fastapi_endpoint", return_value=lambda func: func)


@pytest.fixture
def mock_job_store_connector(mock_modal_dict) -> MagicMock:
    """Mock JobStoreConnector with working dict storage."""
    from database.job_store_connector import JobStoreConnector

    mock_connector = MagicMock(spec=JobStoreConnector)
    mock_connector.job_dict = mock_modal_dict

    # Implement basic CRUD operations
    def create_job(job_id: str, initial_data: dict) -> bool:
        mock_modal_dict[job_id] = initial_data
        return True

    def get_job(job_id: str):
        return mock_modal_dict.get(job_id)

    def update_job(job_id: str, update_data: dict) -> bool:
        if job_id in mock_modal_dict:
            mock_modal_dict[job_id].update(update_data)
            return True
        return False

    mock_connector.create_job = create_job
    mock_connector.get_job = get_job
    mock_connector.update_job = update_job
    mock_connector.job_exists = lambda job_id: job_id in mock_modal_dict

    return mock_connector


@pytest.fixture
def mock_pinecone_connector(mock_pinecone_index) -> MagicMock:
    """Mock PineconeConnector with working index operations."""
    from database.pinecone_connector import PineconeConnector

    mock_connector = MagicMock(spec=PineconeConnector)
    mock_connector.index = mock_pinecone_index

    return mock_connector
