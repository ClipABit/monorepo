"""
Shared pytest fixtures for ClipABit test suite.

Fixtures are reusable test setup/data automatically available to all tests.
Just add fixture name as a function parameter to use it.

Types:
    - Paths: Directories for temp files
    - Videos: Test video files
    - Data: Raw test data (frames, arrays, metadata objects)
    - Components: Pre-configured class instances ready to use
    - Mocks: Fake external dependencies to test in isolation without side effects
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import subprocess
from unittest.mock import MagicMock, patch
import sys
import importlib

from preprocessing.chunker import Chunker
from preprocessing.frame_extractor import FrameExtractor
from preprocessing.compressor import Compressor
from preprocessing.preprocessor import Preprocessor
from models.metadata import VideoChunk
from database.pinecone_connector import PineconeConnector
from database.r2_connector import R2Connector


# ==============================================================================
# PATH FIXTURES
# ==============================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory that auto-cleans after test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


# ==============================================================================
# VIDEO FIXTURES (Auto-generated with OpenCV)
# ==============================================================================

@pytest.fixture(scope="session")
def sample_video_5s(tmp_path_factory) -> Path:
    """5-second test video with mixed motion patterns."""
    import cv2

    video_dir = tmp_path_factory.mktemp("test_videos")
    video_path = video_dir / "sample_5s.mp4"
    
    # Create a simple video using OpenCV
    fps, duration = 30, 5
    width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for frame_num in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        color_shift = (frame_num * 2) % 255
        frame[:, :] = (color_shift, 100 + color_shift // 2, 150)

        cv2.circle(frame, (320 + frame_num * 2, 240), 50, (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 100 + frame_num), (200, 200 + frame_num), (0, 255, 0), 2)

        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture(scope="session")
def sample_video_h264(tmp_path_factory) -> Path:
    """1-second H.264 test video generated with ffmpeg."""
    video_dir = tmp_path_factory.mktemp("test_videos_h264")
    video_path = video_dir / "sample_h264.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
        "-c:v", "libx264",
        "-preset", "fast",
        str(video_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate H.264 video: {e.stderr.decode()}")
    except FileNotFoundError:
        pytest.skip("ffmpeg not found, skipping H.264 test")
        
    return video_path


@pytest.fixture(scope="session")
def sample_video_vp9(tmp_path_factory) -> Path:
    """1-second VP9 test video generated with ffmpeg."""
    video_dir = tmp_path_factory.mktemp("test_videos_vp9")
    video_path = video_dir / "sample_vp9.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
        "-c:v", "libvpx-vp9",
        "-b:v", "0", "-crf", "30",
        str(video_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate VP9 video: {e.stderr.decode()}")
    except FileNotFoundError:
        pytest.skip("ffmpeg not found, skipping VP9 test")
        
    return video_path


@pytest.fixture(scope="session")
def sample_video_av1(tmp_path_factory) -> Path:
    """1-second AV1 test video generated with ffmpeg."""
    video_dir = tmp_path_factory.mktemp("test_videos_av1")
    video_path = video_dir / "sample_av1.mp4"
    
    # Generate AV1 video using ffmpeg
    # -f lavfi -i testsrc=duration=1:size=320x240:rate=30
    # -c:v libsvtav1 -preset 8 -crf 50
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
        "-c:v", "libsvtav1",
        "-preset", "8",
        "-crf", "50",
        str(video_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate AV1 video (ffmpeg might not support libsvtav1): {e.stderr.decode()}")
    except FileNotFoundError:
        pytest.skip("ffmpeg not found, skipping AV1 test")
        
    return video_path


@pytest.fixture(scope="session")
def sample_video_static(tmp_path_factory) -> Path:
    """10-second static video with minimal motion."""
    import cv2

    video_dir = tmp_path_factory.mktemp("test_videos")
    video_path = video_dir / "sample_static.mp4"

    fps, duration = 30, 10
    width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for frame_num in range(fps * duration):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        noise = np.random.randint(-2, 3, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def sample_video_bytes(sample_video_5s) -> bytes:
    """Video file as bytes for upload testing."""
    return sample_video_5s.read_bytes()


# ==============================================================================
# DATA FIXTURES
# ==============================================================================

@pytest.fixture
def sample_frame() -> np.ndarray:
    """Single 640x480 BGR frame. Shape: (480, 640, 3)"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames() -> np.ndarray:
    """Array of 10 frames. Shape: (10, 480, 640, 3)"""
    return np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_video_chunk() -> VideoChunk:
    """Basic VideoChunk for testing."""
    return VideoChunk(
        chunk_id="test_video_chunk_0000",
        start_time=0.0,
        end_time=5.0
    )


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Sample embedding vector for testing (512-dimensional, typical CLIP embedding size)."""
    return np.random.rand(512).astype(np.float32)


# ==============================================================================
# COMPONENT FIXTURES
# ==============================================================================

@pytest.fixture
def chunker() -> Chunker:
    """Chunker with test configuration."""
    return Chunker(
        min_duration=1.0,
        max_duration=10.0,
        scene_threshold=13.0
    )


@pytest.fixture
def frame_extractor() -> FrameExtractor:
    """FrameExtractor with test configuration."""
    return FrameExtractor(
        min_fps=0.5,
        max_fps=2.0,
        motion_threshold=25.0
    )


@pytest.fixture
def compressor() -> Compressor:
    """Compressor with test configuration."""
    return Compressor(
        target_width=640,
        target_height=480
    )


@pytest.fixture
def preprocessor() -> Preprocessor:
    """Preprocessor with test configuration."""
    return Preprocessor(
        min_chunk_duration=1.0,
        max_chunk_duration=10.0,
        scene_threshold=13.0,
        min_sampling_fps=0.5,
        max_sampling_fps=2.0,
        motion_threshold=25.0,
        target_width=640,
        target_height=480
    )


# ==============================================================================
# MOCKS
# ==============================================================================

@pytest.fixture
def mock_modal_dict(mocker):
    """Mock Modal Dict with Python dict behavior."""
    fake_dict = {}
    mock_dict = mocker.MagicMock()

    def getitem(_, key):
        return fake_dict[key]

    def setitem(_, key, value):
        fake_dict[key] = value

    def delitem(_, key):
        del fake_dict[key]

    def contains(_, key):
        return key in fake_dict

    def get(_, key, default=None):
        return fake_dict.get(key, default)

    mock_dict.__getitem__ = getitem
    mock_dict.__setitem__ = setitem
    mock_dict.__delitem__ = delitem
    mock_dict.__contains__ = contains
    mock_dict.get = get
    mock_dict.keys.side_effect = fake_dict.keys

    mocker.patch('modal.Dict.from_name', return_value=mock_dict)
    return fake_dict


@pytest.fixture
def mock_pinecone_connector(mocker):
    """Mock PineconeConnector with all necessary mocks set up"""
    
    mock_pinecone = mocker.patch('database.pinecone_connector.Pinecone')
    mock_client = mocker.MagicMock()
    mock_index = mocker.MagicMock()
    mock_pinecone.return_value = mock_client
    mock_client.Index.return_value = mock_index
    
    connector = PineconeConnector(api_key="test-key", index_name="test-index")
    
    return connector, mock_index, mock_client, mock_pinecone


@pytest.fixture
def mock_r2_connector(mocker, mock_modal_dict):
    """Mock R2Connector with all necessary mocks set up"""
    mock_boto3 = mocker.patch('database.r2_connector.boto3')
    mock_client = mocker.MagicMock()
    mock_boto3.client.return_value = mock_client
    
    connector = R2Connector(
        account_id="test-account",
        access_key_id="test-key",
        secret_access_key="test-secret",
        environment="test"
    )
    
    return connector, mock_client, mock_boto3


@pytest.fixture
def server_instance(mocker):
    """
    Creates a Server instance with all dependencies mocked.
    We bypass the actual startup() logic and manually inject mocks.
    """
    # Create a mock for the modal module
    mock_modal = MagicMock()
    
    # Configure the mock decorators to just return the original class/function
    def identity_decorator(*args, **kwargs):
        def wrapper(obj):
            return obj
        return wrapper
    
    # Handle @app.cls() -> returns decorator -> returns class
    mock_modal.App.return_value.cls.side_effect = identity_decorator
    
    # Handle @modal.method() -> returns decorator -> returns function
    mock_modal.method.side_effect = identity_decorator
    
    # Handle @modal.enter() -> returns decorator -> returns function
    mock_modal.enter.side_effect = identity_decorator
    
    # Handle @modal.fastapi_endpoint() -> returns decorator -> returns function
    mock_modal.fastapi_endpoint.side_effect = identity_decorator

    # Patch sys.modules to use our mock_modal
    with patch.dict(sys.modules, {'modal': mock_modal}):
        # Now import main. It will use the mocked modal.
        # We need to reload it if it was already imported
        if 'main' in sys.modules:
            import main
            importlib.reload(main)
        else:
            import main
        
        # Now Server is a regular Python class, not a Modal wrapped one
        server = main.Server()
        
        # Mock all the components that would be set in startup()
        server.r2_connector = mocker.MagicMock()
        server.pinecone_connector = mocker.MagicMock()
        server.preprocessor = mocker.MagicMock()
        server.video_embedder = mocker.MagicMock()
        server.job_store = mocker.MagicMock()
        server.searcher = mocker.MagicMock()
        
        yield server


# ==============================================================================
# PYTEST CONFIG
# ==============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(session, config, items):
    """Ensure API endpoint tests execute after the rest for isolation."""
    api_items = [item for item in items if "tests/integration/test_api_endpoints.py" in item.nodeid]
    if not api_items:
        return
    remaining = [item for item in items if item not in api_items]
    items[:] = remaining + api_items
