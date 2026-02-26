import io
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import modal
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi import Request

from api.server_fastapi_router import ServerFastAPIRouter


class FakeJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, job_id: str, data: Dict[str, Any]) -> None:
        # Add backward compatible fields if not present
        if "job_type" not in data:
            data["job_type"] = "video"
        if "parent_batch_id" not in data:
            data["parent_batch_id"] = None
        self._jobs[job_id] = data

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        return self._jobs.get(job_id)

    def set_job_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        self._jobs[job_id] = result

    def set_job_failed(self, job_id: str, error: str) -> None:
        self._jobs[job_id] = {"job_id": job_id, "status": "failed", "error": error}

    def create_batch_job(
        self, batch_job_id: str, child_job_ids: List[str], namespace: str
    ) -> bool:
        """Create a new batch job entry."""
        batch_data = {
            "batch_job_id": batch_job_id,
            "job_type": "batch",
            "status": "processing",
            "namespace": namespace,
            "total_videos": len(child_job_ids),
            "child_job_ids": child_job_ids,
            "completed_count": 0,
            "failed_count": 0,
            "processing_count": len(child_job_ids),
        }
        self._jobs[batch_job_id] = batch_data
        return True

    def update_batch_on_child_completion(
        self, batch_job_id: str, child_job_id: str, child_result: Dict[str, Any]
    ) -> bool:
        """Update batch job when a child completes."""
        if batch_job_id not in self._jobs:
            return False

        batch_job = self._jobs[batch_job_id]
        child_status = child_result.get("status")

        if child_status == "completed":
            batch_job["completed_count"] += 1
            batch_job["processing_count"] -= 1
        elif child_status == "failed":
            batch_job["failed_count"] += 1
            batch_job["processing_count"] -= 1

        # Update batch status
        total = batch_job["total_videos"]
        completed = batch_job["completed_count"]
        failed = batch_job["failed_count"]

        if completed + failed == total:
            if failed == 0:
                batch_job["status"] = "completed"
            elif completed == 0:
                batch_job["status"] = "failed"
            else:
                batch_job["status"] = "partial"

        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from the store."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


class FakeSpawner:
    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def spawn(self, *args: Any) -> None:
        self.calls.append(args)


class FakeModalFunction:
    """Fake Modal function that tracks spawn/remote calls."""
    def __init__(self) -> None:
        self.spawn_calls: List[Tuple[Any, ...]] = []
        self.remote_calls: List[Tuple[Any, ...]] = []
        self.remote_return_value: Any = []

    def spawn(self, *args: Any) -> None:
        self.spawn_calls.append(args)

    def remote(self, *args: Any) -> Any:
        self.remote_calls.append(args)
        return self.remote_return_value


class FakeR2Connector:
    def __init__(self, videos: List[Dict[str, Any]] | None = None) -> None:
        self.videos = videos or []
        self.last_namespace: str | None = None

    def fetch_all_video_data(self, namespace: str) -> List[Dict[str, Any]]:
        self.last_namespace = namespace
        return self.videos

    def list_videos_page(
        self,
        namespace: str,
        page_size: int,
        continuation_token: str | None,
    ) -> Tuple[List[Dict[str, Any]], str | None, int, int]:
        self.last_namespace = namespace
        total_videos = len(self.videos)
        total_pages = 1 if total_videos else 0
        return self.videos[:page_size], None, total_videos, total_pages


class FakeAuthConnector:
    """Fake auth connector that always succeeds."""

    async def __call__(self, request: Request) -> str:
        return "test-user-id"


class ServerStub:
    """
    Minimal server stub providing the attributes ServerFastAPIRouter uses.
    """

    def __init__(self) -> None:
        self.job_store = FakeJobStore()
        self.delete_video_background = FakeSpawner()
        self.auth_connector = FakeAuthConnector()
        self.r2_connector = FakeR2Connector(
            videos=[
                {
                    "file_name": "sample.mp4",
                    "presigned_url": "https://example.com/video.mp4",
                    "hashed_identifier": "abc123",
                }
            ]
        )


@pytest.fixture()
def mock_modal_lookup():
    """Mock modal.Cls.from_name to return fake service classes."""
    fake_process_fn = FakeModalFunction()

    class FakeServiceClass:
        """Fake service class that returns fake modal functions."""
        def __init__(self, func):
            self.func = func

        def __call__(self):
            """Return self to allow chaining like ServiceClass().method"""
            return self

        @property
        def process_video_background(self):
            return self.func

    def lookup_side_effect(app_name: str, class_name: str, **kwargs):
        if "processing" in app_name or class_name == "ProcessingService":
            return FakeServiceClass(fake_process_fn)
        raise ValueError(f"Unknown app: {app_name}, class: {class_name}")

    # Mock modal.Cls.from_name
    with patch.object(modal.Cls, "from_name", side_effect=lookup_side_effect):
        yield {
            "process_fn": fake_process_fn,
        }


@pytest.fixture()
def test_client_internal(mock_modal_lookup) -> Tuple[TestClient, ServerStub, dict]:
    """
    FastAPI TestClient with is_file_change_enabled=True, so delete is allowed.
    """
    server = ServerStub()
    app = FastAPI()
    router = ServerFastAPIRouter(server, is_file_change_enabled=True, environment="dev")
    app.include_router(router.router)
    return TestClient(app), server, mock_modal_lookup


@pytest.fixture()
def test_client_external(mock_modal_lookup) -> Tuple[TestClient, ServerStub, dict]:
    """
    FastAPI TestClient with is_file_change_enabled=False, so delete is forbidden.
    """
    server = ServerStub()
    app = FastAPI()
    router = ServerFastAPIRouter(server, is_file_change_enabled=False, environment="prod")
    app.include_router(router.router)
    return TestClient(app), server, mock_modal_lookup


def test_health_ok(test_client_internal: Tuple[TestClient, ServerStub, dict]) -> None:
    client, _, _ = test_client_internal
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_videos_returns_data(test_client_internal: Tuple[TestClient, ServerStub, dict]) -> None:
    client, server, _ = test_client_internal
    resp = client.get("/videos", params={"namespace": "ns1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["namespace"] == "ns1"
    assert isinstance(data["videos"], list)
    assert data["videos"][0]["file_name"] == "sample.mp4"
    assert data["total_videos"] == 1
    assert data["total_pages"] == 1
    assert data["next_page_token"] is None
    assert server.r2_connector.last_namespace == "ns1"


def test_status_processing_when_unknown_job(test_client_internal: Tuple[TestClient, ServerStub, dict]) -> None:
    client, _, _ = test_client_internal
    resp = client.get("/status", params={"job_id": "does-not-exist"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"


def test_upload_creates_job_and_spawns_processing_app(
    test_client_internal: Tuple[TestClient, ServerStub, dict]
) -> None:
    client, server, mock_fns = test_client_internal
    files = [("files", ("clip.mp4", io.BytesIO(b"fake-bytes"), "video/mp4"))]
    resp = client.post("/upload", files=files, data={"namespace": "ns1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    job_id = data["job_id"]
    # Job created
    assert server.job_store.get_job(job_id) is not None
    # Processing app spawn triggered
    assert len(mock_fns["process_fn"].spawn_calls) == 1
    call_args = mock_fns["process_fn"].spawn_calls[0]
    # args: (contents, filename, job_id, namespace, parent_batch_id)
    assert call_args[1] == "clip.mp4"
    assert call_args[2] == job_id
    assert call_args[3] == "ns1"
    assert call_args[4] is None  # No parent batch


def test_batch_upload_creates_batch_job_and_spawns_children(
    test_client_internal: Tuple[TestClient, ServerStub, dict]
) -> None:
    client, server, mock_fns = test_client_internal
    # Upload 3 videos
    files = [
        ("files", ("video1.mp4", io.BytesIO(b"fake-bytes-1"), "video/mp4")),
        ("files", ("video2.mp4", io.BytesIO(b"fake-bytes-2"), "video/mp4")),
        ("files", ("video3.mp4", io.BytesIO(b"fake-bytes-3"), "video/mp4")),
    ]
    resp = client.post("/upload", files=files, data={"namespace": "batch-ns"})
    assert resp.status_code == 200
    data = resp.json()

    # Check batch response
    assert data["status"] == "processing"
    assert "batch_job_id" in data
    assert data["total_videos"] == 3
    assert data["successfully_spawned"] == 3
    assert data["failed_validation"] == 0

    batch_job_id = data["batch_job_id"]
    assert batch_job_id.startswith("batch-")

    # Batch job created
    batch_job = server.job_store.get_job(batch_job_id)
    assert batch_job is not None
    assert batch_job["job_type"] == "batch"
    assert len(batch_job["child_job_ids"]) == 3

    # All child jobs spawned
    assert len(mock_fns["process_fn"].spawn_calls) == 3

    # Check each child job was created and linked to batch
    for i, call_args in enumerate(mock_fns["process_fn"].spawn_calls):
        filename = call_args[1]
        job_id = call_args[2]
        namespace = call_args[3]
        parent_batch_id = call_args[4]

        assert filename in ["video1.mp4", "video2.mp4", "video3.mp4"]
        assert namespace == "batch-ns"
        assert parent_batch_id == batch_job_id

        # Child job exists and is linked to batch
        child_job = server.job_store.get_job(job_id)
        assert child_job is not None
        assert child_job["parent_batch_id"] == batch_job_id


def test_batch_upload_with_validation_failures(
    test_client_internal: Tuple[TestClient, ServerStub, dict]
) -> None:
    client, server, mock_fns = test_client_internal
    # Upload mix of valid and invalid files
    files = [
        ("files", ("video1.mp4", io.BytesIO(b"fake-bytes-1"), "video/mp4")),
        ("files", ("bad.txt", io.BytesIO(b"not-a-video"), "text/plain")),  # Invalid extension
        ("files", ("video2.mp4", io.BytesIO(b"fake-bytes-2"), "video/mp4")),
    ]
    resp = client.post("/upload", files=files, data={"namespace": "ns1"})
    assert resp.status_code == 200
    data = resp.json()

    # Check that only valid files were processed
    assert data["total_submitted"] == 3
    assert data["failed_validation"] == 1
    assert data["total_videos"] == 2
    assert data["successfully_spawned"] == 2

    # Only 2 spawns for valid files
    assert len(mock_fns["process_fn"].spawn_calls) == 2


def test_batch_upload_rejects_empty_list(
    test_client_internal: Tuple[TestClient, ServerStub, dict]
) -> None:
    client, _, _ = test_client_internal
    resp = client.post("/upload", files=[], data={"namespace": "ns1"})
    assert resp.status_code == 400
    assert "No files provided" in resp.json()["detail"]


def test_status_completed_after_job_store_update(
    test_client_internal: Tuple[TestClient, ServerStub, dict]
) -> None:
    client, server, _ = test_client_internal
    # Create job and mark complete
    server.job_store.create_job("j1", {"job_id": "j1", "status": "processing"})
    server.job_store.set_job_completed("j1", {"job_id": "j1", "status": "completed"})
    resp = client.get("/status", params={"job_id": "j1"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_delete_video_forbidden_when_external(test_client_external: Tuple[TestClient, ServerStub, dict]) -> None:
    client, _, _ = test_client_external
    resp = client.delete("/videos/abc123", params={"filename": "clip.mp4", "namespace": "ns1"})
    assert resp.status_code == 403


def test_delete_video_triggers_background_when_internal(
    test_client_internal: Tuple[TestClient, ServerStub, dict]
) -> None:
    client, server, _ = test_client_internal
    resp = client.delete("/videos/abc123", params={"filename": "clip.mp4", "namespace": "ns1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    assert len(server.delete_video_background.calls) == 1
    call_args = server.delete_video_background.calls[0]
    # args: (job_id, hashed_identifier, namespace)
    assert call_args[1] == "abc123"
    assert call_args[2] == "ns1"
