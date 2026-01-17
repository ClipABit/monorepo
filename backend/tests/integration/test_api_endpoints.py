import io
from typing import Any, Dict, List, Tuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.fastapi_router import FastAPIRouter


class FakeJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self._jobs[job_id] = data

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        return self._jobs.get(job_id)

    def set_job_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        self._jobs[job_id] = result

    def set_job_failed(self, job_id: str, error: str) -> None:
        self._jobs[job_id] = {"job_id": job_id, "status": "failed", "error": error}


class FakeSpawner:
    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def spawn(self, *args: Any) -> None:
        self.calls.append(args)


class FakeSearcher:
    def __init__(self, results: List[Dict[str, Any]] | None = None) -> None:
        self.results = results or []
        self.last_call: Tuple[str, int, str] | None = None

    def search(self, query: str, top_k: int, namespace: str) -> List[Dict[str, Any]]:
        self.last_call = (query, top_k, namespace)
        return self.results


class FakeR2Connector:
    def __init__(self, videos: List[Dict[str, Any]] | None = None) -> None:
        self.videos = videos or []
        self.last_namespace: str | None = None

    def fetch_all_video_data(self, namespace: str) -> List[Dict[str, Any]]:
        self.last_namespace = namespace
        return self.videos


class ServerStub:
    """
    Minimal server stub providing the attributes FastAPIRouter uses.
    """

    def __init__(self) -> None:
        self.job_store = FakeJobStore()
        self.process_video_background = FakeSpawner()
        self.delete_video_background = FakeSpawner()
        self.searcher = FakeSearcher(
            results=[
                {
                    "score": 0.99,
                    "metadata": {
                        "presigned_url": "https://example.com/video.mp4",
                        "start_time_s": 0,
                        "file_filename": "video.mp4",
                        "hashed_identifier": "abc123",
                    },
                }
            ]
        )
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
def test_client_internal() -> Tuple[TestClient, ServerStub]:
    """
    FastAPI TestClient with is_internal_env=True, so delete is allowed.
    """
    server = ServerStub()
    app = FastAPI()
    router = FastAPIRouter(server, is_internal_env=True)
    app.include_router(router.router)
    return TestClient(app), server


@pytest.fixture()
def test_client_external() -> Tuple[TestClient, ServerStub]:
    """
    FastAPI TestClient with is_internal_env=False, so delete is forbidden.
    """
    server = ServerStub()
    app = FastAPI()
    router = FastAPIRouter(server, is_internal_env=False)
    app.include_router(router.router)
    return TestClient(app), server


def test_health_ok(test_client_internal: Tuple[TestClient, ServerStub]) -> None:
    client, _ = test_client_internal
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_videos_returns_data(test_client_internal: Tuple[TestClient, ServerStub]) -> None:
    client, server = test_client_internal
    resp = client.get("/videos", params={"namespace": "ns1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["namespace"] == "ns1"
    assert isinstance(data["videos"], list)
    assert data["videos"][0]["file_name"] == "sample.mp4"
    assert server.r2_connector.last_namespace == "ns1"


def test_search_invokes_searcher_and_returns_results(
    test_client_internal: Tuple[TestClient, ServerStub]
) -> None:
    client, server = test_client_internal
    resp = client.get("/search", params={"query": "hello world", "namespace": "web-demo", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data and isinstance(data["results"], list)
    assert server.searcher.last_call == ("hello world", 5, "web-demo")


def test_status_processing_when_unknown_job(test_client_internal: Tuple[TestClient, ServerStub]) -> None:
    client, _ = test_client_internal
    resp = client.get("/status", params={"job_id": "does-not-exist"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"


def test_upload_creates_job_and_spawns_background(
    test_client_internal: Tuple[TestClient, ServerStub]
) -> None:
    client, server = test_client_internal
    files = {"file": ("clip.mp4", io.BytesIO(b"fake-bytes"), "video/mp4")}
    resp = client.post("/upload", files=files, data={"namespace": "ns1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    job_id = data["job_id"]
    # Job created
    assert server.job_store.get_job(job_id) is not None
    # Background process triggered
    assert len(server.process_video_background.calls) == 1
    call_args = server.process_video_background.calls[0]
    # args: (contents, filename, job_id, namespace)
    assert call_args[1] == "clip.mp4"
    assert call_args[2] == job_id
    assert call_args[3] == "ns1"


def test_status_completed_after_job_store_update(
    test_client_internal: Tuple[TestClient, ServerStub]
) -> None:
    client, server = test_client_internal
    # Create job and mark complete
    server.job_store.create_job("j1", {"job_id": "j1", "status": "processing"})
    server.job_store.set_job_completed("j1", {"job_id": "j1", "status": "completed"})
    resp = client.get("/status", params={"job_id": "j1"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_delete_video_forbidden_when_external(test_client_external: Tuple[TestClient, ServerStub]) -> None:
    client, _ = test_client_external
    resp = client.delete("/videos/abc123", params={"filename": "clip.mp4", "namespace": "ns1"})
    assert resp.status_code == 403


def test_delete_video_triggers_background_when_internal(
    test_client_internal: Tuple[TestClient, ServerStub]
) -> None:
    client, server = test_client_internal
    resp = client.delete("/videos/abc123", params={"filename": "clip.mp4", "namespace": "ns1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    assert len(server.delete_video_background.calls) == 1
    call_args = server.delete_video_background.calls[0]
    # args: (job_id, hashed_identifier, namespace)
    assert call_args[1] == "abc123"
    assert call_args[2] == "ns1"

