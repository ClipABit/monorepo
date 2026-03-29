"""
Tests for the shared namespace pool system.

Covers:
1. Quota exceeded rejects upload (HTTP 429)
2. Two users in different namespaces — vectors isolated
3. Two users in same namespace — vectors coexist, search filtered by user_id
4. Even-spread assignment
5. Permanent namespace binding
"""

import io
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.server_fastapi_router import ServerFastAPIRouter


# ── Helpers ──────────────────────────────────────────────────────────

AUTH_HEADERS = {"Authorization": "Bearer test-token"}


def _mock_doc(exists: bool, data: dict = None):
    doc = MagicMock()
    doc.exists = exists
    doc.to_dict.return_value = data
    return doc


class FakeJobStore:
    def __init__(self):
        self._jobs = {}

    def create_job(self, job_id, data):
        self._jobs[job_id] = data

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def create_batch_job(self, batch_job_id, child_job_ids, namespace):
        self._jobs[batch_job_id] = {
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
        return True

    def update_batch_on_child_completion(self, batch_job_id, child_job_id, child_result):
        return True

    def set_job_failed(self, job_id, error):
        self._jobs[job_id] = {"job_id": job_id, "status": "failed", "error": error}

    def delete_job(self, job_id):
        self._jobs.pop(job_id, None)


class FakeAuthConnector:
    def __init__(self, user_id="test-user-a"):
        self._user_id = user_id

    async def __call__(self, request):
        return self._user_id


class FakeModalFunction:
    def __init__(self):
        self.spawn_calls = []

    def spawn(self, *args):
        self.spawn_calls.append(args)


class FakeR2Connector:
    def list_videos_page(self, **kwargs):
        return [], None, 0, 0


def _make_app(server, fake_fn):
    """Create FastAPI app using dev combined mode (processing_service_cls) to avoid Modal lookups."""
    fake_processing_cls = MagicMock()
    fake_processing_cls.return_value.process_video_background = fake_fn

    app = FastAPI()
    router = ServerFastAPIRouter(
        server, is_file_change_enabled=True, environment="dev",
        processing_service_cls=fake_processing_cls,
    )
    app.include_router(router.router)
    return app


# ── Test: Quota Exceeded ─────────────────────────────────────────────


class TestQuotaExceeded:
    """Upload should return 429 when user is at or over their vector quota."""

    def _make_client(self, vector_count, vector_quota=10_000):
        class UserStore:
            def get_or_create_user(self, user_id):
                return {
                    "user_id": user_id,
                    "namespace": "ns_00",
                    "vector_count": vector_count,
                    "vector_quota": vector_quota,
                }

            def check_quota(self, user_id):
                return (vector_count < vector_quota, vector_count, vector_quota)

        server = MagicMock()
        server.job_store = FakeJobStore()
        server.auth_connector = FakeAuthConnector()
        server.user_store = UserStore()
        server.r2_connector = FakeR2Connector()

        fake_fn = FakeModalFunction()
        app = _make_app(server, fake_fn)
        return TestClient(app), fake_fn

    def test_rejects_at_limit(self):
        """User at exactly their quota gets 429."""
        client, _ = self._make_client(10_000)
        files = [("files", ("clip.mp4", io.BytesIO(b"fake"), "video/mp4"))]
        resp = client.post("/upload", files=files, data={"namespace": "x", "hashed_identifier": "testhash123"}, headers=AUTH_HEADERS)
        assert resp.status_code == 429
        assert "quota" in resp.json()["detail"].lower()

    def test_rejects_over_limit(self):
        """User over their quota gets 429."""
        client, _ = self._make_client(12_000)
        files = [("files", ("clip.mp4", io.BytesIO(b"fake"), "video/mp4"))]
        resp = client.post("/upload", files=files, data={"namespace": "x", "hashed_identifier": "testhash123"}, headers=AUTH_HEADERS)
        assert resp.status_code == 429

    def test_allows_under_limit(self):
        """User at 9,999 still gets through."""
        client, fn = self._make_client(9_999)
        files = [("files", ("clip.mp4", io.BytesIO(b"fake"), "video/mp4"))]
        resp = client.post("/upload", files=files, data={"namespace": "x", "hashed_identifier": "testhash123"}, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert len(fn.spawn_calls) == 1

    def test_batch_rejected_at_limit(self):
        """Batch upload also gets 429 when at quota."""
        client, _ = self._make_client(10_000)
        files = [
            ("files", ("v1.mp4", io.BytesIO(b"fake1"), "video/mp4")),
            ("files", ("v2.mp4", io.BytesIO(b"fake2"), "video/mp4")),
        ]
        resp = client.post("/upload", files=files, data={"namespace": "x", "hashed_identifier": "testhash123"}, headers=AUTH_HEADERS)
        assert resp.status_code == 429


# ── Test: Two Users, Different Namespaces ────────────────────────────


class TestTwoUsersDifferentNamespaces:
    """Two users assigned to different namespaces get vectors isolated."""

    def test_vectors_go_to_assigned_namespace(self):
        """User A -> ns_00, User B -> ns_01. Spawn calls target correct namespace."""

        class UserStoreMulti:
            def __init__(self):
                self._users = {
                    "user-a": {"user_id": "user-a", "namespace": "ns_00", "vector_count": 0, "vector_quota": 10_000},
                    "user-b": {"user_id": "user-b", "namespace": "ns_01", "vector_count": 0, "vector_quota": 10_000},
                }

            def get_or_create_user(self, user_id):
                return self._users[user_id]

            def check_quota(self, user_id):
                u = self._users[user_id]
                return (True, u["vector_count"], u["vector_quota"])

        current_user = {"id": "user-a"}

        class SwitchableAuth:
            async def __call__(self, request):
                return current_user["id"]

        server = MagicMock()
        server.job_store = FakeJobStore()
        server.auth_connector = SwitchableAuth()
        server.user_store = UserStoreMulti()
        server.r2_connector = FakeR2Connector()

        fake_fn = FakeModalFunction()
        app = _make_app(server, fake_fn)
        client = TestClient(app)

        # User A uploads
        current_user["id"] = "user-a"
        files_a = [("files", ("a.mp4", io.BytesIO(b"video-a"), "video/mp4"))]
        resp_a = client.post("/upload", files=files_a, data={"namespace": "", "hashed_identifier": "testhash123"}, headers=AUTH_HEADERS)
        assert resp_a.status_code == 200
        assert resp_a.json()["namespace"] == "ns_00"

        # User B uploads
        current_user["id"] = "user-b"
        files_b = [("files", ("b.mp4", io.BytesIO(b"video-b"), "video/mp4"))]
        resp_b = client.post("/upload", files=files_b, data={"namespace": "", "hashed_identifier": "testhash123"}, headers=AUTH_HEADERS)
        assert resp_b.status_code == 200
        assert resp_b.json()["namespace"] == "ns_01"

        # Verify spawn calls targeted different namespaces
        assert len(fake_fn.spawn_calls) == 2
        ns_a = fake_fn.spawn_calls[0][3]
        ns_b = fake_fn.spawn_calls[1][3]
        assert ns_a == "ns_00"
        assert ns_b == "ns_01"

        # Verify user_id passed correctly
        uid_a = fake_fn.spawn_calls[0][5]
        uid_b = fake_fn.spawn_calls[1][5]
        assert uid_a == "user-a"
        assert uid_b == "user-b"


# ── Test: Two Users, Same Namespace ──────────────────────────────────


class TestTwoUsersSameNamespace:
    """Two users in the same namespace — vectors coexist, search filtered by user_id."""

    def test_both_users_upload_to_same_namespace(self):
        """Both users' spawn calls target the same namespace with different user_ids."""
        class UserStoreShared:
            def __init__(self):
                self._users = {
                    "user-x": {"user_id": "user-x", "namespace": "ns_00", "vector_count": 0, "vector_quota": 10_000},
                    "user-y": {"user_id": "user-y", "namespace": "ns_00", "vector_count": 0, "vector_quota": 10_000},
                }

            def get_or_create_user(self, uid):
                return self._users[uid]

            def check_quota(self, uid):
                return (True, 0, 10_000)

        current_user = {"id": "user-x"}

        class SwitchableAuth:
            async def __call__(self, request):
                return current_user["id"]

        server = MagicMock()
        server.job_store = FakeJobStore()
        server.auth_connector = SwitchableAuth()
        server.user_store = UserStoreShared()
        server.r2_connector = FakeR2Connector()

        fake_fn = FakeModalFunction()
        app = _make_app(server, fake_fn)
        client = TestClient(app)

        # User X uploads
        current_user["id"] = "user-x"
        files_x = [("files", ("x.mp4", io.BytesIO(b"video-x"), "video/mp4"))]
        resp_x = client.post("/upload", files=files_x, data={"namespace": "", "hashed_identifier": "testhash123"}, headers=AUTH_HEADERS)
        assert resp_x.status_code == 200

        # User Y uploads
        current_user["id"] = "user-y"
        files_y = [("files", ("y.mp4", io.BytesIO(b"video-y"), "video/mp4"))]
        resp_y = client.post("/upload", files=files_y, data={"namespace": "", "hashed_identifier": "testhash123"}, headers=AUTH_HEADERS)
        assert resp_y.status_code == 200

        # Both hit same namespace
        assert fake_fn.spawn_calls[0][3] == "ns_00"
        assert fake_fn.spawn_calls[1][3] == "ns_00"

        # But different user_ids
        assert fake_fn.spawn_calls[0][5] == "user-x"
        assert fake_fn.spawn_calls[1][5] == "user-y"

    def test_search_filters_by_user_id(self):
        """Authenticated search adds user_id metadata filter for isolation in shared namespace."""
        from api.search_fastapi_router import SearchFastAPIRouter

        mock_search_service = MagicMock()
        mock_search_service._search_internal.return_value = []
        mock_search_service.user_store = MagicMock()
        mock_search_service.user_store.get_or_create_user.return_value = {
            "user_id": "user-x",
            "namespace": "ns_00",
            "vector_count": 500,
            "vector_quota": 10_000,
        }

        mock_auth = AsyncMock(return_value="user-x")

        router = SearchFastAPIRouter(
            search_service_instance=mock_search_service,
            auth_connector=mock_auth,
        )

        app = FastAPI()
        app.include_router(router.router)
        client = TestClient(app)

        resp = client.get("/search", params={"query": "sunset"}, headers=AUTH_HEADERS)
        assert resp.status_code == 200

        call_kwargs = mock_search_service._search_internal.call_args[1]
        assert call_kwargs["metadata_filter"] == {"user_id": {"$eq": "user-x"}}

    def test_search_different_users_get_different_filters(self):
        """Two users searching same namespace get different user_id filters."""
        from api.search_fastapi_router import SearchFastAPIRouter

        mock_search_service = MagicMock()
        mock_search_service._search_internal.return_value = []

        current_user = {"id": "user-x"}

        mock_search_service.user_store = MagicMock()
        mock_search_service.user_store.get_or_create_user.side_effect = lambda uid: {
            "user_id": uid,
            "namespace": "ns_00",
            "vector_count": 0,
            "vector_quota": 10_000,
        }

        mock_auth = AsyncMock(side_effect=lambda req: current_user["id"])

        router = SearchFastAPIRouter(
            search_service_instance=mock_search_service,
            auth_connector=mock_auth,
        )

        app = FastAPI()
        app.include_router(router.router)
        client = TestClient(app)

        # User X searches
        current_user["id"] = "user-x"
        client.get("/search", params={"query": "cat"}, headers=AUTH_HEADERS)
        filter_x = mock_search_service._search_internal.call_args[1]["metadata_filter"]
        assert filter_x == {"user_id": {"$eq": "user-x"}}

        mock_search_service._search_internal.reset_mock()

        # User Y searches
        current_user["id"] = "user-y"
        client.get("/search", params={"query": "cat"}, headers=AUTH_HEADERS)
        filter_y = mock_search_service._search_internal.call_args[1]["metadata_filter"]
        assert filter_y == {"user_id": {"$eq": "user-y"}}


# ── Test: Processing injects user_id and project_id into metadata ────


class TestProcessingMetadataInjection:
    """Verify user_id and project_id are injected into chunk metadata before Pinecone upsert."""

    def _create_service_with_mocks(self):
        from services.processing_service import ProcessingService

        service = ProcessingService.__new__(ProcessingService)
        service.preprocessor = MagicMock()
        service.video_embedder = MagicMock()
        service.pinecone_connector = MagicMock()
        service.job_store = MagicMock()
        service.user_store = MagicMock()

        mock_chunk = {
            "chunk_id": "job1_chunk_0000",
            "frames": [np.zeros((480, 640, 3), dtype=np.uint8)],
            "metadata": {
                "frame_count": 8,
                "complexity_score": 0.5,
                "timestamp_range": (0.0, 5.0),
                "file_info": {"filename": "test.mp4", "type": "video/mp4", "hashed_identifier": "hash123"},
            },
            "memory_mb": 1.0,
        }
        service.preprocessor.process_video_from_bytes.return_value = [mock_chunk]

        mock_embedding = MagicMock()
        mock_embedding.numpy.return_value = np.zeros(512)
        service.video_embedder._generate_clip_embedding.return_value = mock_embedding
        service.pinecone_connector.upsert_chunk.return_value = True
        service.user_store.check_quota.return_value = (True, 0, 10_000)
        service.user_store.reserve_quota.return_value = (True, 0, 10_000)

        return service

    def test_user_id_in_metadata(self):
        """user_id is injected into chunk metadata."""
        service = self._create_service_with_mocks()

        service.process_video_background(
            video_bytes=b"fake",
            filename="test.mp4",
            job_id="job1",
            namespace="ns_00",
            user_id="auth0|user1",
            hashed_identifier="hash123",
            project_id="proj_abc",
        )

        upsert_call = service.pinecone_connector.upsert_chunk.call_args
        metadata = upsert_call[1]["metadata"]
        assert metadata["user_id"] == "auth0|user1"

    def test_project_id_in_metadata(self):
        """project_id is injected into chunk metadata."""
        service = self._create_service_with_mocks()

        service.process_video_background(
            video_bytes=b"fake",
            filename="test.mp4",
            job_id="job1",
            namespace="ns_00",
            user_id="auth0|user1",
            hashed_identifier="hash123",
            project_id="proj_abc",
        )

        upsert_call = service.pinecone_connector.upsert_chunk.call_args
        metadata = upsert_call[1]["metadata"]
        assert metadata["project_id"] == "proj_abc"

    def test_namespace_level_increment(self):
        """reserve_quota is called with namespace for dual-level tracking."""
        service = self._create_service_with_mocks()

        service.process_video_background(
            video_bytes=b"fake",
            filename="test.mp4",
            job_id="job1",
            namespace="ns_05",
            user_id="auth0|user1",
            hashed_identifier="hash123",
        )

        service.user_store.reserve_quota.assert_called_once_with("auth0|user1", 1, "ns_05")
