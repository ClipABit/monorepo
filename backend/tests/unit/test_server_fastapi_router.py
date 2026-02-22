"""
Unit tests for ServerFastAPIRouter auth protection.

Tests that protected endpoints require authentication and health is public.
"""

import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.server_fastapi_router import ServerFastAPIRouter
from auth.auth_connector import AuthConnector


class FakeAuthConnector:
    """Fake auth connector that always succeeds."""

    async def __call__(self, request: Request) -> str:
        return "test-user-id"


def _make_server_instance(auth_connector=None):
    """Create a mock server instance with required attributes."""
    instance = MagicMock()
    instance.auth_connector = auth_connector or FakeAuthConnector()
    instance.job_store = MagicMock()
    instance.r2_connector = MagicMock()
    return instance


def _make_client(auth_connector=None):
    """Create a TestClient wired to ServerFastAPIRouter."""
    server = _make_server_instance(auth_connector)
    app = FastAPI()
    router = ServerFastAPIRouter(
        server_instance=server,
        is_file_change_enabled=True,
        environment="test",
    )
    app.include_router(router.router)
    return TestClient(app, raise_server_exceptions=False), server


AUTH_HEADERS = {"Authorization": "Bearer test-token"}


class TestHealthEndpoint:
    """Test /health is public."""

    def test_health_returns_ok_without_auth(self):
        """Verify health check works with no auth header."""
        client, _ = _make_client()
        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestProtectedEndpointsWithFakeAuth:
    """Test protected endpoints succeed with fake auth."""

    def test_status_returns_ok(self):
        """Verify /status works with auth."""
        client, server = _make_client()
        server.job_store.get_job.return_value = None

        resp = client.get("/status", params={"job_id": "j1"}, headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_list_videos_returns_ok(self):
        """Verify /videos works with auth."""
        client, server = _make_client()
        server.r2_connector.list_videos_page.return_value = ([], None, 0, 0)

        resp = client.get("/videos", headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_upload_returns_ok(self):
        """Verify /upload works with auth (empty files still validates)."""
        client, _ = _make_client()
        resp = client.post("/upload", data={"namespace": ""}, headers=AUTH_HEADERS)
        # Upload handler handles validation, just check auth didn't block
        assert resp.status_code != 401

    def test_delete_video_returns_ok(self):
        """Verify /videos/{id} DELETE works with auth."""
        client, server = _make_client()
        server.delete_video_background = MagicMock()
        server.delete_video_background.spawn = MagicMock()

        resp = client.delete(
            "/videos/abc123",
            params={"filename": "test.mp4", "namespace": ""},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200

    def test_clear_cache_returns_ok(self):
        """Verify /cache/clear works with auth."""
        client, server = _make_client()
        server.r2_connector.clear_cache.return_value = 3

        resp = client.post("/cache/clear", headers=AUTH_HEADERS)
        assert resp.status_code == 200


class TestProtectedEndpointsRejectUnauthenticated:
    """Test protected endpoints return 401 without auth."""

    @pytest.fixture()
    def real_auth_client(self):
        """Client with real AuthConnector (will reject since no valid token)."""
        auth = AuthConnector(domain="test.auth0.com", audience="https://test-api")
        return _make_client(auth_connector=auth)

    def test_status_rejects_no_auth(self, real_auth_client):
        client, _ = real_auth_client
        resp = client.get("/status", params={"job_id": "j1"})
        assert resp.status_code == 401

    def test_upload_rejects_no_auth(self, real_auth_client):
        client, _ = real_auth_client
        resp = client.post("/upload", data={"namespace": ""})
        assert resp.status_code == 401

    def test_list_videos_rejects_no_auth(self, real_auth_client):
        client, _ = real_auth_client
        resp = client.get("/videos")
        assert resp.status_code == 401

    def test_delete_video_rejects_no_auth(self, real_auth_client):
        client, _ = real_auth_client
        resp = client.delete("/videos/abc123", params={"filename": "test.mp4"})
        assert resp.status_code == 401

    def test_clear_cache_rejects_no_auth(self, real_auth_client):
        client, _ = real_auth_client
        resp = client.post("/cache/clear")
        assert resp.status_code == 401

    def test_status_rejects_invalid_scheme(self, real_auth_client):
        client, _ = real_auth_client
        resp = client.get("/status", params={"job_id": "j1"}, headers={"Authorization": "Basic abc"})
        assert resp.status_code == 401
