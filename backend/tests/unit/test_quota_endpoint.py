"""
Tests for the GET /quota endpoint.

Verifies it returns user namespace, vector count, quota, and remaining vectors.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request

from api.server_fastapi_router import ServerFastAPIRouter


def _make_mock_request(auth_header="Bearer test_token"):
    """Create a mock FastAPI Request."""
    request = MagicMock(spec=Request)
    request.headers = {"Authorization": auth_header}
    return request


def _create_router_with_mocks(user_data=None):
    """Create a ServerFastAPIRouter with mocked server instance."""
    if user_data is None:
        user_data = {
            "user_id": "auth0|user1",
            "namespace": "user_abc123",
            "vector_count": 4821,
            "vector_quota": 10_000,
        }

    server_instance = MagicMock()
    server_instance.auth_connector = AsyncMock(return_value="auth0|user1")
    server_instance.user_store = MagicMock()
    server_instance.user_store.get_or_create_user.return_value = user_data
    server_instance.user_store.check_quota.return_value = (
        user_data.get("vector_count", 0) < user_data.get("vector_quota", 10_000),
        user_data.get("vector_count", 0),
        user_data.get("vector_quota", 10_000),
    )
    server_instance.job_store = MagicMock()
    server_instance.r2_connector = MagicMock()

    router = ServerFastAPIRouter(
        server_instance=server_instance,
        is_file_change_enabled=True,
        environment="test",
    )

    # Replace the spawn function with a no-op mock to avoid real Modal calls
    router.upload_handler.process_video_spawn = MagicMock()

    return router, server_instance


class TestQuotaEndpoint:
    """Tests for GET /quota endpoint."""

    @pytest.mark.asyncio
    async def test_returns_user_info(self):
        """Returns namespace, vector_count, vector_quota, vectors_remaining."""
        router, _ = _create_router_with_mocks()
        request = _make_mock_request()

        result = await router.quota(request)

        assert result["user_id"] == "auth0|user1"
        assert result["namespace"] == "user_abc123"
        assert result["vector_count"] == 4821
        assert result["vector_quota"] == 10_000
        assert result["vectors_remaining"] == 5179

    @pytest.mark.asyncio
    async def test_new_user_returns_defaults(self):
        """Fresh user sees zero usage with full quota remaining."""
        router, _ = _create_router_with_mocks(user_data={
            "user_id": "auth0|new_user",
            "namespace": "user_newns",
            "vector_count": 0,
            "vector_quota": 10_000,
        })
        request = _make_mock_request()

        result = await router.quota(request)

        assert result["vector_count"] == 0
        assert result["vector_quota"] == 10_000
        assert result["vectors_remaining"] == 10_000

    @pytest.mark.asyncio
    async def test_full_quota_shows_zero_remaining(self):
        """User at quota shows 0 remaining."""
        router, _ = _create_router_with_mocks(user_data={
            "user_id": "auth0|full_user",
            "namespace": "user_fullns",
            "vector_count": 10_000,
            "vector_quota": 10_000,
        })
        request = _make_mock_request()

        result = await router.quota(request)

        assert result["vectors_remaining"] == 0

    @pytest.mark.asyncio
    async def test_over_quota_shows_zero_remaining(self):
        """User over quota shows 0 remaining (not negative)."""
        router, _ = _create_router_with_mocks(user_data={
            "user_id": "auth0|over_user",
            "namespace": "user_overns",
            "vector_count": 11_000,
            "vector_quota": 10_000,
        })
        request = _make_mock_request()

        result = await router.quota(request)

        assert result["vectors_remaining"] == 0

    @pytest.mark.asyncio
    async def test_calls_auth(self):
        """Auth connector is called to extract user_id."""
        router, server_instance = _create_router_with_mocks()
        request = _make_mock_request()

        await router.quota(request)

        server_instance.auth_connector.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_premium_user_higher_quota(self):
        """Premium user with custom quota is handled correctly."""
        router, _ = _create_router_with_mocks(user_data={
            "user_id": "auth0|premium",
            "namespace": "user_premns",
            "vector_count": 25_000,
            "vector_quota": 50_000,
        })
        request = _make_mock_request()

        result = await router.quota(request)

        assert result["vector_quota"] == 50_000
        assert result["vectors_remaining"] == 25_000


class TestUploadQuotaCheck:
    """Tests for upload endpoint quota checking."""

    @pytest.mark.asyncio
    async def test_upload_rejects_when_over_quota(self):
        """Returns 429 when user exceeds quota."""
        router, server_instance = _create_router_with_mocks(user_data={
            "user_id": "auth0|over",
            "namespace": "user_over",
            "vector_count": 10_000,
            "vector_quota": 10_000,
        })
        server_instance.user_store.check_quota.return_value = (False, 10_000, 10_000)
        request = _make_mock_request()

        from fastapi import UploadFile, HTTPException
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.mp4"
        mock_file.content_type = "video/mp4"
        mock_file.read = AsyncMock(return_value=b"x" * 1000)

        with pytest.raises(HTTPException) as exc_info:
            await router.upload(request, files=[mock_file], hashed_identifier="testhash123")

        assert exc_info.value.status_code == 429
        assert "storage limit" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_upload_response_includes_namespace(self):
        """Upload response includes namespace for plugin storage."""
        router, server_instance = _create_router_with_mocks()
        server_instance.user_store.check_quota.return_value = (True, 4821, 10_000)
        request = _make_mock_request()

        from fastapi import UploadFile
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.mp4"
        mock_file.content_type = "video/mp4"
        mock_file.read = AsyncMock(return_value=b"x" * 1000)

        result = await router.upload(request, files=[mock_file], hashed_identifier="testhash123")

        assert result["namespace"] == "user_abc123"

    @pytest.mark.asyncio
    async def test_upload_response_includes_quota_info(self):
        """Upload response includes vector_count and vector_quota."""
        router, server_instance = _create_router_with_mocks()
        server_instance.user_store.check_quota.return_value = (True, 4821, 10_000)
        request = _make_mock_request()

        from fastapi import UploadFile
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.mp4"
        mock_file.content_type = "video/mp4"
        mock_file.read = AsyncMock(return_value=b"x" * 1000)

        result = await router.upload(request, files=[mock_file], hashed_identifier="testhash123")

        assert "vector_count" in result
        assert "vector_quota" in result
        assert result["vector_count"] == 4821
        assert result["vector_quota"] == 10_000

    @pytest.mark.asyncio
    async def test_upload_uses_user_namespace_not_client(self):
        """Server overrides client-provided namespace with user's assigned one."""
        router, server_instance = _create_router_with_mocks()
        server_instance.user_store.check_quota.return_value = (True, 100, 10_000)
        request = _make_mock_request()

        from fastapi import UploadFile
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.mp4"
        mock_file.content_type = "video/mp4"
        mock_file.read = AsyncMock(return_value=b"x" * 1000)

        # Client tries to send a different namespace — should be ignored
        result = await router.upload(request, files=[mock_file], namespace="client_ns", hashed_identifier="testhash123")

        # The result namespace should be the user's assigned namespace
        assert result["namespace"] == "user_abc123"
