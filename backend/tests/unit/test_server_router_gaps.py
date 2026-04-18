"""
Tests for ServerFastAPIRouter edge cases — R2 failures, quota endpoint defaults,
upload validation, list_videos error handling, and cache clear errors.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request, HTTPException, UploadFile

from api.server_fastapi_router import ServerFastAPIRouter
from database.firebase.user_store_connector import UserStoreConnector


def _make_mock_request():
    request = MagicMock(spec=Request)
    request.headers = {"Authorization": "Bearer test_token"}
    return request


def _create_router(
    user_data=None, is_file_change_enabled=True, default_vector_quota=None
):
    """Create router with mocked internals."""
    if default_vector_quota is None:
        default_vector_quota = UserStoreConnector.DEFAULT_VECTOR_QUOTA

    if user_data is None:
        user_data = {
            "user_id": "auth0|user1",
            "namespace": "ns_03",
            "vector_count": 500,
            "vector_quota": default_vector_quota,
        }

    server = MagicMock()
    server.auth_connector = AsyncMock(return_value="auth0|user1")
    server.user_store = MagicMock()
    server.user_store.get_or_create_user.return_value = user_data
    server.user_store.check_quota.return_value = (
        user_data.get("vector_count", 0)
        < user_data.get("vector_quota", default_vector_quota),
        user_data.get("vector_count", 0),
        user_data.get("vector_quota", default_vector_quota),
    )
    server.job_store = MagicMock()
    server.r2_connector = MagicMock()

    router = ServerFastAPIRouter(
        server_instance=server,
        is_file_change_enabled=is_file_change_enabled,
        environment="test",
    )
    router.upload_handler.process_video_spawn = MagicMock()

    return router, server


# =============================================================================
# Quota Endpoint — Missing Field Defaults
# =============================================================================


class TestQuotaEndpointDefaults:
    """Tests for quota endpoint when user data is missing fields."""

    @pytest.mark.asyncio
    async def test_missing_vector_count_defaults_to_zero(self, default_vector_quota):
        """User data without vector_count → defaults to 0."""
        router, _ = _create_router(
            user_data={
                "user_id": "auth0|user1",
                "namespace": "ns_03",
                "vector_quota": default_vector_quota,
            },
            default_vector_quota=default_vector_quota,
        )

        result = await router.quota(_make_mock_request())

        assert result["vector_count"] == 0
        assert result["vectors_remaining"] == default_vector_quota

    @pytest.mark.asyncio
    async def test_missing_vector_quota_defaults(self, default_vector_quota):
        """User data without vector_quota → falls back to DEFAULT_VECTOR_QUOTA."""
        router, _ = _create_router(
            user_data={
                "user_id": "auth0|user1",
                "namespace": "ns_03",
                "vector_count": 500,
            },
            default_vector_quota=default_vector_quota,
        )

        result = await router.quota(_make_mock_request())

        assert result["vector_quota"] == default_vector_quota

    @pytest.mark.asyncio
    async def test_missing_namespace_defaults_to_empty(self, default_vector_quota):
        """User data without namespace → defaults to empty string."""
        router, _ = _create_router(
            user_data={
                "user_id": "auth0|user1",
                "vector_count": 500,
                "vector_quota": default_vector_quota,
            },
            default_vector_quota=default_vector_quota,
        )

        result = await router.quota(_make_mock_request())

        assert result["namespace"] == ""


# =============================================================================
# List Videos — R2 Failure
# =============================================================================


class TestListVideosErrors:
    """Tests for list_videos error handling."""

    @pytest.mark.asyncio
    async def test_r2_failure_returns_500(self):
        """R2 connector exception → HTTPException 500."""
        router, server = _create_router()
        server.r2_connector.list_videos_page.side_effect = Exception("R2 unreachable")

        with pytest.raises(HTTPException) as exc_info:
            await router.list_videos(_make_mock_request())

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_negative_page_size_returns_400(self):
        """Negative page_size → 400."""
        router, _ = _create_router()

        with pytest.raises(HTTPException) as exc_info:
            await router.list_videos(_make_mock_request(), page_size=-1)

        assert exc_info.value.status_code == 400


# =============================================================================
# Clear Cache — Error Paths
# =============================================================================


class TestClearCacheErrors:
    """Tests for clear_cache error handling."""

    @pytest.mark.asyncio
    async def test_r2_failure_returns_500(self):
        """R2 connector exception → HTTPException 500."""
        router, server = _create_router()
        server.r2_connector.clear_cache.side_effect = Exception("R2 write error")

        with pytest.raises(HTTPException) as exc_info:
            await router.clear_cache(_make_mock_request())

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_disabled_returns_403(self):
        """is_file_change_enabled=False → 403."""
        router, _ = _create_router(is_file_change_enabled=False)

        with pytest.raises(HTTPException) as exc_info:
            await router.clear_cache(_make_mock_request())

        assert exc_info.value.status_code == 403


# =============================================================================
# Upload — Validation Edge Cases
# =============================================================================


class TestUploadValidation:
    """Tests for upload endpoint validation edge cases."""

    @pytest.mark.asyncio
    async def test_whitespace_hashed_identifier_rejected(self):
        """hashed_identifier='   ' (whitespace only) → 400."""
        router, _ = _create_router()
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.mp4"
        mock_file.content_type = "video/mp4"
        mock_file.read = AsyncMock(return_value=b"x" * 100)

        with pytest.raises(HTTPException) as exc_info:
            await router.upload(
                _make_mock_request(), files=[mock_file], hashed_identifier="   "
            )

        assert exc_info.value.status_code == 400
        assert "hashed_identifier" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_empty_hashed_identifier_rejected(self):
        """hashed_identifier='' → 400."""
        router, _ = _create_router()
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.mp4"

        with pytest.raises(HTTPException) as exc_info:
            await router.upload(
                _make_mock_request(), files=[mock_file], hashed_identifier=""
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_no_namespace_resolved_returns_500(self, default_vector_quota):
        """User with empty namespace → 500."""
        router, _ = _create_router(
            user_data={
                "user_id": "auth0|user1",
                "namespace": "",
                "vector_count": 0,
                "vector_quota": default_vector_quota,
            },
            default_vector_quota=default_vector_quota,
        )
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.mp4"

        with pytest.raises(HTTPException) as exc_info:
            await router.upload(
                _make_mock_request(), files=[mock_file], hashed_identifier="valid_hash"
            )

        assert exc_info.value.status_code == 500
        assert "namespace" in exc_info.value.detail.lower()


# =============================================================================
# Status Endpoint — Edge Cases
# =============================================================================


class TestStatusEndpointEdgeCases:
    """Tests for status endpoint edge cases."""

    @pytest.mark.asyncio
    async def test_job_not_found_returns_processing(self):
        """Unknown job_id → returns processing status (not 404)."""
        router, server = _create_router()
        server.job_store.get_job.return_value = None

        result = await router.status("unknown_job_id")

        assert result["status"] == "processing"
        assert result["job_id"] == "unknown_job_id"

    @pytest.mark.asyncio
    async def test_completed_job_returns_full_data(self):
        """Completed job → returns full job data dict."""
        router, server = _create_router()
        job_data = {
            "job_id": "j1",
            "status": "completed",
            "chunks": 5,
        }
        server.job_store.get_job.return_value = job_data

        result = await router.status("j1")

        assert result == job_data

    @pytest.mark.asyncio
    async def test_failed_job_returns_error(self):
        """Failed job → returns error details."""
        router, server = _create_router()
        job_data = {
            "job_id": "j2",
            "status": "failed",
            "error": "Processing error",
        }
        server.job_store.get_job.return_value = job_data

        result = await router.status("j2")

        assert result["status"] == "failed"
        assert result["error"] == "Processing error"
