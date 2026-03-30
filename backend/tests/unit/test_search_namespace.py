"""
Tests for search namespace resolution.

Verifies authenticated search uses user's assigned namespace,
and demo search continues to use the hardcoded web-demo namespace.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request

from api.search_fastapi_router import SearchFastAPIRouter


def _make_mock_request(auth_header="Bearer test_token"):
    """Create a mock FastAPI Request."""
    request = MagicMock(spec=Request)
    request.headers = {"Authorization": auth_header}
    return request


@pytest.fixture
def mock_search_service():
    """Mock SearchService instance."""
    service = MagicMock()
    service._search_plugin.return_value = [
        {"id": "chunk1", "score": 0.9, "metadata": {"file_filename": "test.mp4"}}
    ]
    service._search_demo.return_value = [
        {"id": "chunk1", "score": 0.9, "metadata": {"file_filename": "test.mp4"}}
    ]
    service.user_store = MagicMock()
    service.user_store.get_or_create_user.return_value = {
        "user_id": "auth0|user1",
        "namespace": "user_abc123",
        "vector_count": 500,
        "vector_quota": 10_000,
    }
    return service


@pytest.fixture
def mock_auth_connector():
    """Mock AuthConnector that returns a user_id."""
    connector = AsyncMock(return_value="auth0|user1")
    return connector


@pytest.fixture
def router(mock_search_service, mock_auth_connector):
    """SearchFastAPIRouter with mocked dependencies."""
    return SearchFastAPIRouter(
        search_service_instance=mock_search_service,
        auth_connector=mock_auth_connector,
    )


class TestSearchUsesUserNamespace:
    """Test that authenticated search resolves and uses user namespace."""

    @pytest.mark.asyncio
    async def test_search_uses_user_namespace(self, router, mock_search_service, mock_auth_connector):
        """Authenticated search resolves namespace from user doc and filters by user_id."""
        request = _make_mock_request()

        await router.search(request, query="cat on a table", top_k=5)

        # Verify auth was called
        mock_auth_connector.assert_called_once_with(request)

        # Verify search used user's namespace with user_id metadata filter
        mock_search_service._search_plugin.assert_called_once()
        call_args = mock_search_service._search_plugin.call_args[0]
        call_kwargs = mock_search_service._search_plugin.call_args[1]
        assert call_args[0] == "cat on a table"  # query
        assert call_args[1] == "user_abc123"  # namespace from user doc
        assert call_args[2] == 5  # top_k
        assert call_kwargs["metadata_filter"] == {"user_id": {"$eq": "auth0|user1"}}

    @pytest.mark.asyncio
    async def test_search_response_structure(self, router, mock_search_service):
        """Search response includes query, results, and timing."""
        request = _make_mock_request()

        result = await router.search(request, query="dog", top_k=3)

        assert "query" in result
        assert "results" in result
        assert "timing" in result
        assert result["query"] == "dog"

    @pytest.mark.asyncio
    async def test_search_calls_get_or_create_user(self, router, mock_search_service):
        """Search calls get_or_create_user to resolve namespace."""
        request = _make_mock_request()

        await router.search(request, query="test")

        mock_search_service.user_store.get_or_create_user.assert_called_once_with("auth0|user1")


class TestDemoSearchUnchanged:
    """Test that demo search still uses web-demo namespace."""

    @pytest.mark.asyncio
    async def test_demo_search_uses_web_demo(self, router, mock_search_service):
        """Demo search uses hardcoded web-demo namespace with no metadata filter."""
        request = _make_mock_request()

        await router.demo_search(request, query="sunset", top_k=5)

        call_args = mock_search_service._search_demo.call_args[0]
        assert call_args[1] == "web-demo"  # namespace
        # Demo search should NOT pass metadata_filter
        call_kwargs = mock_search_service._search_demo.call_args[1] or {}
        assert "metadata_filter" not in call_kwargs

    @pytest.mark.asyncio
    async def test_demo_search_no_auth(self, router, mock_auth_connector):
        """Demo search doesn't call auth connector."""
        request = _make_mock_request()

        await router.demo_search(request, query="test", top_k=5)

        mock_auth_connector.assert_not_called()
