"""
Unit tests for SearchFastAPIRouter.

Tests the search API endpoint with mocked SearchService and auth.
"""

from typing import Any, List, Dict

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.search_fastapi_router import SearchFastAPIRouter


class FakeUserStore:
    """Fake UserStoreConnector for testing."""

    def __init__(self, namespace="user_test_ns"):
        self.namespace = namespace

    def get_or_create_user(self, user_id):
        return {
            "user_id": user_id,
            "namespace": self.namespace,
            "vector_count": 0,
            "vector_quota": 10_000,
        }


class FakeSearchService:
    """Fake SearchService for testing the router."""

    def __init__(self, results: List[Dict[str, Any]] | None = None, namespace="user_test_ns"):
        self.results = results or []
        self.last_query: str | None = None
        self.last_namespace: str | None = None
        self.last_top_k: int | None = None
        self.should_raise: Exception | None = None
        self.user_store = FakeUserStore(namespace=namespace)

    def _search_internal(self, query: str, namespace: str = "", top_k: int = 10, metadata_filter: dict = None) -> List[Dict[str, Any]]:
        """Mock search implementation."""
        self.last_query = query
        self.last_namespace = namespace
        self.last_top_k = top_k
        self.last_metadata_filter = metadata_filter

        if self.should_raise:
            raise self.should_raise

        return self.results


class FakeAuthConnector:
    """Fake auth connector that always succeeds."""

    async def __call__(self, request: Request) -> str:
        return "test-user-id"


AUTH_HEADERS = {"Authorization": "Bearer test-token"}

SAMPLE_RESULTS = [
    {
        "id": "chunk-1",
        "score": 0.95,
        "metadata": {
            "file_hashed_identifier": "abc123",
            "file_name": "video1.mp4",
            "start_time": 0.0,
            "end_time": 5.0,
            "presigned_url": "https://example.com/video1.mp4"
        }
    },
    {
        "id": "chunk-2",
        "score": 0.87,
        "metadata": {
            "file_hashed_identifier": "def456",
            "file_name": "video2.mp4",
            "start_time": 10.0,
            "end_time": 15.0,
            "presigned_url": "https://example.com/video2.mp4"
        }
    }
]


@pytest.fixture()
def search_service() -> FakeSearchService:
    """Create a fake search service with sample results."""
    return FakeSearchService(results=list(SAMPLE_RESULTS))


@pytest.fixture()
def auth_connector() -> FakeAuthConnector:
    """Create a fake auth connector."""
    return FakeAuthConnector()


@pytest.fixture()
def test_client(search_service: FakeSearchService, auth_connector: FakeAuthConnector) -> tuple[TestClient, FakeSearchService]:
    """Create FastAPI test client with search router and auth."""
    app = FastAPI()
    router = SearchFastAPIRouter(search_service_instance=search_service, auth_connector=auth_connector)
    app.include_router(router.router)
    return TestClient(app), search_service


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_ok(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify health check returns ok status."""
        client, _ = test_client
        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "search"

    def test_health_requires_no_auth(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify health endpoint is public (no auth required)."""
        client, _ = test_client
        resp = client.get("/health")

        assert resp.status_code == 200


class TestSearchEndpoint:
    """Test /search endpoint."""

    def test_search_with_query_returns_results(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify search returns results for a query."""
        client, service = test_client
        resp = client.get("/search", params={"query": "woman on a train"}, headers=AUTH_HEADERS)

        assert resp.status_code == 200
        data = resp.json()

        assert data["query"] == "woman on a train"
        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == "chunk-1"
        assert data["results"][0]["score"] == 0.95
        assert "timing" in data
        assert "total_s" in data["timing"]

    def test_search_passes_query_to_service(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify query is passed correctly to service."""
        client, service = test_client
        client.get("/search", params={"query": "test query"}, headers=AUTH_HEADERS)

        assert service.last_query == "test query"

    def test_search_uses_user_namespace(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify user's namespace from Firestore is used, not client-provided."""
        client, service = test_client
        client.get("/search", params={"query": "test"}, headers=AUTH_HEADERS)

        # Namespace should come from user_store, not client params
        assert service.last_namespace == "user_test_ns"

    def test_search_ignores_client_namespace_param(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify client-provided namespace param is ignored."""
        client, service = test_client
        # Even if client sends a namespace param, the user's assigned namespace is used
        client.get("/search", params={"query": "test", "namespace": "client-ns"}, headers=AUTH_HEADERS)

        # Should still be user's namespace, not "client-ns"
        assert service.last_namespace == "user_test_ns"

    def test_search_with_top_k(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify top_k parameter is passed to service."""
        client, service = test_client
        client.get("/search", params={"query": "test", "top_k": 20}, headers=AUTH_HEADERS)

        assert service.last_top_k == 20

    def test_search_with_default_top_k(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify default top_k is 10."""
        client, service = test_client
        client.get("/search", params={"query": "test"}, headers=AUTH_HEADERS)

        assert service.last_top_k == 10

    def test_search_with_custom_namespace_service(self) -> None:
        """Verify search uses the namespace from the user's Firestore doc."""
        service = FakeSearchService(results=list(SAMPLE_RESULTS), namespace="custom-ns")
        app = FastAPI()
        router = SearchFastAPIRouter(search_service_instance=service, auth_connector=FakeAuthConnector())
        app.include_router(router.router)
        client = TestClient(app)

        client.get("/search", params={"query": "my search"}, headers=AUTH_HEADERS)

        assert service.last_query == "my search"
        assert service.last_namespace == "custom-ns"

    def test_search_missing_query_returns_error(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify missing query parameter returns 422."""
        client, _ = test_client
        resp = client.get("/search", headers=AUTH_HEADERS)

        assert resp.status_code == 422  # FastAPI validation error

    def test_search_empty_results(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify empty results are handled correctly."""
        client, service = test_client
        service.results = []

        resp = client.get("/search", params={"query": "nonexistent"}, headers=AUTH_HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    def test_search_service_error_returns_500(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify service errors return 500."""
        client, service = test_client
        service.should_raise = Exception("Database connection failed")

        resp = client.get("/search", params={"query": "test"}, headers=AUTH_HEADERS)

        assert resp.status_code == 500
        assert "Database connection failed" in resp.json()["detail"]

    def test_search_timing_is_positive(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify timing is a positive number."""
        client, _ = test_client
        resp = client.get("/search", params={"query": "test"}, headers=AUTH_HEADERS)

        data = resp.json()
        assert data["timing"]["total_s"] >= 0


class TestSearchAuth:
    """Test auth protection on /search endpoint."""

    def test_search_rejects_missing_auth(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify search returns 401 without auth header."""
        app = FastAPI()
        from auth.auth_connector import AuthConnector
        service = FakeSearchService()
        auth = AuthConnector(domain="test.auth0.com", audience="https://test-api")
        router = SearchFastAPIRouter(search_service_instance=service, auth_connector=auth)
        app.include_router(router.router)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/search", params={"query": "test"})
        assert resp.status_code == 401

    def test_search_rejects_invalid_auth_scheme(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify search returns 401 with non-Bearer auth."""
        app = FastAPI()
        from auth.auth_connector import AuthConnector
        service = FakeSearchService()
        auth = AuthConnector(domain="test.auth0.com", audience="https://test-api")
        router = SearchFastAPIRouter(search_service_instance=service, auth_connector=auth)
        app.include_router(router.router)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/search", params={"query": "test"}, headers={"Authorization": "Basic abc123"})
        assert resp.status_code == 401


class TestRouterInitialization:
    """Test SearchFastAPIRouter initialization."""

    def test_router_stores_service_instance(self) -> None:
        """Verify router stores the service instance."""
        service = FakeSearchService()
        router = SearchFastAPIRouter(search_service_instance=service)

        assert router.search_service is service

    def test_router_stores_auth_connector(self) -> None:
        """Verify router stores the auth connector."""
        service = FakeSearchService()
        auth = FakeAuthConnector()
        router = SearchFastAPIRouter(search_service_instance=service, auth_connector=auth)

        assert router.auth_connector is auth

    def test_router_registers_routes(self) -> None:
        """Verify router registers expected routes."""
        service = FakeSearchService()
        router = SearchFastAPIRouter(search_service_instance=service)

        routes = [route.path for route in router.router.routes]
        assert "/health" in routes
        assert "/search" in routes
