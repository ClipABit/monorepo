"""
Unit tests for SearchFastAPIRouter.

Tests the search API endpoint with mocked SearchService.
"""

from typing import Any, List, Dict
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.search_fastapi_router import SearchFastAPIRouter


class FakeSearchService:
    """Fake SearchService for testing the router."""

    def __init__(self, results: List[Dict[str, Any]] | None = None):
        self.results = results or []
        self.last_query: str | None = None
        self.last_namespace: str | None = None
        self.last_top_k: int | None = None
        self.should_raise: Exception | None = None

    def _search_internal(self, query: str, namespace: str = "", top_k: int = 10) -> List[Dict[str, Any]]:
        """Mock search implementation."""
        self.last_query = query
        self.last_namespace = namespace
        self.last_top_k = top_k

        if self.should_raise:
            raise self.should_raise

        return self.results


@pytest.fixture()
def search_service() -> FakeSearchService:
    """Create a fake search service with sample results."""
    return FakeSearchService(
        results=[
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
    )


@pytest.fixture()
def test_client(search_service: FakeSearchService) -> tuple[TestClient, FakeSearchService]:
    """Create FastAPI test client with search router."""
    app = FastAPI()
    router = SearchFastAPIRouter(search_service_instance=search_service)
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


class TestSearchEndpoint:
    """Test /search endpoint."""

    def test_search_with_query_returns_results(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify search returns results for a query."""
        client, service = test_client
        resp = client.get("/search", params={"query": "woman on a train"})

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
        client.get("/search", params={"query": "test query"})

        assert service.last_query == "test query"

    def test_search_with_namespace(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify namespace parameter is passed to service."""
        client, service = test_client
        client.get("/search", params={"query": "test", "namespace": "my-namespace"})

        assert service.last_namespace == "my-namespace"

    def test_search_with_default_namespace(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify default namespace is empty string."""
        client, service = test_client
        client.get("/search", params={"query": "test"})

        assert service.last_namespace == ""

    def test_search_with_top_k(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify top_k parameter is passed to service."""
        client, service = test_client
        client.get("/search", params={"query": "test", "top_k": 20})

        assert service.last_top_k == 20

    def test_search_with_default_top_k(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify default top_k is 10."""
        client, service = test_client
        client.get("/search", params={"query": "test"})

        assert service.last_top_k == 10

    def test_search_with_all_parameters(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify all parameters are passed correctly."""
        client, service = test_client
        client.get("/search", params={
            "query": "my search",
            "namespace": "custom-ns",
            "top_k": 5
        })

        assert service.last_query == "my search"
        assert service.last_namespace == "custom-ns"
        assert service.last_top_k == 5

    def test_search_missing_query_returns_error(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify missing query parameter returns 422."""
        client, _ = test_client
        resp = client.get("/search")

        assert resp.status_code == 422  # FastAPI validation error

    def test_search_empty_results(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify empty results are handled correctly."""
        client, service = test_client
        service.results = []

        resp = client.get("/search", params={"query": "nonexistent"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    def test_search_service_error_returns_500(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify service errors return 500."""
        client, service = test_client
        service.should_raise = Exception("Database connection failed")

        resp = client.get("/search", params={"query": "test"})

        assert resp.status_code == 500
        assert "Database connection failed" in resp.json()["detail"]

    def test_search_timing_is_positive(self, test_client: tuple[TestClient, FakeSearchService]) -> None:
        """Verify timing is a positive number."""
        client, _ = test_client
        resp = client.get("/search", params={"query": "test"})

        data = resp.json()
        assert data["timing"]["total_s"] >= 0


class TestRouterInitialization:
    """Test SearchFastAPIRouter initialization."""

    def test_router_stores_service_instance(self) -> None:
        """Verify router stores the service instance."""
        service = FakeSearchService()
        router = SearchFastAPIRouter(search_service_instance=service)

        assert router.search_service is service

    def test_router_registers_routes(self) -> None:
        """Verify router registers expected routes."""
        service = FakeSearchService()
        router = SearchFastAPIRouter(search_service_instance=service)

        routes = [route.path for route in router.router.routes]
        assert "/health" in routes
        assert "/search" in routes
