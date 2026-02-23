"""
FastAPI router for the Search service.

Exposes the search endpoint directly from the SearchService,
eliminating the need to go through the Server gateway.
"""

__all__ = ["SearchFastAPIRouter"]

import logging
import time

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)


class SearchFastAPIRouter:
    """
    FastAPI router for the Search service.

    Handles: semantic search queries.
    Exposed directly by SearchService for lower latency (no server hop).
    """

    def __init__(self, search_service_instance):
        """
        Initialize the search router.

        Args:
            search_service_instance: The SearchService instance with embedder and connectors
        """
        self.search_service = search_service_instance
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """Register all search routes."""
        self.router.add_api_route("/health", self.health, methods=["GET"])
        self.router.add_api_route("/search", self.search, methods=["GET"])

    async def health(self):
        """Health check endpoint."""
        return {"status": "ok", "service": "search"}

    async def search(self, query: str, namespace: str = "", top_k: int = 10):
        """
        Search endpoint - accepts a text query and returns semantic search results.

        Args:
            query (str): The search query string (required)
            namespace (str, optional): Namespace for Pinecone search (default: "")
            top_k (int, optional): Number of top results to return (default: 10)

        Returns:
            json: dict with 'query', 'results', and 'timing'

        Raises:
            HTTPException: If search fails (500 Internal Server Error)
        """
        try:
            t_start = time.perf_counter()
            logger.info(f"[Search] Query: '{query}' | namespace='{namespace}' | top_k={top_k}")

            # Call search directly on the service instance (no RPC, no cross-app call)
            results = self.search_service._search_internal(query, namespace, top_k)

            t_done = time.perf_counter()
            logger.info(f"[Search] Found {len(results)} results in {t_done - t_start:.3f}s")

            return {
                "query": query,
                "results": results,
                "timing": {
                    "total_s": round(t_done - t_start, 3)
                }
            }
        except Exception as e:
            logger.error(f"[Search] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
