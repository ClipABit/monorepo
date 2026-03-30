"""
FastAPI router for the Search service.

Exposes the search endpoint directly from the SearchService,
eliminating the need to go through the Server gateway.
"""

__all__ = ["SearchFastAPIRouter", "limiter"]

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


class SearchFastAPIRouter:
    """
    FastAPI router for the Search service.

    Handles: semantic search queries.
    Exposed directly by SearchService for lower latency (no server hop).
    """

    def __init__(self, search_service_instance, auth_connector=None):
        """
        Initialize the search router.

        Args:
            search_service_instance: The SearchService instance with embedder and connectors
            auth_connector: Optional AuthConnector for JWT verification
        """
        self.search_service = search_service_instance
        self.auth_connector = auth_connector
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """Register all search routes."""
        self.router.add_api_route("/health", self.health, methods=["GET"])
        # Search handles auth manually to extract user_id and resolve namespace
        self.router.add_api_route(
            "/search", self.search, methods=["GET"]
        )

        # Apply the limiter to the bound method at registration time.
        # This ensures 'request' is at index 0, which avoids the slowapi IndexError for bound class methods.
        self.router.add_api_route(
            "/demo-search", limiter.limit("5/minute")(self.demo_search), methods=["GET"]
        )

    async def health(self):
        """Health check endpoint."""
        return {"status": "ok", "service": "search"}

    async def demo_search(self, request: Request, query: str, top_k: int = 10):
        """
        Public demo search endpoint - accepts a text query and returns semantic search results for the demo namespace.
        Rate limited.
        """
        try:
            t_start = time.perf_counter()
            namespace = "web-demo"
            logger.info(
                f"[Search] Demo Query: '{query}' | namespace='{namespace}' | top_k={top_k}"
            )

            # Call search directly on the service instance (no RPC, no cross-app call)
            results = self.search_service._search_internal(query, namespace, top_k)

            t_done = time.perf_counter()
            logger.info(
                f"[Search] Found {len(results)} demo results in {t_done - t_start:.3f}s"
            )

            return {
                "query": query,
                "results": results,
                "timing": {"total_s": round(t_done - t_start, 3)},
            }
        except Exception:
            logger.exception("[Search] Error in demo search")
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while processing the demo search request.",
            )

    async def search(self, request: Request, query: str, top_k: int = 10):
        """
        Search endpoint - accepts a text query and returns semantic search results.
        Authenticates user and searches their assigned namespace.

        Args:
            request: FastAPI Request object for auth extraction
            query (str): The search query string (required)
            top_k (int, optional): Number of top results to return (default: 10)

        Returns:
            json: dict with 'query', 'results', and 'timing'

        Raises:
            HTTPException: If search fails (500 Internal Server Error)
        """
        try:
            # Authenticate and resolve user namespace
            if not self.auth_connector:
                raise HTTPException(status_code=401, detail="Authentication is not configured")
            user_id = await self.auth_connector(request)
            import asyncio
            loop = asyncio.get_running_loop()
            user_data = await loop.run_in_executor(
                None, self.search_service.user_store.get_or_create_user, user_id
            )
            namespace = user_data.get("namespace", "")

            t_start = time.perf_counter()
            logger.info(
                f"[Search] Query: '{query}' | namespace='{namespace}' | user={user_id} | top_k={top_k}"
            )

            # Filter results to this user only (shared namespace isolation)
            metadata_filter = {"user_id": {"$eq": user_id}}

            try:
                results = self.search_service._search_plugin(query, namespace, top_k, metadata_filter=metadata_filter)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

            t_done = time.perf_counter()
            logger.info(
                f"[Search] Found {len(results)} results in {t_done - t_start:.3f}s"
            )

            return {
                "query": query,
                "results": results,
                "timing": {"total_s": round(t_done - t_start, 3)},
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Search] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
