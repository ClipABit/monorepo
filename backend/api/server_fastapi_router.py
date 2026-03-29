__all__ = ["ServerFastAPIRouter"]

import asyncio
import logging
import math

import modal
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

logger = logging.getLogger(__name__)

# Rough estimate: 1 vector per second of video, ~500KB per second of video
BYTES_PER_VECTOR_ESTIMATE = 500_000


class ServerFastAPIRouter:
    """
    FastAPI router for the Server service.

    Handles: health, status, upload, list_videos, delete, cache operations, quota.
    Search is handled separately by SearchService with its own ASGI app.
    """

    def __init__(
        self,
        server_instance,
        is_file_change_enabled: bool,
        environment: str = "dev",
        processing_service_cls=None
    ):
        """
        Initializes the API routes, giving them access to the server instance
        for calling background tasks and accessing shared state.

        Args:
            server_instance: The Modal server instance for accessing connectors and spawning local methods
            is_file_change_enabled: Whether this file change is enabled in this environment
            environment: Environment name (dev, staging, prod) for cross-app lookups
            processing_service_cls: Optional ProcessingService class for dev combined mode (direct access)
        """
        self.server_instance = server_instance
        self.is_file_change_enabled = is_file_change_enabled
        self.environment = environment
        self.processing_service_cls = processing_service_cls
        self.router = APIRouter()

        # Initialize UploadHandler with process_video spawn function
        from services.upload_handler import UploadHandler
        self.upload_handler = UploadHandler(
            job_store=server_instance.job_store,
            process_video_spawn_fn=self._get_process_video_spawn_fn()
        )

        self._register_routes()

    def _get_process_video_spawn_fn(self):
        """
        Create a spawn function that works in both dev combined and production modes.

        Returns:
            Callable that spawns process_video_background
        """
        def spawn_process_video(video_bytes: bytes, filename: str, job_id: str, namespace: str, parent_batch_id: str, user_id: str = None, hashed_identifier: str = "", project_id: str = ""):
            try:
                if self.processing_service_cls:
                    # Dev combined mode - direct access
                    self.processing_service_cls().process_video_background.spawn(
                        video_bytes, filename, job_id, namespace, parent_batch_id, user_id, hashed_identifier, project_id
                    )
                    logger.info(f"[Upload] Spawned processing job {job_id} (dev combined mode)")
                else:
                    # Production mode - cross-app call
                    from shared.config import get_modal_environment
                    processing_app_name = f"{self.environment}-processing"
                    ProcessingService = modal.Cls.from_name(
                        processing_app_name,
                        "ProcessingService",
                        environment_name=get_modal_environment()
                    )
                    ProcessingService().process_video_background.spawn(
                        video_bytes, filename, job_id, namespace, parent_batch_id, user_id, hashed_identifier, project_id
                    )
                    logger.info(f"[Upload] Spawned processing job {job_id} to {processing_app_name}")
            except Exception as e:
                logger.error(f"[Upload] Failed to spawn processing job {job_id}: {e}")
                raise

        return spawn_process_video

    def _register_routes(self):
        """Registers all the FastAPI routes."""
        auth = [Depends(self.server_instance.auth_connector)]

        self.router.add_api_route("/health", self.health, methods=["GET"])
        self.router.add_api_route("/status", self.status, methods=["GET"], dependencies=auth)
        self.router.add_api_route("/quota", self.quota, methods=["GET"])
        # Upload, list_videos, clear_cache handle auth manually to get user_id
        self.router.add_api_route("/upload", self.upload, methods=["POST"])
        self.router.add_api_route("/videos", self.list_videos, methods=["GET"])
        # Delete is deactivated — will be re-implemented as a separate feature
        self.router.add_api_route("/cache/clear", self.clear_cache, methods=["POST"])

    async def _get_user_id(self, request: Request) -> str:
        """Extract user_id from request via auth connector."""
        return await self.server_instance.auth_connector(request)

    async def _get_user_data(self, request: Request) -> tuple:
        """
        Authenticate and resolve user data including namespace.

        Returns:
            (user_id, user_data) where user_data includes namespace, vector_count, etc.
        """
        user_id = await self._get_user_id(request)
        loop = asyncio.get_running_loop()
        user_data = await loop.run_in_executor(
            None, self.server_instance.user_store.get_or_create_user, user_id
        )
        return user_id, user_data

    async def health(self):
        """
        Health check endpoint.
        Returns a simple status message indicating the service is running.
        """
        return {"status": "ok"}

    async def status(self, job_id: str):
        """
        Check the status of a video processing job.

        Args:
            job_id (str): The unique identifier for the video processing job.

        Returns:
            dict: Contains:
                - job_id (str): The job identifier
                - status (str): 'processing', 'completed', or 'failed'
                - message (str, optional): If still processing or not found
                - result (dict, optional): Full job result if completed

        This endpoint allows clients (e.g., frontend) to poll for job progress and retrieve results when ready.
        """
        job_data = self.server_instance.job_store.get_job(job_id)
        if job_data is None:
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Job is still processing or not found"
            }
        return job_data

    async def quota(self, request: Request):
        """
        Get current vector quota usage for the authenticated user.

        Returns:
            dict: user_id, namespace, vector_count, vector_quota, vectors_remaining
        """
        user_id, user_data = await self._get_user_data(request)
        vector_count = user_data.get("vector_count", 0)
        vector_quota = user_data.get("vector_quota", 10_000)
        return {
            "user_id": user_id,
            "namespace": user_data.get("namespace", ""),
            "vector_count": vector_count,
            "vector_quota": vector_quota,
            "vectors_remaining": max(0, vector_quota - vector_count),
        }

    async def upload(self, request: Request, files: list[UploadFile] = File(default=[]), namespace: str = Form(""), hashed_identifier: str = Form(""), project_id: str = Form("")):
        """
        Handle video file upload and start background processing.
        Supports both single and batch uploads.

        Authenticates user, checks vector quota, resolves user namespace,
        and returns namespace + quota info in response for plugin local storage.

        Args:
            request: FastAPI Request object for auth extraction
            files (list[UploadFile]): List of uploaded video file(s).
            namespace (str, optional): Ignored — user's assigned namespace is always used.
            hashed_identifier (str, optional): Client-generated hash identifying the video file.
                Generated by the plugin so both sides have the identifier without round-trip.
            project_id (str, optional): Client-provided Resolve project identifier for metadata filtering.

        Returns:
            dict: For single upload: job_id, filename, namespace, quota info, etc.
                  For batch upload: batch_job_id, total_videos, namespace, quota info, etc.

        Raises:
            HTTPException: 400 if validation fails, 429 if quota exceeded, 500 if processing errors
        """
        # Authenticate and resolve user namespace
        user_id, user_data = await self._get_user_data(request)
        user_namespace = user_data.get("namespace", "")

        # Validate required fields (after auth so unauthenticated requests get 401)
        if not hashed_identifier or not hashed_identifier.strip():
            raise HTTPException(status_code=400, detail="hashed_identifier is required — the plugin must generate a hash of the video file")

        if not user_namespace:
            logger.error(f"[Upload] No namespace resolved for user {user_id}")
            raise HTTPException(status_code=500, detail="Failed to resolve user namespace")

        # Check vector quota
        loop = asyncio.get_running_loop()
        is_ok, current_count, vector_quota = await loop.run_in_executor(
            None, self.server_instance.user_store.check_quota, user_id
        )
        if not is_ok:
            raise HTTPException(
                status_code=429,
                detail=f"Vector quota exceeded ({current_count}/{vector_quota}). Delete some videos to free up space."
            )

        # Use user's assigned namespace, not client-provided
        result = await self.upload_handler.handle_upload(files, user_namespace, user_id, hashed_identifier, project_id)

        # Estimate new vectors from total file size (~1 vector per second, ~500KB/s bitrate)
        total_size = result.get("size_bytes", 0)
        if not total_size and "total_submitted" in result:
            # Batch uploads don't have a single size_bytes; estimate will be 0
            total_size = 0
        estimated_new_vectors = max(1, math.ceil(total_size / BYTES_PER_VECTOR_ESTIMATE)) if total_size else 0

        # Add namespace and quota info to response for plugin local storage
        result["namespace"] = user_namespace
        result["vector_count"] = current_count
        result["vector_quota"] = vector_quota
        result["estimated_new_vectors"] = estimated_new_vectors

        return result

    async def list_videos(
        self,
        request: Request,
        page_size: int = 20,
        page_token: str | None = None,
    ):
        """
        List all videos stored in R2 for the authenticated user's namespace.

        Returns:
            json: dict with 'status', 'namespace', and 'videos' list

        Raises:
            HTTPException: If fetching videos fails (500 Internal Server Error)
        """
        user_id, user_data = await self._get_user_data(request)
        namespace = user_data.get("namespace", "__default__")

        if page_size <= 0:
            raise HTTPException(status_code=400, detail="page_size must be positive")

        logger.info(
            "[List Videos] Fetching videos for namespace: %s (page_size=%s, page_token=%s)",
            namespace,
            page_size,
            page_token,
        )
        try:
            videos, next_token, total_videos, total_pages = (
                self.server_instance.r2_connector.list_videos_page(
                    namespace=namespace,
                    page_size=page_size,
                    continuation_token=page_token,
                )
            )
            return {
                "status": "success",
                "namespace": namespace,
                "videos": videos,
                "next_page_token": next_token,
                "total_videos": total_videos,
                "total_pages": total_pages,
            }
        except Exception as e:
            logger.error(f"[List Videos] Error fetching videos: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def clear_cache(self, request: Request):
        """
        Clear the URL cache for the authenticated user's namespace.

        Returns:
            dict: Contains status and number of entries cleared

        Raises:
            HTTPException: If cache clearing is not allowed (403 Forbidden)
        """
        user_id, user_data = await self._get_user_data(request)
        namespace = user_data.get("namespace", "__default__")

        logger.info(f"[Clear Cache] Request to clear cache for namespace: {namespace}")
        if not self.is_file_change_enabled:
            raise HTTPException(status_code=403, detail="Cache clearing is not allowed in the current environment.")

        try:
            cleared_count = self.server_instance.r2_connector.clear_cache(namespace)
            logger.info(f"[Clear Cache] Cleared {cleared_count} cache entries for namespace: {namespace}")
            return {
                "status": "success",
                "namespace": namespace,
                "cleared_entries": cleared_count,
                "message": f"Successfully cleared {cleared_count} cache entries"
            }
        except Exception as e:
            logger.error(f"[Clear Cache] Error clearing cache: {e}")
            raise HTTPException(status_code=500, detail=str(e))
