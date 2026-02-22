__all__ = ["ServerFastAPIRouter"]

import logging
import uuid

import modal
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)


class ServerFastAPIRouter:
    """
    FastAPI router for the Server service.

    Handles: health, status, upload, list_videos, delete, cache operations.
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
        def spawn_process_video(video_bytes: bytes, filename: str, job_id: str, namespace: str, parent_batch_id: str):
            try:
                if self.processing_service_cls:
                    # Dev combined mode - direct access
                    self.processing_service_cls().process_video_background.spawn(
                        video_bytes, filename, job_id, namespace, parent_batch_id
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
                        video_bytes, filename, job_id, namespace, parent_batch_id
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
        self.router.add_api_route("/upload", self.upload, methods=["POST"], dependencies=auth)
        self.router.add_api_route("/videos", self.list_videos, methods=["GET"], dependencies=auth)
        self.router.add_api_route("/videos/{hashed_identifier}", self.delete_video, methods=["DELETE"], dependencies=auth)
        self.router.add_api_route("/cache/clear", self.clear_cache, methods=["POST"], dependencies=auth)

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

    async def upload(self, files: list[UploadFile] = File(default=[]), namespace: str = Form("")):
        """
        Handle video file upload and start background processing.
        Supports both single and batch uploads.

        Args:
            files (list[UploadFile]): List of uploaded video file(s). Client sends files with repeated 'files' field names, which FastAPI collects into a list.
            namespace (str, optional): Namespace for Pinecone and R2 storage (default: "")

        Returns:
            dict: For single upload: job_id, filename, etc.
                  For batch upload: batch_job_id, total_videos, child_jobs, etc.

        Raises:
            HTTPException: 400 if validation fails, 500 if processing errors
        """
        return await self.upload_handler.handle_upload(files, namespace)

    async def list_videos(
        self,
        namespace: str = "__default__",
        page_size: int = 20,
        page_token: str | None = None,
    ):
        """
        List all videos stored in R2 for the given namespace.
        
        Args:
            namespace (str, optional): Namespace for R2 storage (default: "__default__")
        
        Returns:
            json: dict with 'status', 'namespace', and 'videos' list

        Raises:
            HTTPException: If fetching videos fails (500 Internal Server Error)
        """
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

    async def delete_video(self, hashed_identifier: str, filename: str, namespace: str = ""):
        """
        Delete a video and its associated chunks from storage and database.

        Args:
            hashed_identifier (str): The unique identifier of the video in R2 storage.
            filename (str): The original filename of the video.
            namespace (str, optional): Namespace for Pinecone and R2 storage (default: "")

        Returns:
            dict: Contains status and message about deletion result.
        
        Raises:
            HTTPException: If deletion fails at any step.
                - 500 Internal Server Error with details.
                - 400 Bad Request if parameters are missing.
                - 404 Not Found if video does not exist.
                - 403 Forbidden if deletion is not allowed.
        """
        logger.info(f"[Delete Video] Request to delete video: {filename} ({hashed_identifier}) | namespace='{namespace}'")
        if not self.is_file_change_enabled:
            raise HTTPException(status_code=403, detail="Video deletion is not allowed in the current environment.")

        job_id = str(uuid.uuid4())
        self.server_instance.job_store.create_job(job_id, {
            "job_id": job_id,
            "hashed_identifier": hashed_identifier,
            "namespace": namespace,
            "status": "processing",
            "operation": "delete"
        })

        self.server_instance.delete_video_background.spawn(job_id, hashed_identifier, namespace)

        return {
            "job_id": job_id,
            "hashed_identifier": hashed_identifier,
            "namespace": namespace,
            "status": "processing",
            "message": "Video deletion started, processing in background"
        }

    async def clear_cache(self, namespace: str = "__default__"):
        """
        Clear the URL cache for a given namespace.
        
        Args:
            namespace (str, optional): Namespace to clear cache for (default: "__default__")
        
        Returns:
            dict: Contains status and number of entries cleared
        
        Raises:
            HTTPException: If cache clearing is not allowed (403 Forbidden)
        """
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

