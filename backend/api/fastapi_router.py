__all__ = ["FastAPIRouter"]

import logging
import time
import uuid

import modal
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)


class FastAPIRouter:
    def __init__(
        self,
        server_instance,
        is_internal_env: bool,
        environment: str = "dev",
        search_service_cls=None,
        processing_service_cls=None
    ):
        """
        Initializes the API routes, giving them access to the server instance
        for calling background tasks and accessing shared state.

        Args:
            server_instance: The Modal server instance for accessing connectors and spawning local methods
            is_internal_env: Whether this is an internal (dev/staging) environment
            environment: Environment name (dev, staging, prod) for cross-app lookups
            search_service_cls: Optional SearchService class for dev combined mode (direct access)
            processing_service_cls: Optional ProcessingService class for dev combined mode (direct access)
        """
        self.server_instance = server_instance
        self.is_internal_env = is_internal_env
        self.environment = environment
        self.search_service_cls = search_service_cls
        self.processing_service_cls = processing_service_cls
        self.router = APIRouter()

        # Initialize UploadHandler with process_video spawn function
        from services.upload import UploadHandler
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
        self.router.add_api_route("/health", self.health, methods=["GET"])
        self.router.add_api_route("/status", self.status, methods=["GET"])
        self.router.add_api_route("/upload", self.upload, methods=["POST"])
        self.router.add_api_route("/search", self.search, methods=["GET"])
        self.router.add_api_route("/videos", self.list_videos, methods=["GET"])
        self.router.add_api_route("/videos/{hashed_identifier}", self.delete_video, methods=["DELETE"])

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

            # Call search app
            if self.search_service_cls:
                # Dev combined mode - direct access to worker in same app
                results = self.search_service_cls().search.remote(query, namespace, top_k)
            else:
                # Production mode - cross-app call via from_name
                from shared.config import get_modal_environment
                search_app_name = f"{self.environment}-search"
                SearchService = modal.Cls.from_name(
                    search_app_name,
                    "SearchService",
                    environment_name=get_modal_environment()
                )
                results = SearchService().search.remote(query, namespace, top_k)

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
        if not self.is_internal_env:
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

