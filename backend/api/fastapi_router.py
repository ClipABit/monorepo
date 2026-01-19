__all__ = ["FastAPIRouter"]

import logging
import time
import uuid

import modal
from fastapi import APIRouter, Form, HTTPException, UploadFile

from shared.config import get_modal_environment

logger = logging.getLogger(__name__)


class FastAPIRouter:
    def __init__(self, server_instance, is_internal_env: bool, environment: str = "dev"):
        """
        Initializes the API routes, giving them access to the server instance
        for calling background tasks and accessing shared state.
        
        Args:
            server_instance: The Modal server instance for accessing connectors and spawning local methods
            is_internal_env: Whether this is an internal (dev/staging) environment
            environment: Environment name (dev, staging, prod) for cross-app lookups
        """
        self.server_instance = server_instance
        self.is_internal_env = is_internal_env
        self.environment = environment
        self.router = APIRouter()
        self._register_routes()

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

    async def upload(self, file: UploadFile, namespace: str = Form("")):
        """
        Handle video file upload and start background processing.

        Args:
            file (UploadFile): The uploaded video file.
            namespace (str, optional): Namespace for Pinecone and R2 storage (default: "")

        Returns:
            dict: Contains job_id, filename, content_type, size_bytes, status, and message.
        """
        contents = await file.read()
        file_size = len(contents)
        job_id = str(uuid.uuid4())

        self.server_instance.job_store.create_job(job_id, {
            "job_id": job_id,
            "filename": file.filename,
            "status": "processing",
            "size_bytes": file_size,
            "content_type": file.content_type,
            "namespace": namespace
        })

        # Spawn to processing app (cross-app call)
        try:
            processing_app_name = f"{self.environment} processing"
            ProcessingWorker = modal.Cls.from_name(processing_app_name, "ProcessingWorker", environment_name=get_modal_environment())
            ProcessingWorker().process_video_background.spawn(contents, file.filename, job_id, namespace)
            logger.info(f"[Upload] Spawned processing job {job_id} to {processing_app_name}")
        except Exception as e:
            logger.error(f"[Upload] Failed to spawn processing job: {e}")
            self.server_instance.job_store.set_job_failed(job_id, f"Failed to start processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

        return {
            "job_id": job_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "status": "processing",
            "message": "Video uploaded successfully, processing in background"
        }

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

            # Call search app (cross-app call)
            search_app_name = f"{self.environment} search"
            SearchWorker = modal.Cls.from_name(search_app_name, "SearchWorker", environment_name=get_modal_environment())
            results = SearchWorker().search.remote(query, namespace, top_k)

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

