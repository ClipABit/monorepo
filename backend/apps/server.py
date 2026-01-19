"""
Server Modal App

Handles all HTTP endpoints with minimal dependencies for fast cold starts (~3-5s).
"""

import logging
import modal

from shared.config import (
    get_environment,
    get_secrets,
    is_internal_env,
    get_pinecone_index,
    get_env_var,
)
from shared.images import get_server_image

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()
IS_INTERNAL_ENV = is_internal_env()

# Create Modal app with minimal image
app = modal.App(
    name=f"{env}-api",
    image=get_server_image(),
    secrets=[get_secrets()]
)


@app.cls(cpu=2.0, memory=2048, timeout=120)
class Server:

    @modal.enter()
    def startup(self):
        from datetime import datetime, timezone
        from database.pinecone_connector import PineconeConnector
        from database.job_store_connector import JobStoreConnector
        from database.r2_connector import R2Connector
        from api import FastAPIRouter
        from fastapi import FastAPI

        logger.info(f"Server container starting up! Environment = {env}")
        self.start_time = datetime.now(timezone.utc)

        # Get environment variables
        PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
        R2_ACCOUNT_ID = get_env_var("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = get_env_var("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = get_env_var("R2_SECRET_ACCESS_KEY")

        pinecone_index = get_pinecone_index()
        logger.info(f"Using Pinecone index: {pinecone_index}")

        # Initialize lightweight connectors
        self.pinecone_connector = PineconeConnector(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index
        )
        self.job_store = JobStoreConnector(dict_name="clipabit-jobs")
        self.r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=env
        )

        # FastAPI app with routes from FastAPIRouter
        self.fastapi_app = FastAPI(title="Clipabit Server")
        api_router = FastAPIRouter(self, IS_INTERNAL_ENV, env)
        self.fastapi_app.include_router(api_router.router)

        logger.info("Server initialized and ready!")

    @modal.asgi_app()
    def asgi_app(self):
        return self.fastapi_app

    @modal.method()
    def delete_video_background(self, job_id: str, hashed_identifier: str, namespace: str = ""):
        """Background job that deletes a video and all associated chunks."""
        logger.info(f"[Job {job_id}] Deletion started: {hashed_identifier} | namespace='{namespace}'")

        try:
            # Delete chunks from Pinecone
            pinecone_success = self.pinecone_connector.delete_by_identifier(
                hashed_identifier=hashed_identifier,
                namespace=namespace
            )

            if not pinecone_success:
                raise Exception("Failed to delete chunks from Pinecone")

            # Delete from R2
            r2_success = self.r2_connector.delete_video(hashed_identifier)
            if not r2_success:
                logger.critical(
                    f"[Job {job_id}] INCONSISTENCY: Chunks deleted but R2 deletion failed for {hashed_identifier}"
                )
                raise Exception("Failed to delete video from R2 after deleting chunks. System may be inconsistent.")

            result = {
                "job_id": job_id,
                "status": "completed",
                "hashed_identifier": hashed_identifier,
                "namespace": namespace,
                "r2": {"deleted": r2_success},
                "pinecone": {"deleted": pinecone_success}
            }

            logger.info(f"[Job {job_id}] Deletion completed: R2={r2_success}, Pinecone={pinecone_success}")
            self.job_store.set_job_completed(job_id, result)

            # Clear cache
            try:
                self.r2_connector.clear_cache(namespace or "__default__")
            except Exception as cache_exc:
                logger.error(f"[Job {job_id}] Failed to clear cache: {cache_exc}")

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Job {job_id}] Deletion failed: {error_msg}")

            import traceback
            traceback.print_exc()

            self.job_store.set_job_failed(job_id, error_msg)
            return {"job_id": job_id, "status": "failed", "error": error_msg}
