"""
ServerService class - base class shared between server.py and dev_combined.py
"""

import logging
import modal

from shared.config import get_environment, get_env_var, get_pinecone_index

logger = logging.getLogger(__name__)


class ServerService:
    """
    Server service base class - handles HTTP endpoints and background deletion.
    """

    def _initialize_connectors(self):
        """Initialize connectors (non-decorated, can be called from subclasses)."""
        from datetime import datetime, timezone
        from database.pinecone_connector import PineconeConnector
        from database.job_store_connector import JobStoreConnector
        from database.r2_connector import R2Connector
        from auth.auth_connector import AuthConnector

        env = get_environment()
        logger.info(f"[{self.__class__.__name__}] Starting up in '{env}' environment")
        self.start_time = datetime.now(timezone.utc)

        # Initialize Firebase Admin SDK (required for token verification)
        try:
            import firebase_admin
            if not firebase_admin._apps:
                import json
                firebase_credentials = json.loads(get_env_var("FIREBASE_ADMIN_KEY"))
                from firebase_admin import credentials
                cred = credentials.Certificate(firebase_credentials)
                firebase_admin.initialize_app(cred)
                logger.info(f"[{self.__class__.__name__}] Firebase Admin SDK initialized")
        except Exception as e:
            logger.warning(f"[{self.__class__.__name__}] Firebase initialization failed: {e}")

        # Get environment variables
        PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
        R2_ACCOUNT_ID = get_env_var("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = get_env_var("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = get_env_var("R2_SECRET_ACCESS_KEY")
        IS_FILE_CHANGE_ENABLED = get_env_var("IS_FILE_CHANGE_ENABLED").lower() == "true"

        pinecone_index = get_pinecone_index()
        logger.info(f"[{self.__class__.__name__}] Using Pinecone index: {pinecone_index}")

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
        self.auth_connector = AuthConnector()

        # Store config for router
        self.env = env
        self.is_file_change_enabled = IS_FILE_CHANGE_ENABLED

        logger.info(f"[{self.__class__.__name__}] Initialized and ready!")

    @modal.enter()
    def startup(self):
        """Initialize connectors and FastAPI app."""
        self._initialize_connectors()

    def create_fastapi_app(self, processing_service_cls=None):
        """
        Create FastAPI app with routes.

        Note: Search is now handled by SearchService with its own ASGI app.

        Args:
            processing_service_cls: Optional ProcessingService class for dev combined mode
        """
        from api import ServerFastAPIRouter
        from fastapi import FastAPI

        self.fastapi_app = FastAPI(title="Clipabit Server")
        api_router = ServerFastAPIRouter(
            server_instance=self,
            is_file_change_enabled=self.is_file_change_enabled,
            environment=self.env,
            processing_service_cls=processing_service_cls
        )
        self.fastapi_app.include_router(api_router.router)
        return self.fastapi_app

    @modal.method()
    def delete_video_background(self, job_id: str, hashed_identifier: str, namespace: str = ""):
        """Background job that deletes a video and all associated chunks."""
        logger.info(f"[{self.__class__.__name__}][Job {job_id}] Deletion started: {hashed_identifier} | namespace='{namespace}'")

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
                    f"[{self.__class__.__name__}][Job {job_id}] INCONSISTENCY: Chunks deleted but R2 deletion failed"
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

            logger.info(f"[{self.__class__.__name__}][Job {job_id}] Deletion completed")
            self.job_store.set_job_completed(job_id, result)

            # Clear cache
            try:
                self.r2_connector.clear_cache(namespace or "__default__")
            except Exception as cache_exc:
                logger.error(f"[{self.__class__.__name__}][Job {job_id}] Failed to clear cache: {cache_exc}")

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{self.__class__.__name__}][Job {job_id}] Deletion failed: {error_msg}")

            import traceback
            traceback.print_exc()

            self.job_store.set_job_failed(job_id, error_msg)
            return {"job_id": job_id, "status": "failed", "error": error_msg}
