import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobStoreConnector:
    """
    Modal Dict wrapper for cross-container job status tracking.

    Provides CRUD operations for job results shared across Modal containers.
    All operations include error handling and logging.
    """

    DEFAULT_DICT_NAME = "clipabit-jobs"

    def __init__(self, dict_name: str = DEFAULT_DICT_NAME):
        self.dict_name = dict_name
        self.job_store = modal.Dict.from_name(dict_name, create_if_missing=True)
        logger.info(f"Initialized JobStoreConnector with Dict: {dict_name}")

    def create_job(self, job_id: str, initial_data: Dict[str, Any]) -> bool:
        """Create new job entry in store."""
        try:
            # Add backward compatible fields if not present
            if "job_type" not in initial_data:
                initial_data["job_type"] = "video"
            if "parent_batch_id" not in initial_data:
                initial_data["parent_batch_id"] = None

            self.job_store[job_id] = initial_data
            logger.info(f"Created job {job_id} with status: {initial_data.get('status', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error creating job {job_id}: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job data from store, returns None if not found."""
        try:
            if job_id in self.job_store:
                job_data = self.job_store[job_id]
                logger.info(f"Retrieved job {job_id} with status: {job_data.get('status', 'unknown')}")
                return job_data
            else:
                logger.info(f"Job {job_id} not found in store")
                return None
        except Exception as e:
            logger.error(f"Error retrieving job {job_id}: {e}")
            return None

    def update_job(self, job_id: str, update_data: Dict[str, Any]) -> bool:
        """Update existing job by merging update_data with current data."""
        try:
            if job_id in self.job_store:
                existing_data = self.job_store[job_id]
                existing_data.update(update_data)
                self.job_store[job_id] = existing_data
                logger.info(f"Updated job {job_id} with new status: {existing_data.get('status', 'unknown')}")
                return True
            else:
                logger.warning(f"Cannot update - job {job_id} not found in store")
                return False
        except Exception as e:
            logger.error(f"Error updating job {job_id}: {e}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """Remove job from store."""
        try:
            if job_id in self.job_store:
                del self.job_store[job_id]
                logger.info(f"Deleted job {job_id} from store")
                return True
            else:
                logger.warning(f"Cannot delete - job {job_id} not found in store")
                return False
        except Exception as e:
            logger.error(f"Error deleting job {job_id}: {e}")
            return False

    def job_exists(self, job_id: str) -> bool:
        """Check if job exists in store."""
        try:
            return job_id in self.job_store
        except Exception as e:
            logger.error(f"Error checking existence of job {job_id}: {e}")
            return False

    def set_job_completed(self, job_id: str, result_data: Dict[str, Any]) -> bool:
        """Mark job as completed with result data."""
        result_data["status"] = "completed"
        return self.update_job(job_id, result_data)

    def set_job_failed(self, job_id: str, error: str) -> bool:
        """Mark job as failed with error message."""
        error_data = {
            "status": "failed",
            "error": error
        }
        return self.update_job(job_id, error_data)
