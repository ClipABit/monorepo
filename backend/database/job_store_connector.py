import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from copy import deepcopy
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
                # Return a deep copy to prevent accidental mutations
                return deepcopy(job_data)
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

    def create_batch_job(
        self,
        batch_job_id: str,
        child_job_ids: List[str],
        namespace: str
    ) -> bool:
        """Create a new batch job entry."""
        batch_data = {
            "batch_job_id": batch_job_id,
            "job_type": "batch",
            "status": "processing",
            "namespace": namespace,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_videos": len(child_job_ids),
            "child_jobs": child_job_ids,
            "completed_count": 0,
            "failed_count": 0,
            "processing_count": len(child_job_ids),
            "total_chunks": 0,
            "total_frames": 0,
            "total_memory_mb": 0.0,
            "avg_complexity": 0.0,
            "failed_jobs": [],
            "completed_jobs": [],
            "_version": 0  # Optimistic locking version
        }
        return self.create_job(batch_job_id, batch_data)

    def update_batch_on_child_completion(
        self,
        batch_job_id: str,
        child_job_id: str,
        child_result: Dict[str, Any],
        max_retries: int = 10
    ) -> bool:
        """
        Update batch job when a child completes (success or failure).

        Uses optimistic locking with version-based retries to prevent race conditions
        when multiple child jobs complete simultaneously.
        """
        for attempt in range(max_retries):
            # Read current batch state
            batch_job = self.get_job(batch_job_id)
            if not batch_job:
                logger.error(f"Batch job {batch_job_id} not found")
                return False

            # Get current version for optimistic locking
            expected_version = batch_job.get("_version", 0)

            # Update counts
            child_status = child_result.get("status")

            if child_status == "completed":
                batch_job["completed_count"] += 1
                batch_job["processing_count"] -= 1

                # Aggregate metrics
                batch_job["total_chunks"] += child_result.get("chunks", 0)
                batch_job["total_frames"] += child_result.get("total_frames", 0)
                batch_job["total_memory_mb"] += child_result.get("total_memory_mb", 0.0)

                # Update average complexity (running average)
                prev_avg = batch_job["avg_complexity"]
                n = batch_job["completed_count"]
                new_complexity = child_result.get("avg_complexity", 0.0)
                batch_job["avg_complexity"] = (prev_avg * (n - 1) + new_complexity) / n

                # Track completed job summary
                batch_job["completed_jobs"].append({
                    "job_id": child_job_id,
                    "filename": child_result.get("filename"),
                    "chunks": child_result.get("chunks", 0),
                    "frames": child_result.get("total_frames", 0)
                })

            elif child_status == "failed":
                batch_job["failed_count"] += 1
                batch_job["processing_count"] -= 1

                # Track failed job details
                batch_job["failed_jobs"].append({
                    "job_id": child_job_id,
                    "filename": child_result.get("filename"),
                    "error": child_result.get("error", "Unknown error")
                })

            else:
                # Unexpected status value - treat as failure to prevent orphaned processing count
                logger.error(
                    f"Unexpected child status '{child_status}' for job {child_job_id} "
                    f"in batch {batch_job_id}. Expected 'completed' or 'failed'. "
                    f"Treating as failure."
                )
                batch_job["failed_count"] += 1
                batch_job["processing_count"] -= 1

                # Track as failed job with detailed error
                batch_job["failed_jobs"].append({
                    "job_id": child_job_id,
                    "filename": child_result.get("filename", "unknown"),
                    "error": f"Invalid status: {child_status}"
                })

            # Update batch status
            total = batch_job["total_videos"]
            completed = batch_job["completed_count"]
            failed = batch_job["failed_count"]

            if completed + failed == total:
                # All jobs finished
                if failed == 0:
                    batch_job["status"] = "completed"
                elif completed == 0:
                    batch_job["status"] = "failed"
                else:
                    batch_job["status"] = "partial"

            batch_job["updated_at"] = datetime.now(timezone.utc).isoformat()
            batch_job["_version"] = expected_version + 1

            # Attempt atomic update with version check
            try:
                # Verify version hasn't changed (optimistic locking)
                # Access Modal Dict directly to minimize race window
                if batch_job_id in self.job_store:
                    current_version = self.job_store[batch_job_id].get("_version", 0)
                    if current_version == expected_version:
                        # Version matches, safe to update
                        self.job_store[batch_job_id] = batch_job
                        logger.info(
                            f"Updated batch {batch_job_id} for child {child_job_id} "
                            f"(attempt {attempt + 1}, version {expected_version} -> {expected_version + 1})"
                        )
                        return True
                    else:
                        # Version mismatch, retry
                        logger.warning(
                            f"Version mismatch for batch {batch_job_id} "
                            f"(expected {expected_version}, got {current_version}). "
                            f"Retrying... (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                else:
                    logger.error(f"Batch job {batch_job_id} disappeared during update")
                    return False
            except Exception as e:
                logger.error(f"Error updating batch {batch_job_id}: {e}")
                return False

        # Max retries exceeded
        logger.error(
            f"Failed to update batch {batch_job_id} after {max_retries} attempts. "
            f"Concurrent updates are too frequent."
        )
        return False

    def get_batch_child_jobs(self, batch_job_id: str) -> List[Dict[str, Any]]:
        """Retrieve all child job data for a batch."""
        batch_job = self.get_job(batch_job_id)
        if not batch_job or batch_job.get("job_type") != "batch":
            return []

        child_job_ids = batch_job.get("child_jobs", [])
        child_jobs = []
        for child_id in child_job_ids:
            child_data = self.get_job(child_id)
            if child_data:
                child_jobs.append(child_data)

        return child_jobs

    def get_batch_progress(self, batch_job_id: str) -> Optional[Dict[str, Any]]:
        """Get simplified batch progress summary."""
        batch_job = self.get_job(batch_job_id)
        if not batch_job or batch_job.get("job_type") != "batch":
            return None

        # Calculate progress percentage, handling empty batch case
        total = batch_job["total_videos"]
        if total > 0:
            progress_percent = (
                (batch_job["completed_count"] + batch_job["failed_count"])
                / total * 100
            )
        else:
            progress_percent = 0.0

        return {
            "batch_job_id": batch_job_id,
            "status": batch_job["status"],
            "total_videos": total,
            "completed": batch_job["completed_count"],
            "failed": batch_job["failed_count"],
            "processing": batch_job["processing_count"],
            "progress_percent": progress_percent
        }
