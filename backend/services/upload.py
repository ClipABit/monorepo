import logging
import uuid
from typing import Tuple
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)


class UploadHandler:
    """Handles video upload validation and orchestration."""

    # Allowed video MIME types
    ALLOWED_MIME_TYPES = {
        'video/mp4',
        'video/mpeg',
        'video/quicktime',  # .mov
        'video/x-msvideo',  # .avi
        'video/x-matroska',  # .mkv
        'video/webm',
        'video/x-flv',
        'application/octet-stream'  # Some clients send this for video files
    }

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.mp4', '.mpeg', '.mpg', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v'}

    # Maximum individual file size: 2GB
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB in bytes

    # Maximum batch size
    MAX_BATCH_SIZE = 200

    def __init__(self, job_store, process_video_method):
        """
        Initialize upload service.

        Args:
            job_store: JobStoreConnector instance for tracking jobs
            process_video_method: Modal method for spawning video processing
        """
        self.job_store = job_store
        self.process_video = process_video_method

    def validate_file(self, file: UploadFile, file_contents: bytes = None) -> Tuple[bool, str]:
        """Validate uploaded file for security and compatibility."""
        if not file.filename:
            return False, "File has no filename"

        # Check path traversal
        if any(c in file.filename for c in ['..', '/', '\\']):
            return False, "Filename contains invalid characters"

        # Check file extension
        filename_lower = file.filename.lower()
        if not any(filename_lower.endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
            return False, f"Invalid file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"

        # Warn about unexpected MIME type (lenient)
        if file.content_type and file.content_type not in self.ALLOWED_MIME_TYPES:
            logger.warning(f"{file.filename}: unexpected MIME {file.content_type}")

        # Check file size (if contents provided)
        if file_contents is not None:
            size = len(file_contents)
            if size == 0:
                return False, "File is empty"
            if size > self.MAX_FILE_SIZE:
                return False, f"File too large ({size / 1024 / 1024:.1f} MB, max {self.MAX_FILE_SIZE / 1024 / 1024:.0f} MB)"

        return True, ""

    async def handle_single_upload(self, file: UploadFile, namespace: str) -> dict:
        """Handle single file upload."""
        # Validation before reading
        is_valid, error_msg = self.validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        job_id = str(uuid.uuid4())
        contents = await file.read()
        file_size = len(contents)

        # Validation after reading
        is_valid, error_msg = self.validate_file(file, contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        self.job_store.create_job(job_id, {
            "job_id": job_id,
            "job_type": "video",
            "parent_batch_id": None,
            "filename": file.filename,
            "status": "processing",
            "size_bytes": file_size,
            "content_type": file.content_type,
            "namespace": namespace
        })

        # Spawn background processing (non-blocking - returns immediately)
        self.process_video.spawn(contents, file.filename, job_id, namespace, None)

        return {
            "job_id": job_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "status": "processing",
            "message": "Video uploaded successfully, processing in background"
        }

    async def handle_batch_upload(self, files: list[UploadFile], namespace: str) -> dict:
        """Handle batch file upload."""
        if not files:
            raise ValueError("Cannot create batch with zero files")

        batch_job_id = f"batch-{uuid.uuid4()}"
        batch_created = False

        try:
            # Validate files and build metadata (no reading yet)
            validated = []
            for file in files:
                is_valid, error_msg = self.validate_file(file)
                if is_valid:
                    validated.append({"job_id": str(uuid.uuid4()), "file": file})
                else:
                    logger.warning(f"[Batch {batch_job_id}] Skipped {file.filename}: {error_msg}")

            if not validated:
                raise HTTPException(status_code=400, detail="All files failed validation")

            # Create batch job FIRST (before spawning children)
            self.job_store.create_batch_job(
                batch_job_id=batch_job_id,
                child_job_ids=[v["job_id"] for v in validated],
                namespace=namespace
            )
            batch_created = True
            logger.info(f"[Batch {batch_job_id}] Created batch with {len(validated)} videos")

            # Process files one at a time (read, validate size, spawn, discard)
            spawned, total_size = [], 0

            for meta in validated:
                try:
                    contents = await meta["file"].read()
                    is_valid, error_msg = self.validate_file(meta["file"], contents)
                    if not is_valid:
                        raise ValueError(error_msg)

                    total_size += len(contents)

                    # Create job and spawn processing
                    self.job_store.create_job(meta["job_id"], {
                        "job_id": meta["job_id"],
                        "job_type": "video",
                        "parent_batch_id": batch_job_id,
                        "filename": meta["file"].filename,
                        "status": "processing",
                        "size_bytes": len(contents),
                        "content_type": meta["file"].content_type,
                        "namespace": namespace
                    })

                    self.process_video.spawn(
                        contents, meta["file"].filename, meta["job_id"], namespace, batch_job_id
                    )
                    spawned.append(meta["job_id"])

                except Exception as e:
                    logger.error(f"[Batch {batch_job_id}] Failed {meta['file'].filename}: {e}")
                    # Mark failed and update parent
                    try:
                        self.job_store.set_job_failed(meta["job_id"], f"Upload failed: {e}")
                        self.job_store.update_batch_on_child_completion(
                            batch_job_id, meta["job_id"],
                            {"job_id": meta["job_id"], "status": "failed",
                             "filename": meta["file"].filename, "error": str(e)}
                        )
                    except Exception as ue:
                        logger.error(f"[Batch {batch_job_id}] Update failed: {ue}")

            if not spawned:
                raise HTTPException(status_code=500, detail="All videos failed to process")

            logger.info(
                f"[Batch {batch_job_id}] Spawned {len(spawned)}/{len(validated)} videos, "
                f"{total_size / 1024 / 1024:.2f} MB"
            )

            return {
                "batch_job_id": batch_job_id,
                "status": "processing",
                "total_videos": len(files),
                "successfully_spawned": len(spawned),
                "failed_at_upload": len(files) - len(spawned),
                "message": "Batch upload complete, videos processing in background"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Batch {batch_job_id}] Batch upload failed: {e}")
            if batch_created:
                try:
                    self.job_store.delete_job(batch_job_id)
                except Exception as ce:
                    logger.error(f"[Batch {batch_job_id}] Cleanup failed: {ce}")
            raise HTTPException(status_code=500, detail=f"Batch upload failed: {e}")

    async def handle_upload(self, files: list[UploadFile], namespace: str) -> dict:
        """
        Handle video file upload (single or batch).

        Args:
            files: List of uploaded video files
            namespace: Namespace for storage isolation

        Returns:
            dict: Upload response with job details
        """
        # Validation
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > self.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size ({len(files)}) exceeds maximum ({self.MAX_BATCH_SIZE})"
            )

        # Single file upload (backward compatible)
        if len(files) == 1:
            return await self.handle_single_upload(files[0], namespace)

        # Batch upload
        else:
            return await self.handle_batch_upload(files, namespace)
