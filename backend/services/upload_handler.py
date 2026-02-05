"""Upload validation and orchestration service."""

import logging
import uuid
from typing import Tuple
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)


class UploadHandler:
    """Handles video upload validation and orchestration."""

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

    ALLOWED_EXTENSIONS = {'.mp4', '.mpeg', '.mpg', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v'}
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB in bytes
    MAX_BATCH_SIZE = 200

    def __init__(self, job_store, process_video_spawn_fn):
        """
        Initialize upload handler.

        Args:
            job_store: JobStoreConnector instance for job tracking and status updates
            process_video_spawn_fn: Callable that spawns async video processing
                For dev mode: ProcessingService().process_video_background.spawn
                For prod mode: modal.Cls.from_name(...).process_video_background.spawn
        """
        self.job_store = job_store
        self.process_video_spawn = process_video_spawn_fn

    def validate_file(self, file: UploadFile, file_contents: bytes = None) -> Tuple[bool, str]:
        """
        Validate uploaded file for security and compatibility.

        Checks filename, extension, MIME type, size, and path traversal attempts.

        Args:
            file: The uploaded file object
            file_contents: Optional file bytes for size validation

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
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
        """
        Handle single file upload.

        Validates file, creates job entry, spawns background processing, and returns immediately.

        Args:
            file: The uploaded video file
            namespace: Storage namespace for isolation

        Returns:
            dict: Job details with job_id, filename, size, and status

        Raises:
            HTTPException: 400 if validation fails, 500 if processing errors
        """
        is_valid, error_msg = self.validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        job_id = str(uuid.uuid4())
        contents = await file.read()

        is_valid, error_msg = self.validate_file(file, contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        self.job_store.create_job(job_id, {
            "job_id": job_id,
            "job_type": "video",
            "parent_batch_id": None,
            "filename": file.filename,
            "status": "processing",
            "size_bytes": len(contents),
            "content_type": file.content_type,
            "namespace": namespace
        })

        self.process_video_spawn(contents, file.filename, job_id, namespace, None)

        return {
            "job_id": job_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(contents),
            "status": "processing",
            "message": "Video uploaded successfully, processing in background"
        }

    async def handle_batch_upload(self, files: list[UploadFile], namespace: str) -> dict:
        """
        Handle batch file upload with streaming and partial failure support.

        Validates all files, creates parent batch job, then processes files one at a time.
        Memory efficient - only one file in memory at a time.

        Args:
            files: List of uploaded video files
            namespace: Storage namespace for isolation

        Returns:
            dict: Batch details with batch_job_id, counts, and processing status

        Raises:
            ValueError: If files list is empty
            HTTPException: 400 if all files fail validation, 500 if all fail processing
        """
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
                logger.error(f"[Batch {batch_job_id}] All {len(files)} files failed validation")
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

                    self.process_video_spawn(
                        contents, meta["file"].filename, meta["job_id"], namespace, batch_job_id
                    )
                    spawned.append(meta["job_id"])

                except Exception as e:
                    logger.error(f"[Batch {batch_job_id}] Failed {meta['file'].filename}: {e}")
                    # Mark failed and update parent
                    try:
                        self.job_store.set_job_failed(meta["job_id"], f"Upload failed: {e}")
                        update_success = self.job_store.update_batch_on_child_completion(
                            batch_job_id, meta["job_id"],
                            {"job_id": meta["job_id"], "status": "failed",
                             "filename": meta["file"].filename, "error": str(e)}
                        )
                        if not update_success:
                            logger.error(
                                f"[Batch {batch_job_id}] CRITICAL: Failed to update batch for job {meta['job_id']} "
                                f"after max retries. Batch state may be inconsistent."
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
                "total_submitted": len(files),
                "failed_validation": len(files) - len(validated),
                "total_videos": len(validated),
                "successfully_spawned": len(spawned),
                "failed_at_upload": len(validated) - len(spawned),
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
        Handle video file upload - auto-detects single or batch mode.

        Routes to single or batch handler based on file count.

        Args:
            files: List of uploaded video files
            namespace: Namespace for storage isolation

        Returns:
            dict: Single upload returns job_id, batch returns batch_job_id with counts

        Raises:
            HTTPException: 400 if no files provided or batch exceeds limit
        """
        # Debug logging for upload diagnostics
        logger.info(f"[Upload] Received {len(files)} files, namespace='{namespace}'")
        for i, f in enumerate(files):
            logger.info(f"[Upload] File {i}: filename={f.filename!r}, content_type={f.content_type}, size={f.size}")

        # Validation
        if not files:
            logger.error("[Upload] No files provided in request")
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
