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
        """
        Validate uploaded file for security and compatibility.

        Args:
            file: The UploadFile object to validate
            file_contents: Optional file contents for size validation (if already read)

        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        # 1. Check filename exists
        if not file.filename:
            return False, "File has no filename"

        # 2. Check file extension
        filename_lower = file.filename.lower()
        file_ext = None
        for ext in self.ALLOWED_EXTENSIONS:
            if filename_lower.endswith(ext):
                file_ext = ext
                break

        if not file_ext:
            return False, f"Invalid file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"

        # 3. Check MIME type (if provided)
        if file.content_type and file.content_type not in self.ALLOWED_MIME_TYPES:
            # Some clients send generic MIME types, so we're lenient if extension is valid
            logger.warning(
                f"File {file.filename} has unexpected MIME type {file.content_type}, "
                f"but extension {file_ext} is valid. Proceeding."
            )

        # 4. Check file size (if contents provided)
        if file_contents is not None:
            file_size = len(file_contents)
            if file_size == 0:
                return False, "File is empty (0 bytes)"
            if file_size > self.MAX_FILE_SIZE:
                return False, f"File too large ({file_size / 1024 / 1024:.1f} MB). Maximum: {self.MAX_FILE_SIZE / 1024 / 1024:.0f} MB"

        # 5. Check for path traversal attempts in filename
        if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
            return False, "Filename contains invalid characters (path separators)"

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
        # Validate input: ensure files list is not empty
        if not files or len(files) == 0:
            logger.error("Batch upload attempted with empty files list")
            raise ValueError("Cannot create batch with zero files. At least one file is required.")

        # Generate batch job ID
        batch_job_id = f"batch-{uuid.uuid4()}"
        batch_created = False
        successfully_spawned = []

        try:
            # Step 1: Collect metadata and validate (no reading)
            file_metadata = []
            child_job_ids = []
            validation_errors = []

            for idx, file in enumerate(files):
                #  Validate file (filename, extension, MIME type)
                is_valid, error_msg = self.validate_file(file)
                if not is_valid:
                    validation_errors.append(f"File #{idx + 1} ({file.filename}): {error_msg}")
                    logger.warning(f"[Batch {batch_job_id}] Validation failed for {file.filename}: {error_msg}")
                    continue  # Skip invalid files

                job_id = str(uuid.uuid4())
                file_metadata.append({
                    "job_id": job_id,
                    "file": file,
                    "filename": file.filename,
                    "content_type": file.content_type
                })
                child_job_ids.append(job_id)

            # Check if any files passed validation
            if len(file_metadata) == 0:
                error_details = "; ".join(validation_errors[:5])  # Limit to first 5 errors
                if len(validation_errors) > 5:
                    error_details += f" (and {len(validation_errors) - 5} more)"
                raise HTTPException(
                    status_code=400,
                    detail=f"All {len(files)} files failed validation: {error_details}"
                )

            logger.info(
                f"[Batch {batch_job_id}] Starting batch upload with {len(files)} videos"
            )

            # Step 2: Create batch job entry FIRST
            try:
                self.job_store.create_batch_job(
                    batch_job_id=batch_job_id,
                    child_job_ids=child_job_ids,
                    namespace=namespace
                )
                batch_created = True
                logger.info(
                    f"[Batch {batch_job_id}] Created parent job entry with {len(files)} children"
                )
            except Exception as e:
                logger.error(f"[Batch {batch_job_id}] Failed to create batch job: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize batch job: {str(e)}"
                )

            # Step 3: Process files one at a time
            child_jobs = []
            total_size = 0

            for idx, metadata in enumerate(file_metadata):
                try:
                    contents = await metadata["file"].read()
                    file_size = len(contents)

                    # Validate file size
                    is_valid, error_msg = self.validate_file(metadata["file"], contents)
                    if not is_valid:
                        raise ValueError(f"File validation failed: {error_msg}")

                    total_size += file_size

                    self.job_store.create_job(metadata["job_id"], {
                        "job_id": metadata["job_id"],
                        "job_type": "video",
                        "parent_batch_id": batch_job_id,
                        "filename": metadata["filename"],
                        "status": "processing",
                        "size_bytes": file_size,
                        "content_type": metadata["content_type"],
                        "namespace": namespace
                    })

                    self.process_video.spawn(
                        contents,
                        metadata["filename"],
                        metadata["job_id"],
                        namespace,
                        batch_job_id
                    )

                    successfully_spawned.append(metadata["job_id"])

                    child_jobs.append({
                        "job_id": metadata["job_id"],
                        "filename": metadata["filename"],
                        "size_bytes": file_size
                    })

                except Exception as e:
                    logger.error(
                        f"[Batch {batch_job_id}] Failed to process file {metadata['filename']} "
                        f"(#{idx + 1}/{len(file_metadata)}): {e}"
                    )

                    # Mark job as failed and update parent batch
                    try:
                        self.job_store.set_job_failed(
                            metadata["job_id"],
                            f"Upload failed: {str(e)}"
                        )
                        self.job_store.update_batch_on_child_completion(
                            batch_job_id,
                            metadata["job_id"],
                            {
                                "job_id": metadata["job_id"],
                                "status": "failed",
                                "filename": metadata["filename"],
                                "error": f"Upload failed: {str(e)}"
                            }
                        )
                    except Exception as update_error:
                        logger.error(
                            f"[Batch {batch_job_id}] Failed to update job status: {update_error}"
                        )

            # Check if any jobs succeeded
            if len(successfully_spawned) == 0:
                logger.error(f"[Batch {batch_job_id}] All {len(files)} jobs failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"All {len(files)} videos failed to process. Check server logs."
                )

            logger.info(
                f"[Batch {batch_job_id}] Spawned {len(successfully_spawned)}/{len(files)} videos, "
                f"total size: {total_size / 1024 / 1024:.2f} MB"
            )

            return {
                "batch_job_id": batch_job_id,
                "status": "processing",
                "total_videos": len(files),
                "successfully_spawned": len(successfully_spawned),
                "failed_at_upload": len(files) - len(successfully_spawned),
                "child_jobs": child_jobs,
                "message": "Batch upload complete, videos processing in background"
            }

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"[Batch {batch_job_id}] Batch upload failed: {e}")

            # Clean up: delete batch job if it was created
            if batch_created:
                try:
                    self.job_store.delete_job(batch_job_id)
                    logger.info(f"[Batch {batch_job_id}] Cleaned up batch job entry")
                except Exception as cleanup_error:
                    logger.error(f"[Batch {batch_job_id}] Failed to cleanup batch job: {cleanup_error}")

            raise HTTPException(
                status_code=500,
                detail=f"Batch upload failed: {str(e)}"
            )

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
