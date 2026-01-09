import os
import logging
from fastapi import UploadFile, HTTPException, Form
import modal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Modal app and image
# dependencies found in pyproject.toml
image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("ffmpeg", "libsm6", "libxext6") # for video processing
            .uv_sync(extra_options="--no-dev")  # exclude dev dependencies to avoid package conflicts
            .add_local_python_source(  # add all local modules here
                "preprocessing",
                "embeddings",
                "models",
                "database",
                "search",
            )
        )

# Environment: "dev" (default) or "prod" (set via ENVIRONMENT variable)
env = os.environ.get("ENVIRONMENT", "dev")

# Create Modal app
app = modal.App(
    name=env,
    image=image,
    secrets=[modal.Secret.from_name(env)]
)


@app.cls(cpu=4.0, memory=4096, timeout=600)
class Server:

    @modal.enter()
    def startup(self):
        """
            Startup logic. This runs once when the container starts.
            Here is where you would instantiate classes and load models that are
            reused across multiple requests to avoid reloading them each time.
        """

        # Import local module inside class
        import os
        from datetime import datetime, timezone

        # Import classes here
        from preprocessing.preprocessor import Preprocessor
        from embeddings.embedder import VideoEmbedder
        from database.pinecone_connector import PineconeConnector
        from database.job_store_connector import JobStoreConnector
        from search.searcher import Searcher
        from database.r2_connector import R2Connector


        logger.info(f"Container starting up! Environment = {env}")
        self.start_time = datetime.now(timezone.utc)

        # Get environment variables (TODO: abstract to config module)
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
        if not R2_ACCOUNT_ID:
            raise ValueError("R2_ACCOUNT_ID not found in environment variables")

        R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
        if not R2_ACCESS_KEY_ID:
            raise ValueError("R2_ACCESS_KEY_ID not found in environment variables")

        R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
        if not R2_SECRET_ACCESS_KEY:
            raise ValueError("R2_SECRET_ACCESS_KEY not found in environment variables")

        ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
        if ENVIRONMENT not in ["dev", "prod", "staging"]:
            raise ValueError(f"Invalid ENVIRONMENT value: {ENVIRONMENT}. Must be one of: dev, prod, staging")
        logger.info(f"Running in environment: {ENVIRONMENT}")

        # Select Pinecone index based on environment
        pinecone_index = f"{ENVIRONMENT}-chunks"
        logger.info(f"Using Pinecone index: {pinecone_index}")

        # Instantiate classes
        self.preprocessor = Preprocessor(min_chunk_duration=1.0, max_chunk_duration=10.0, scene_threshold=13.0)
        self.video_embedder = VideoEmbedder()
        self.pinecone_connector = PineconeConnector(api_key=PINECONE_API_KEY, index_name=pinecone_index)
        self.job_store = JobStoreConnector(dict_name="clipabit-jobs")

        self.r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=ENVIRONMENT
        )

        self.searcher = Searcher(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index,
            r2_connector=self.r2_connector
        )

        logger.info("Container modules initialized and ready!")

        print(f"[Container] Started at {self.start_time.isoformat()}")

    @modal.method()
    async def process_video(self, video_bytes: bytes, filename: str, job_id: str, namespace: str = "", parent_batch_id: str = None):
        logger.info(f"[Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes) | namespace='{namespace}' | batch={parent_batch_id or 'None'}")

        hashed_identifier = None
        upserted_chunk_ids = []

        try:
            # Upload original video to R2 bucket
            # TODO: do this in parallel with processing
            success, hashed_identifier = self.r2_connector.upload_video(
                video_data=video_bytes,
                filename=filename,
                namespace=namespace
            )
            if not success:
                # Capture error details returned in hashed_identifier before resetting it
                upload_error_details = hashed_identifier
                # Reset hashed_identifier if upload failed to avoid rollback attempting to delete it
                hashed_identifier = None
                raise Exception(f"Failed to upload video to R2 storage: {upload_error_details}")

            # Process video through preprocessing pipeline
            processed_chunks = self.preprocessor.process_video_from_bytes(
                video_bytes=video_bytes,
                video_id=job_id,
                filename=filename,
                hashed_identifier=hashed_identifier
            )

            # Calculate summary statistics
            total_frames = sum(chunk['metadata']['frame_count'] for chunk in processed_chunks)
            total_memory = sum(chunk['memory_mb'] for chunk in processed_chunks)
            avg_complexity = sum(chunk['metadata']['complexity_score'] for chunk in processed_chunks) / len(processed_chunks) if processed_chunks else 0

            logger.info(f"[Job {job_id}] Complete: {len(processed_chunks)} chunks, {total_frames} frames, {total_memory:.2f} MB, avg_complexity={avg_complexity:.3f}")

            # Embed frames and store in Pinecone
            logger.info(f"[Job {job_id}] Embedding and upserting {len(processed_chunks)} chunks")

            # Prepare chunk details for response (without frame arrays)
            chunk_details = []
            for chunk in processed_chunks:
                embedding = self.video_embedder._generate_clip_embedding(chunk["frames"], num_frames=8)

                logger.info(f"[Job {job_id}] Generated CLIP embedding for chunk {chunk['chunk_id']}")
                logger.info(f"[Job {job_id}] Upserting embedding for chunk {chunk['chunk_id']} to Pinecone...")


                # 1. Handle timestamp_range (List of Numbers -> Two Numbers)
                if 'timestamp_range' in chunk['metadata']:
                    start_time, end_time = chunk['metadata'].pop('timestamp_range')
                    chunk['metadata']['start_time_s'] = start_time
                    chunk['metadata']['end_time_s'] = end_time

                # 2. Handle file_info (Nested Dict -> Flat Keys)
                if 'file_info' in chunk['metadata']:
                    file_info = chunk['metadata'].pop('file_info')
                    for key, value in file_info.items():
                        chunk['metadata'][f'file_{key}'] = value

                # 3. Final Check: Remove Nulls (Optional but good practice)
                # Pinecone rejects keys with null values.
                keys_to_delete = [k for k, v in chunk['metadata'].items() if v is None]
                for k in keys_to_delete:
                    del chunk['metadata'][k]


                success = self.pinecone_connector.upsert_chunk(
                    chunk_id=chunk['chunk_id'],
                    chunk_embedding=embedding.numpy(),
                    namespace=namespace,
                    metadata=chunk['metadata']
                )

                if success:
                    upserted_chunk_ids.append(chunk['chunk_id'])
                else:
                    raise Exception(f"Failed to upsert chunk {chunk['chunk_id']} to Pinecone")

                chunk_details.append({
                    "chunk_id": chunk['chunk_id'],
                    "metadata": chunk['metadata'],
                    "memory_mb": chunk['memory_mb'],
                })

            result = {
                "job_id": job_id,
                "status": "completed",
                "hashed_identifier": hashed_identifier,
                "filename": filename,
                "chunks": len(processed_chunks),
                "total_frames": total_frames,
                "total_memory_mb": total_memory,
                "avg_complexity": avg_complexity,
                "chunk_details": chunk_details,
            }

            logger.info(f"[Job {job_id}] Finished processing {filename}")

            # Store result for polling endpoint in shared storage
            self.job_store.set_job_completed(job_id, result)

            # Update parent batch if exists
            if parent_batch_id:
                self.job_store.update_batch_on_child_completion(
                    parent_batch_id,
                    job_id,
                    result
                )
                logger.info(f"[Job {job_id}] Updated parent batch {parent_batch_id}")

            return result

        except Exception as e:
            logger.error(f"[Job {job_id}] Processing failed: {e}")

            # --- ROLLBACK LOGIC ---
            logger.warning(f"[Job {job_id}] Initiating rollback due to failure...")

            # 1. Delete video from R2
            if hashed_identifier:
                logger.info(f"[Job {job_id}] Rolling back: Deleting video from R2 ({hashed_identifier})")
                success = self.r2_connector.delete_video(hashed_identifier)
                if not success:
                    logger.error(f"[Job {job_id}] Rollback failed for R2 video deletion: {hashed_identifier}")

            # 2. Delete chunks from Pinecone
            if upserted_chunk_ids:
                logger.info(f"[Job {job_id}] Rolling back: Deleting {len(upserted_chunk_ids)} chunks from Pinecone")
                success = self.pinecone_connector.delete_chunks(upserted_chunk_ids, namespace=namespace)
                if not success:
                    logger.error(f"[Job {job_id}] Rollback failed for Pinecone chunks deletion: {len(upserted_chunk_ids)} chunks")

            logger.info(f"[Job {job_id}] Rollback complete.")
            # ----------------------

            import traceback
            traceback.print_exc()  # Print full stack trace for debugging

            # Store error result for polling endpoint in shared storage
            self.job_store.set_job_failed(job_id, str(e))

            # Update parent batch if exists
            if parent_batch_id:
                error_result = {
                    "job_id": job_id,
                    "status": "failed",
                    "filename": filename,
                    "error": str(e)
                }
                self.job_store.update_batch_on_child_completion(
                    parent_batch_id,
                    job_id,
                    error_result
                )
                logger.info(f"[Job {job_id}] Updated parent batch {parent_batch_id} with failure status")

            return {"job_id": job_id, "status": "failed", "error": str(e)}

    @modal.fastapi_endpoint(method="GET")
    async def status(self, job_id: str, include_children: bool = False):
        """
        Check the status of a video processing job or batch job.

        Args:
            job_id (str): The unique identifier for the video processing job or batch job.
            include_children (bool): If True and job is a batch, include full child job details.

        Returns:
            dict: For video jobs: job_id, status, and result data
                  For batch jobs: batch_job_id, status, progress, metrics, etc.

        This endpoint allows clients (e.g., frontend) to poll for job progress and retrieve results when ready.
        """
        job_data = self.job_store.get_job(job_id)

        if job_data is None:
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Job is still processing or not found"
            }

        job_type = job_data.get("job_type", "video")

        # Individual video job - return as-is
        if job_type == "video":
            return job_data

        # Batch job - return batch-specific format
        elif job_type == "batch":
            # Calculate progress percentage, handling empty batch case
            total_videos = job_data["total_videos"]
            if total_videos > 0:
                progress_percent = (
                    (job_data["completed_count"] + job_data["failed_count"])
                    / total_videos * 100
                )
            else:
                progress_percent = 0.0

            response = {
                "batch_job_id": job_data["batch_job_id"],
                "status": job_data["status"],
                "total_videos": total_videos,
                "completed_count": job_data["completed_count"],
                "failed_count": job_data["failed_count"],
                "processing_count": job_data["processing_count"],
                "progress_percent": progress_percent,
                "namespace": job_data["namespace"],
                "created_at": job_data["created_at"],
                "updated_at": job_data["updated_at"]
            }

            # Include aggregated metrics if available
            if job_data["completed_count"] > 0:
                response["metrics"] = {
                    "total_chunks": job_data["total_chunks"],
                    "total_frames": job_data["total_frames"],
                    "total_memory_mb": job_data["total_memory_mb"],
                    "avg_complexity": job_data["avg_complexity"]
                }

            # Include failed job summaries if any
            if job_data["failed_count"] > 0:
                response["failed_jobs"] = job_data["failed_jobs"]

            # Include child details if requested
            if include_children:
                response["child_jobs"] = self.job_store.get_batch_child_jobs(job_id)
            else:
                # Just include job IDs for reference
                response["child_job_ids"] = job_data["child_jobs"]

            return response

        else:
            return {
                "job_id": job_id,
                "error": f"Unknown job type: {job_type}"
            }

    @modal.fastapi_endpoint(method="POST")
    async def upload(self, files: list[UploadFile] = None, namespace: str = Form("")):
        """
        Handle video file upload and start background processing.
        Supports both single and batch uploads.

        Args:
            files (list[UploadFile]): The uploaded video file(s). FastAPI wraps single file in list.
            namespace (str, optional): Namespace for Pinecone and R2 storage (default: "")

        Returns:
            dict: For single upload: job_id, filename, etc.
                  For batch upload: batch_job_id, total_videos, child_jobs, etc.
        """
        # Validation
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        MAX_BATCH_SIZE = 200
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size ({len(files)}) exceeds maximum ({MAX_BATCH_SIZE})"
            )

        # Single file upload (backward compatible)
        if len(files) == 1:
            return await self._handle_single_upload(files[0], namespace)

        # Batch upload
        else:
            return await self._handle_batch_upload(files, namespace)

    def _validate_file(self, file: UploadFile, file_contents: bytes = None) -> tuple[bool, str]:
        """
        Validate uploaded file for security and compatibility.

        Args:
            file: The UploadFile object to validate
            file_contents: Optional file contents for size validation (if already read)

        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
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

        # 1. Check filename exists
        if not file.filename:
            return False, "File has no filename"

        # 2. Check file extension
        filename_lower = file.filename.lower()
        file_ext = None
        for ext in ALLOWED_EXTENSIONS:
            if filename_lower.endswith(ext):
                file_ext = ext
                break

        if not file_ext:
            return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

        # 3. Check MIME type (if provided)
        if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
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
            if file_size > MAX_FILE_SIZE:
                return False, f"File too large ({file_size / 1024 / 1024:.1f} MB). Maximum: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB"

        # 5. Check for path traversal attempts in filename
        if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
            return False, "Filename contains invalid characters (path separators)"

        return True, ""

    async def _handle_single_upload(self, file: UploadFile, namespace: str) -> dict:
        """Handle single file upload (existing logic, backward compatible)."""
        import uuid

        # Validation before reading
        is_valid, error_msg = self._validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        job_id = str(uuid.uuid4())
        contents = await file.read()
        file_size = len(contents)

        # Validation after reading
        is_valid, error_msg = self._validate_file(file, contents)
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

    async def _handle_batch_upload(self, files: list[UploadFile], namespace: str) -> dict:
        """Handle batch file upload."""
        import uuid

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
                is_valid, error_msg = self._validate_file(file)
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
                    is_valid, error_msg = self._validate_file(metadata["file"], contents)
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
                    detail=f"Batch upload failed: All {len(files)} videos failed to process"
                )
            elif len(successfully_spawned) < len(files):
                logger.warning(
                    f"[Batch {batch_job_id}] Partial success: {len(successfully_spawned)}/{len(files)}"
                )

            logger.info(
                f"[Batch {batch_job_id}] Spawned {len(successfully_spawned)}/{len(files)} videos, "
                f"total size: {total_size / 1024 / 1024:.2f} MB"
            )

            return {
                "batch_job_id": batch_job_id,
                "total_videos": len(files),
                "successfully_spawned": len(successfully_spawned),
                "failed_count": len(files) - len(successfully_spawned),
                "total_size_bytes": total_size,
                "status": "processing" if len(successfully_spawned) == len(files) else "partial",
                "namespace": namespace,
                "child_jobs": child_jobs,
                "message": (
                    f"Batch upload started with {len(successfully_spawned)}/{len(files)} videos" +
                    (f" ({len(files) - len(successfully_spawned)} failed during upload)"
                     if len(successfully_spawned) < len(files) else "")
                )
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Batch {batch_job_id}] Unexpected error: {e}")

            # Mark batch as failed if it was created
            if batch_created:
                try:
                    batch_job = self.job_store.get_job(batch_job_id)
                    if batch_job:
                        batch_job["status"] = "failed"
                        batch_job["error"] = str(e)
                        self.job_store.update_job(batch_job_id, batch_job)
                except Exception as cleanup_error:
                    logger.error(f"[Batch {batch_job_id}] Failed cleanup: {cleanup_error}")

            raise HTTPException(
                status_code=500,
                detail=f"Batch upload failed unexpectedly: {str(e)}"
            )


    @modal.fastapi_endpoint(method="GET")
    async def search(self, query: str, namespace: str = ""):
        """
        Search endpoint - accepts a text query and returns semantic search results.

        Args:
        - query (str): The search query string (required)
        - namespace (str, optional): Namespace for Pinecone search (default: "")
        - top_k (int, optional): Number of top results to return (default: 10)

        Returns: dict with 'query', 'results', and 'timing'.
        """
        try:
            import time
            t_start = time.perf_counter()

            # Parse request
            if not query:
                raise HTTPException(status_code=400, detail="Missing 'query' parameter")

            top_k = 10
            logger.info(f"[Search] Query: '{query}' | namespace='{namespace}' | top_k={top_k}")

            # Execute semantic search
            results = self.searcher.search(
                query=query,
                top_k=top_k,
                namespace=namespace
            )

            t_done = time.perf_counter()

            # Log chunk-level results only
            logger.info(f"[Search] Found {len(results)} chunk-level results in {t_done - t_start:.3f}s")

            return {
                "query": query,
                "results": results,
                "timing": {
                    "total_s": round(t_done - t_start, 3)
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Search] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @modal.fastapi_endpoint(method="GET")
    async def list_videos(self, namespace: str = "__default__"):
        """
        List all videos for a specific namespace (namespace).
        Returns a list of video data objects containing filename, identifier, and presigned URL.
        """
        logger.info(f"[List Videos] Fetching videos for namespace: {namespace}")

        try:
            video_data = self.r2_connector.fetch_all_video_data(namespace)
            return {
                "status": "success",
                "namespace": namespace,
                "videos": video_data
            }
        except Exception as e:
            logger.error(f"[List Videos] Error fetching videos: {e}")
            raise HTTPException(status_code=500, detail=str(e))
