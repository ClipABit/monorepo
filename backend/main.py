import os
import logging
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
                "api",
                "preprocessing",
                "embeddings",
                "models",
                "database",
                "search",
                "services",
            )
        )

# Environment: "dev" (default) or "prod" (set via ENVIRONMENT variable)
env = os.environ.get("ENVIRONMENT", "dev")
if env not in ["dev", "prod", "staging"]:
    raise ValueError(f"Invalid ENVIRONMENT value: {env}. Must be one of: dev, prod, staging")
logger.info(f"Starting Modal app in '{env}' environment")

IS_INTERNAL_ENV = env in ["dev", "staging"]

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
        from api import FastAPIRouter
        from fastapi import FastAPI
        from services.upload import UploadHandler

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

        self.upload_handler = UploadHandler(
            job_store=self.job_store,
            process_video_method=self.process_video
        )

        #FastAPI app
        self.fastapi_app = FastAPI()
        self.api = FastAPIRouter(self, IS_INTERNAL_ENV)
        self.fastapi_app.include_router(self.api.router)

        logger.info("Container modules initialized and ready!")

        print(f"[Container] Started at {self.start_time.isoformat()}")

   
    @modal.asgi_app()
    def asgi_app(self):
        return self.fastapi_app

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
                update_success = self.job_store.update_batch_on_child_completion(
                    parent_batch_id,
                    job_id,
                    result
                )
                if update_success:
                    logger.info(f"[Job {job_id}] Updated parent batch {parent_batch_id}")
                else:
                    logger.error(
                        f"[Job {job_id}] CRITICAL: Failed to update parent batch {parent_batch_id} "
                        f"after max retries. Batch state may be inconsistent."
                    )

            # Invalidate cached pages for namespace after successful processing
            try:
                self.r2_connector.clear_cache(namespace or "__default__")
            except Exception as cache_exc:
                logger.error(f"[Job {job_id}] Failed to clear cache for namespace {namespace}: {cache_exc}")

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
                update_success = self.job_store.update_batch_on_child_completion(
                    parent_batch_id,
                    job_id,
                    error_result
                )
                if update_success:
                    logger.info(f"[Job {job_id}] Updated parent batch {parent_batch_id} with failure status")
                else:
                    logger.error(
                        f"[Job {job_id}] CRITICAL: Failed to update parent batch {parent_batch_id} "
                        f"after max retries. Batch state may be inconsistent."
                    )

            return {"job_id": job_id, "status": "failed", "error": str(e)}


    @modal.method()
    async def delete_video_background(self, job_id: str, hashed_identifier: str, namespace: str = ""):
        """
        Background job that deletes a video and all associated chunks from R2 and Pinecone.

        This method is intended to run asynchronously as part of a job lifecycle. It:

        1. Attempts to delete all chunks in Pinecone associated with ``hashed_identifier`` and
           the given ``namespace`` using ``pinecone_connector.delete_by_identifier``.
        2. If Pinecone deletion is successful, attempts to delete the corresponding video
           object from R2 via ``r2_connector.delete_video``.
        3. On full success (both deletions succeed), builds a result payload and records
           the job as completed in ``self.job_store`` by calling
           ``self.job_store.set_job_completed(job_id, result)``.
        4. On any failure (including partial failures where Pinecone succeeds but R2 fails,
           or Pinecone deletion itself fails), logs the error, records the job as failed in
           ``self.job_store`` via ``self.job_store.set_job_failed(job_id, error_message)``,
           and returns a failure payload.

        The return value is the same object stored in ``job_store`` and has the following
        general shape:

        - On success::

            {
                "job_id": "<job id>",
                "status": "completed",
                "hashed_identifier": "<hashed id>",
                "namespace": "<namespace>",
                "r2": {"deleted": true},
                "pinecone": {"deleted": true}
            }

        - On failure (including partial deletion failures)::

            {
                "job_id": "<job id>",
                "status": "failed",
                "error": "<human-readable error message>"
            }

        In particular, if Pinecone deletion succeeds but R2 deletion fails, the method logs
        a critical inconsistency, raises an exception internally, and ultimately marks the
        job as failed in ``job_store`` with an appropriate error message, indicating that
        chunks may have been removed while the video object remains in R2.
        """
        logger.info(f"[Job {job_id}] Deletion started: {hashed_identifier} | namespace='{namespace}'")

        try:
            # Delete chunks from Pinecone
            pinecone_success = self.pinecone_connector.delete_by_identifier(
                hashed_identifier=hashed_identifier,
                namespace=namespace
            )

            # NOTE: idk if we acc need to raise exception here because this isn't a critical failure
            if not pinecone_success:
                raise Exception("Failed to delete chunks from Pinecone")
            
            # Delete from R2. If this fails, chunks are gone but video remains - notify client.
            r2_success = self.r2_connector.delete_video(hashed_identifier)
            if not r2_success:
                logger.critical(f"[Job {job_id}] INCONSISTENCY: Chunks deleted but R2 deletion failed for {hashed_identifier}")
                raise Exception("Failed to delete video from R2 after deleting chunks. System may be inconsistent.")

            # Build success response
            result = {
                "job_id": job_id,
                "status": "completed",
                "hashed_identifier": hashed_identifier,
                "namespace": namespace,
                "r2": {
                    "deleted": r2_success
                },
                "pinecone": {
                    "deleted": pinecone_success
                }
            }

            logger.info(f"[Job {job_id}] Deletion completed: R2={r2_success}, Pinecone chunks={pinecone_success}")

            # Store result for polling endpoint
            self.job_store.set_job_completed(job_id, result)

            try:
                self.r2_connector.clear_cache(namespace or "__default__")
            except Exception as cache_exc:
                logger.error(
                    f"[Job {job_id}] Failed to clear cache after deletion for namespace {namespace}: {cache_exc}"
                )
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Job {job_id}] Deletion failed: {error_msg}")

            import traceback
            traceback.print_exc()

            # Store error result
            self.job_store.set_job_failed(job_id, error_msg)
            return {"job_id": job_id, "status": "failed", "error": error_msg}