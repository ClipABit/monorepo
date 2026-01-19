"""
Processing Modal App

Handles video processing with full CLIP image encoder and preprocessing pipeline.
Heavy dependencies (~15-20s cold start) - acceptable for background jobs.

This app is spawned by the Server for video uploads.
"""

import logging
import modal

# Import shared config (also configures logging to stdout)
from shared.config import (
    get_environment,
    get_secrets,
    get_pinecone_index,
    get_env_var,
)
from shared.images import get_processing_image

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()
logger.info(f"Starting Processing App in '{env}' environment")

# Create Modal app with processing-specific image
app = modal.App(
    name=f"{env} processing",
    image=get_processing_image(),
    secrets=[get_secrets()]
)


@app.cls(cpu=4.0, memory=4096, timeout=600)
class ProcessingWorker:
    """
    Processing worker class.
    
    Loads full CLIP model and preprocessing pipeline on startup.
    Handles video processing including:
    - Video upload to R2
    - Scene-based chunking
    - Frame extraction and compression
    - CLIP embedding generation
    - Upsert to Pinecone
    """

    @modal.enter()
    def startup(self):
        """
        Load CLIP model and initialize all connectors.
        
        The models are loaded eagerly to avoid latency on first request.
        """
        from preprocessing.preprocessor import Preprocessor
        from embeddings.embedder import VideoEmbedder
        from database.pinecone_connector import PineconeConnector
        from database.job_store_connector import JobStoreConnector
        from database.r2_connector import R2Connector

        logger.info(f"Processing worker starting up! Environment = {env}")

        # Get environment variables
        PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
        R2_ACCOUNT_ID = get_env_var("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = get_env_var("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = get_env_var("R2_SECRET_ACCESS_KEY")
        ENVIRONMENT = get_environment()

        pinecone_index = get_pinecone_index()
        logger.info(f"Using Pinecone index: {pinecone_index}")

        # Initialize preprocessor and embedder
        self.preprocessor = Preprocessor(
            min_chunk_duration=1.0,
            max_chunk_duration=10.0,
            scene_threshold=13.0
        )
        self.video_embedder = VideoEmbedder()
        logger.info("CLIP image encoder and preprocessor loaded")

        # Initialize connectors
        self.pinecone_connector = PineconeConnector(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index
        )
        self.job_store = JobStoreConnector(dict_name="clipabit-jobs")
        self.r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=ENVIRONMENT
        )

        logger.info("Processing worker initialized and ready!")

    @modal.method()
    def process_video_background(
        self,
        video_bytes: bytes,
        filename: str,
        job_id: str,
        namespace: str = "",
        parent_batch_id: str = None
    ):
        """
        Process an uploaded video through the full pipeline.
        
        Pipeline stages:
        1. Upload original video to R2
        2. Preprocess video (chunk, extract frames, compress)
        3. Generate CLIP embeddings for each chunk
        4. Upsert embeddings to Pinecone
        5. Update job store with results
        
        Args:
            video_bytes: Raw video file bytes
            filename: Original filename
            job_id: Unique job identifier for tracking
            namespace: Pinecone/R2 namespace for organization
            parent_batch_id: Optional batch ID for batch processing
            
        Returns:
            dict: Processing result with status, chunk details, and statistics
        """
        logger.info(
            f"[Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes) "
            f"| namespace='{namespace}' | batch={parent_batch_id or 'None'}"
        )

        hashed_identifier = None
        upserted_chunk_ids = []

        try:
            # Stage 1: Upload original video to R2 bucket
            success, hashed_identifier = self.r2_connector.upload_video(
                video_data=video_bytes,
                filename=filename,
                namespace=namespace
            )
            if not success:
                # Capture error details returned in hashed_identifier before resetting it
                upload_error_details = hashed_identifier
                hashed_identifier = None
                raise Exception(f"Failed to upload video to R2 storage: {upload_error_details}")

            # Stage 2: Process video through preprocessing pipeline
            processed_chunks = self.preprocessor.process_video_from_bytes(
                video_bytes=video_bytes,
                video_id=job_id,
                filename=filename,
                hashed_identifier=hashed_identifier
            )

            # Calculate summary statistics
            total_frames = sum(chunk['metadata']['frame_count'] for chunk in processed_chunks)
            total_memory = sum(chunk['memory_mb'] for chunk in processed_chunks)
            avg_complexity = (
                sum(chunk['metadata']['complexity_score'] for chunk in processed_chunks)
                / len(processed_chunks)
                if processed_chunks else 0
            )

            logger.info(
                f"[Job {job_id}] Complete: {len(processed_chunks)} chunks, "
                f"{total_frames} frames, {total_memory:.2f} MB, avg_complexity={avg_complexity:.3f}"
            )

            # Stage 3-4: Embed frames and store in Pinecone
            logger.info(f"[Job {job_id}] Embedding and upserting {len(processed_chunks)} chunks")

            chunk_details = []
            for chunk in processed_chunks:
                embedding = self.video_embedder._generate_clip_embedding(
                    chunk["frames"],
                    num_frames=8
                )

                logger.info(f"[Job {job_id}] Generated CLIP embedding for chunk {chunk['chunk_id']}")
                logger.info(f"[Job {job_id}] Upserting embedding for chunk {chunk['chunk_id']} to Pinecone...")

                # Transform metadata for Pinecone compatibility
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

                # 3. Remove null values (Pinecone rejects keys with null values)
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

            # Stage 5: Store result for polling endpoint in shared storage
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
                    logger.error(
                        f"[Job {job_id}] Rollback failed for Pinecone chunks deletion: "
                        f"{len(upserted_chunk_ids)} chunks"
                    )

            logger.info(f"[Job {job_id}] Rollback complete.")
            # ----------------------

            import traceback
            traceback.print_exc()

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
