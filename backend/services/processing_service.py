"""
ProcessingService class - base class shared between processing_app.py and dev_combined.py
"""

import logging
import modal

from shared.config import get_environment, get_env_var, get_pinecone_index

logger = logging.getLogger(__name__)


class ProcessingService:
    """
    Processing service base class.
    
    Loads full CLIP model and preprocessing pipeline on startup.
    """

    @modal.enter()
    def startup(self):
        """Load CLIP model and initialize all connectors."""
        from preprocessing.preprocessor import Preprocessor
        from embeddings.video_embedder import VideoEmbedder
        from database.pinecone_connector import PineconeConnector
        from database.cache.job_store_connector import JobStoreConnector
        from database.firebase.user_store_connector import UserStoreConnector

        env = get_environment()
        logger.info(f"[{self.__class__.__name__}] Starting up in '{env}' environment")

        # Initialize Firebase Admin SDK (required for Firestore)
        import firebase_admin
        import json
        from firebase_admin import credentials, firestore
        firebase_credentials = json.loads(get_env_var("FIREBASE_ADMIN_KEY"))
        cred = credentials.Certificate(firebase_credentials)
        try:
            firebase_admin.initialize_app(cred)
        except ValueError:
            pass  # Already initialized
        firestore_client = firestore.client()

        # Get environment variables
        PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")

        pinecone_index = get_pinecone_index()
        logger.info(f"[{self.__class__.__name__}] Using Pinecone index: {pinecone_index}")

        # Initialize preprocessor and embedder
        self.preprocessor = Preprocessor(
            min_chunk_duration=1.0,
            max_chunk_duration=10.0,
            scene_threshold=13.0
        )
        self.video_embedder = VideoEmbedder()
        logger.info(f"[{self.__class__.__name__}] CLIP image encoder and preprocessor loaded")

        # Initialize connectors
        self.pinecone_connector = PineconeConnector(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index
        )
        self.job_store = JobStoreConnector(dict_name="clipabit-jobs")
        self.user_store = UserStoreConnector(firestore_client=firestore_client)

        logger.info(f"[{self.__class__.__name__}] Initialized and ready!")

    @modal.method()
    def process_video_background(
        self,
        video_bytes: bytes,
        filename: str,
        job_id: str,
        namespace: str = "",
        parent_batch_id: str = None,
        user_id: str = None,
        hashed_identifier: str = "",
        project_id: str = "",
    ):
        """
        Process an uploaded video through the full pipeline.

        Video files are not stored server-side — they live on the user's local machine.
        We only extract embeddings, store vectors in Pinecone, and track metadata.
        The hashed_identifier is generated client-side by the plugin and passed through.
        """
        logger.info(
            f"[{self.__class__.__name__}][Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes) "
            f"| namespace='{namespace}' | batch={parent_batch_id or 'None'} | hash={hashed_identifier or 'None'} | project={project_id or 'None'}"
        )

        upserted_chunk_ids = []

        try:
            # Stage 1: Process video through preprocessing pipeline
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
                f"[{self.__class__.__name__}][Job {job_id}] Complete: {len(processed_chunks)} chunks, "
                f"{total_frames} frames, {total_memory:.2f} MB, avg_complexity={avg_complexity:.3f}"
            )

            # Hard quota check before upserting any vectors
            # The upload endpoint does a soft pre-check, but concurrent uploads
            # could pass that check simultaneously. This is the hard gate.
            if user_id:
                is_ok, current_count, vector_quota = self.user_store.check_quota(user_id)
                total_new = len(processed_chunks)
                if not is_ok:
                    raise Exception(
                        f"Quota exceeded before upsert: {current_count}/{vector_quota} vectors. "
                        f"Aborting {total_new} chunks."
                    )
                if current_count + total_new > vector_quota:
                    raise Exception(
                        f"Upload would exceed quota: {current_count} + {total_new} = {current_count + total_new} > {vector_quota}. "
                        f"Aborting."
                    )

            # Stage 2: Embed frames and store in Pinecone
            logger.info(f"[{self.__class__.__name__}][Job {job_id}] Embedding and upserting {len(processed_chunks)} chunks")

            chunk_details = []
            for chunk in processed_chunks:
                embedding = self.video_embedder._generate_clip_embedding(
                    chunk["frames"],
                    num_frames=8
                )

                logger.info(f"[{self.__class__.__name__}][Job {job_id}] Generated CLIP embedding for chunk {chunk['chunk_id']}")

                # Transform metadata for Pinecone compatibility
                if 'timestamp_range' in chunk['metadata']:
                    start_time, end_time = chunk['metadata'].pop('timestamp_range')
                    chunk['metadata']['start_time_s'] = start_time
                    chunk['metadata']['end_time_s'] = end_time

                if 'file_info' in chunk['metadata']:
                    file_info = chunk['metadata'].pop('file_info')
                    for key, value in file_info.items():
                        chunk['metadata'][f'file_{key}'] = value

                keys_to_delete = [k for k, v in chunk['metadata'].items() if v is None]
                for k in keys_to_delete:
                    del chunk['metadata'][k]

                # Inject user and project identifiers for search filtering
                if user_id:
                    chunk['metadata']['user_id'] = user_id
                if project_id:
                    chunk['metadata']['project_id'] = project_id

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

            # Stage 3: Update vector quota tracking
            if user_id:
                chunk_count = len(upserted_chunk_ids)
                try:
                    self.user_store.increment_vector_count(user_id, chunk_count, namespace)
                    self.user_store.register_video(user_id, hashed_identifier, chunk_count, filename)
                    logger.info(
                        f"[{self.__class__.__name__}][Job {job_id}] Updated quota: +{chunk_count} vectors for user {user_id}"
                    )
                except Exception as quota_exc:
                    logger.critical(
                        f"[{self.__class__.__name__}][Job {job_id}] CRITICAL: Failed to update vector quota for user {user_id}: {quota_exc}. "
                        f"Vectors are in Pinecone but quota may be out of sync."
                    )

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

            logger.info(f"[{self.__class__.__name__}][Job {job_id}] Finished processing {filename}")

            # Stage 4: Store result
            self.job_store.set_job_completed(job_id, result)

            # Update parent batch if exists
            if parent_batch_id:
                update_success = self.job_store.update_batch_on_child_completion(
                    parent_batch_id,
                    job_id,
                    result
                )
                if update_success:
                    logger.info(f"[{self.__class__.__name__}][Job {job_id}] Updated parent batch {parent_batch_id}")
                else:
                    logger.error(
                        f"[{self.__class__.__name__}][Job {job_id}] CRITICAL: Failed to update parent batch {parent_batch_id}"
                    )

            return result

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}][Job {job_id}] Processing failed: {e}")

            # Rollback: delete any vectors already upserted to Pinecone
            if upserted_chunk_ids:
                logger.info(f"[{self.__class__.__name__}][Job {job_id}] Rolling back: Deleting chunks from Pinecone")
                self.pinecone_connector.delete_chunks(upserted_chunk_ids, namespace=namespace)

            import traceback
            traceback.print_exc()

            self.job_store.set_job_failed(job_id, str(e))

            if parent_batch_id:
                error_result = {
                    "job_id": job_id,
                    "status": "failed",
                    "filename": filename,
                    "error": str(e)
                }
                self.job_store.update_batch_on_child_completion(parent_batch_id, job_id, error_result)

            return {"job_id": job_id, "status": "failed", "error": str(e)}
