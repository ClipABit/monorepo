import logging
from fastapi import FastAPI
import modal

logger = logging.getLogger(__name__)

@modal.functions.cls(
    secret=modal.Secret.from_name("dev"),
    image=modal.Image.debian_slim(python_version="3.12")
)
class BackgroundTasks:
    @modal.enter()
    def startup(self):
        """
        Startup logic for background tasks. This runs once when the container starts.
        """
        import os
        from preprocessing.preprocessor import Preprocessor
        from embeddings.embedder import VideoEmbedder
        from database.pinecone_connector import PineconeConnector
        from database.job_store_connector import JobStoreConnector
        from database.r2_connector import R2Connector

        logger.info("Container for background tasks starting up!")

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
        ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

        pinecone_index = f"{ENVIRONMENT}-chunks"

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
        logger.info("Background task modules initialized and ready!")

    @modal.method()
    async def process_video_background(self, video_bytes: bytes, filename: str, job_id: str, namespace: str = ""):
        logger.info(f"[Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes) | namespace='{namespace}'")
        
        hashed_identifier = None
        upserted_chunk_ids = []

        try:
            success, hashed_identifier = self.r2_connector.upload_video(
                video_data=video_bytes,
                filename=filename,
                namespace=namespace
            )
            if not success:
                upload_error_details = hashed_identifier
                hashed_identifier = None
                raise Exception(f"Failed to upload video to R2 storage: {upload_error_details}")

            processed_chunks = self.preprocessor.process_video_from_bytes(
                video_bytes=video_bytes,
                video_id=job_id,
                filename=filename,
                hashed_identifier=hashed_identifier
            )
            
            total_frames = sum(chunk['metadata']['frame_count'] for chunk in processed_chunks)
            total_memory = sum(chunk['memory_mb'] for chunk in processed_chunks)
            avg_complexity = sum(chunk['metadata']['complexity_score'] for chunk in processed_chunks) / len(processed_chunks) if processed_chunks else 0

            logger.info(f"[Job {job_id}] Complete: {len(processed_chunks)} chunks, {total_frames} frames, {total_memory:.2f} MB, avg_complexity={avg_complexity:.3f}")
            
            logger.info(f"[Job {job_id}] Embedding and upserting {len(processed_chunks)} chunks")

            chunk_details = []
            for chunk in processed_chunks:
                embedding = self.video_embedder._generate_clip_embedding(chunk["frames"], num_frames=8)
               
                logger.info(f"[Job {job_id}] Generated CLIP embedding for chunk {chunk['chunk_id']}")
                logger.info(f"[Job {job_id}] Upserting embedding for chunk {chunk['chunk_id']} to Pinecone...")
              
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
            self.job_store.set_job_completed(job_id, result)
            return result

        except Exception as e:
            logger.error(f"[Job {job_id}] Processing failed: {e}")
            logger.warning(f"[Job {job_id}] Initiating rollback due to failure...")
            
            if hashed_identifier:
                logger.info(f"[Job {job_id}] Rolling back: Deleting video from R2 ({hashed_identifier})")
                success = self.r2_connector.delete_video(hashed_identifier)
                if not success:
                    logger.error(f"[Job {job_id}] Rollback failed for R2 video deletion: {hashed_identifier}")
            
            if upserted_chunk_ids:
                logger.info(f"[Job {job_id}] Rolling back: Deleting {len(upserted_chunk_ids)} chunks from Pinecone")
                success = self.pinecone_connector.delete_chunks(upserted_chunk_ids, namespace=namespace)
                if not success:
                    logger.error(f"[Job {job_id}] Rollback failed for Pinecone chunks deletion: {len(upserted_chunk_ids)} chunks")
            
            logger.info(f"[Job {job_id}] Rollback complete.")

            import traceback
            traceback.print_exc()

            self.job_store.set_job_failed(job_id, str(e))
            return {"job_id": job_id, "status": "failed", "error": str(e)}

    @modal.method()
    async def delete_video_background(self, job_id: str, hashed_identifier: str, namespace: str = ""):
        """Background method to delete a video and all associated chunks from R2 and Pinecone."""
        logger.info(f"[Job {job_id}] Deletion started: {hashed_identifier} | namespace='{namespace}'")

        try:
            pinecone_success = self.pinecone_connector.delete_by_identifier(
                hashed_identifier=hashed_identifier,
                namespace=namespace
            )

            if not pinecone_success:
                raise Exception("Failed to delete chunks from Pinecone")
            
            r2_success = self.r2_connector.delete_video(hashed_identifier)
            if not r2_success:
                logger.critical(f"[Job {job_id}] INCONSISTENCY: Chunks deleted but R2 deletion failed for {hashed_identifier}")
                raise Exception("Failed to delete video from R2 after deleting chunks. System may be inconsistent.")

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
            self.job_store.set_job_completed(job_id, result)
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Job {job_id}] Deletion failed: {error_msg}")

            import traceback
            traceback.print_exc()

            self.job_store.set_job_failed(job_id, error_msg)
            return {"job_id": job_id, "status": "failed", "error": error_msg}
