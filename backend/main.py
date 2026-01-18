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
                "face_recognition",
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


@app.cls(cpu=4.0, timeout=600)
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
        from face_recognition import FaceDetector, FaceRepository


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
        if ENVIRONMENT not in ["dev", "prod"]:
            raise ValueError(f"Invalid ENVIRONMENT value: {ENVIRONMENT}. Must be one of: dev, prod")
        logger.info(f"Running in environment: {ENVIRONMENT}")

        # Select Pinecone index based on environment
        pinecone_index = f"{ENVIRONMENT}-chunks"    # video chunks index
        logger.info(f"Using Pinecone index: {pinecone_index}")
        pinecone_face_index = f"{ENVIRONMENT}-faces"  # face embeddings index
        logger.info(f"Using Pinecone face index: {pinecone_face_index}")

        # Instantiate classes
        self.preprocessor = Preprocessor(min_chunk_duration=1.0, max_chunk_duration=10.0, scene_threshold=13.0)
        self.video_embedder = VideoEmbedder()
        self.pinecone_connector = PineconeConnector(api_key=PINECONE_API_KEY, index_name=pinecone_index)
        self.face_recognition_pinecone_connector = PineconeConnector(api_key=PINECONE_API_KEY, index_name=pinecone_face_index)
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

        self.face_detector = FaceDetector()
        self.face_repository = FaceRepository(self.face_recognition_pinecone_connector, self.r2_connector, threshold=0.35)

        logger.info("Container modules initialized and ready!")

        print(f"[Container] Started at {self.start_time.isoformat()}")

    @modal.method()
    async def process_video(self, video_bytes: bytes, filename: str, job_id: str, namespace: str = ""):
        logger.info(f"[Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes) | namespace='{namespace}'")
        
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

                # Face recognition for this chunk
                try:
                    from face_recognition.frame_face_pipeline import FrameFacePipeline

                    pipeline = FrameFacePipeline(namespace=namespace, face_detector=self.face_detector, face_repository=self.face_repository)

                    # sample up to 8 frames evenly from the chunk for face processing
                    frames = chunk.get('frames')
                    sampled = []
                    if frames is not None and len(frames) > 0:
                        n = min(8, len(frames))
                        if len(frames) <= n:
                            sampled = list(frames)
                        else:
                            import numpy as _np
                            idx = _np.linspace(0, len(frames) - 1, n).astype(int)
                            sampled = [frames[i] for i in idx]

                    # Aggregate face mappings for the chunk (face_id -> img_access_id)
                    face_map = {}
                    for f in sampled:
                        try:
                            mapping = pipeline.process_frame(f, chunk_id=chunk['chunk_id'])
                            # mapping may contain multiple faces; merge into face_map
                            for k, v in mapping.items():
                                face_map.setdefault(k, v)
                        except Exception as fe:
                            logger.error(f"Face processing failed for chunk {chunk['chunk_id']}: {fe}")
                except Exception as e:
                    logger.error(f"Failed to run face recognition for chunk {chunk['chunk_id']}: {e}")
                    face_map = {}

                chunk_details.append({
                    "chunk_id": chunk['chunk_id'],
                    "metadata": chunk['metadata'],
                    "memory_mb": chunk['memory_mb'],
                    "faces": face_map,
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
            
            # 3. Delete face embeddings from Pinecone
            # TODO

            # 4. Delete face images from R2
            # TODO


            logger.info(f"[Job {job_id}] Rollback complete.")
            # ----------------------

            import traceback
            traceback.print_exc()  # Print full stack trace for debugging

            # Store error result for polling endpoint in shared storage
            self.job_store.set_job_failed(job_id, str(e))
            return {"job_id": job_id, "status": "failed", "error": str(e)}

    @modal.fastapi_endpoint(method="GET")
    async def status(self, job_id: str):
        """
        Check the status of a video processing job.

        Args:
            job_id (str): The unique identifier for the video processing job.

        Returns:
            dict: Contains:
                - job_id (str): The job identifier
                - status (str): 'processing', 'completed', or 'failed'
                - message (str, optional): If still processing or not found
                - result (dict, optional): Full job result if completed

        This endpoint allows clients (e.g., frontend) to poll for job progress and retrieve results when ready.
        """
        job_data = self.job_store.get_job(job_id)

        if job_data is None:
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Job is still processing or not found"
            }

        return job_data

    @modal.fastapi_endpoint(method="POST")
    async def upload(self, file: UploadFile = None, namespace: str = Form("")):
        """
        Handle video file upload and start background processing.

        Args:
            file (UploadFile): The uploaded video file.
            namespace (str, optional): Namespace for Pinecone and R2 storage (default: "")

        Returns:
            dict: Contains job_id, filename, content_type, size_bytes, status, and message.
        """
        # TODO: Add error handling for file types and sizes
        import uuid
        job_id = str(uuid.uuid4())
        contents = await file.read()
        file_size = len(contents)
        self.job_store.create_job(job_id, {
            "job_id": job_id,
            "filename": file.filename,
            "status": "processing",
            "size_bytes": file_size,
            "content_type": file.content_type,
            "namespace": namespace
        })

        # Spawn background processing (non-blocking - returns immediately)
        self.process_video.spawn(contents, file.filename, job_id, namespace)

        return {
            "job_id": job_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "status": "processing",
            "message": "Video uploaded successfully, processing in background"
        }

    
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