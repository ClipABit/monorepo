import os
import logging
from fastapi import UploadFile, HTTPException
import modal

# Constants
PINECONE_CHUNKS_INDEX = "chunks-index"

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
            .uv_sync(extra_options="--no-dev")  # exclude dev dependencies to avoid package conflicts
            .add_local_python_source(  # add all local modules here
                "preprocessing",
                "embeddings",
                "models",
                "database",
                "search",
            )
        )

# Environment: "dev" (default) or "prod" (set via ENV variable)
env = os.environ.get("ENV", "dev")

# Create Modal app
app = modal.App(
    name=env,
    image=image,
    secrets=[modal.Secret.from_name(env)]
)


@app.cls()
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


        logger.info("Container starting up!")
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
        if ENVIRONMENT not in ["dev", "test", "prod"]:
            raise ValueError(f"Invalid ENVIRONMENT value: {ENVIRONMENT}. Must be one of: dev, test, prod")
        logger.info(f"Running in environment: {ENVIRONMENT}")

        # Instantiate classes
        self.preprocessor = Preprocessor(min_chunk_duration=1.0, max_chunk_duration=10.0, scene_threshold=13.0)
        self.video_embedder = VideoEmbedder()
        self.pinecone_connector = PineconeConnector(api_key=PINECONE_API_KEY, index_name=PINECONE_CHUNKS_INDEX)
        self.job_store = JobStoreConnector(dict_name="clipabit-jobs")
        
        # Frame storage for testing/debugging (stores PIL Images as base64)
        self.frame_store = modal.Dict.from_name("clipabit-frames", create_if_missing=True)
        
        # Initialize semantic searcher
        self.searcher = Searcher(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_CHUNKS_INDEX,
            namespace=""  # Use default namespace
        )
        logger.info(f"Searcher initialized (device: {self.searcher.device})")
        
        # Removed FrameEmbedder initialization

        self.r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=ENVIRONMENT
        )

        logger.info("Container modules initialized and ready!")

        print(f"[Container] Started at {self.start_time.isoformat()}")

    @modal.method()
    async def process_video(self, video_bytes: bytes, filename: str, job_id: str):
        logger.info(f"[Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes)")
        
        try:
            # Upload original video to R2 bucket
            # TODO: do this in parallel with processing and provide url once done
            success, hashed_identifier = self.r2_connector.upload_video(
                video_data=video_bytes,
                filename=filename,
                # user_id="user1" # Specify user ID once we have user management
            )
            if not success:
                raise Exception(f"Failed to upload video to R2 storage: {hashed_identifier}")

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
            
            # TODO: Upload processed data to S3 (video storage)

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
              
               
                self.pinecone_connector.upsert_chunk(
                    chunk_id=chunk['chunk_id'],
                    chunk_embedding=embedding.numpy(),
                    namespace="",
                    metadata=chunk['metadata']
                )            
                
                chunk_details.append({
                    "chunk_id": chunk['chunk_id'],
                    "metadata": chunk['metadata'],
                    "memory_mb": chunk['memory_mb'],
                })
              
            # TODO: Upload processed data to S3

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
    async def upload(self, file: UploadFile = None):
        """
        Handle video file upload and start background processing.

        Args:
            file (UploadFile): The uploaded video file.

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
            "content_type": file.content_type
        })

        # Spawn background processing (non-blocking - returns immediately)
        self.process_video.spawn(contents, file.filename, job_id)

        return {
            "job_id": job_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "status": "processing",
            "message": "Video uploaded successfully, processing in background"
        }

    
    @modal.fastapi_endpoint(method="GET")
    async def search(self, query: str):
        """
        Search endpoint - accepts a text query and returns semantic search results.

        Args:
        - query (str): The search query string (required)
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
            logger.info(f"[Search] Query: '{query}' | top_k={top_k}")

            # Execute semantic search
            results = self.searcher.search(
                query=query,
                top_k=top_k
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