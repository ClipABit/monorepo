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
            )
        )

# Load secrets from .env file
modal.Secret.from_dotenv(filename=".env")
secrets = modal.Secret.objects.list()

# Create Modal app
app = modal.App(name="ClipABit", image=image, secrets=secrets)


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
        from database.pinecone_connector import PineconeConnector
        from database.job_store_connector import JobStoreConnector


        logger.info("Container starting up!")
        self.start_time = datetime.now(timezone.utc)

        # Get environment variables
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        # Instantiate classes

        logger.info("Container modules initialized and ready!")
        self.preprocessor = Preprocessor(min_chunk_duration=1.0, max_chunk_duration=10.0, scene_threshold=13.0)
        self.pinecone_connector = PineconeConnector(api_key=PINECONE_API_KEY, index_name=PINECONE_CHUNKS_INDEX)
        self.job_store = JobStoreConnector(dict_name="clipabit-jobs")

        print(f"[Container] Started at {self.start_time.isoformat()}")

    @modal.method()
    async def process_video(self, video_bytes: bytes, filename: str, job_id: str):
        """Background video processing task - runs in its own container."""
        import time
        logger.info(f"[Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes)")
        
        try:
            # Process video through preprocessing pipeline
            processed_chunks = self.preprocessor.process_video_from_bytes(
                video_bytes=video_bytes,
                video_id=job_id,
                filename=filename,
                s3_url=""  # TODO: Add S3 URL when storage is implemented
            )
            
            # Calculate summary statistics
            total_frames = sum(chunk['metadata']['frame_count'] for chunk in processed_chunks)
            total_memory = sum(chunk['memory_mb'] for chunk in processed_chunks)
            avg_complexity = sum(chunk['metadata']['complexity_score'] for chunk in processed_chunks) / len(processed_chunks) if processed_chunks else 0

            print(f"[Job {job_id}] Complete: {len(processed_chunks)} chunks, {total_frames} frames, {total_memory:.2f} MB, avg_complexity={avg_complexity:.3f}")
            
            # TODO: Send chunks to embedding module
            # TODO: Store results in database
            # TODO: Upload processed data to S3

            # Prepare chunk details for response (without frame arrays)
            chunk_details = []
            for chunk in processed_chunks:
                chunk_details.append({
                    "chunk_id": chunk['chunk_id'],
                    "metadata": chunk['metadata'],
                    "memory_mb": chunk['memory_mb']
                })

            result = {
                "job_id": job_id,
                "status": "completed",
                "filename": filename,
                "chunks": len(processed_chunks),
                "total_frames": total_frames,
                "total_memory_mb": total_memory,
                "avg_complexity": avg_complexity,
                "chunk_details": chunk_details
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
        """Poll job status - returns processing status and results when complete."""
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
        Video upload endpoint - accepts video file uploads and starts background processing.
        Returns a job ID for polling status.
        """
        # TODO: Add error handling for file types and sizes
        import uuid
        
        if file is None:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file contents
        contents = await file.read()
        file_size = len(contents)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Log upload details
        logger.info(f"[Upload] {job_id}: {file.filename} ({file_size} bytes, {file.content_type})")

        # Create initial job entry in shared storage
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

    @modal.fastapi_endpoint(method="POST")
    async def search(self, query: str):
        """Search endpoint - accepts a text query and returns semantic search results."""
        logger.info(f"[Search] Query: {query}")
        
        # TODO: Implement search and rerank logic and use class models here

        return {
            "query": query,
            "status": "success"
        }
