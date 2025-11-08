import os
import logging
from fastapi import UploadFile, HTTPException
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
            .uv_sync()
            .add_local_python_source(  # add all local modules here
                "preprocessing",
                "embeddings",
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


        logger.info("Container starting up!")
        self.start_time = datetime.now(timezone.utc)
        
        # Get environment variables
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        # Instantiate classes
        self.preprocessor = Preprocessor()
        self.pinecone_connector = PineconeConnector(api_key=PINECONE_API_KEY)

        logger.info("Container modules initialized and ready!")

    @modal.method()
    async def process_video(self, video_bytes: bytes, filename: str, job_id: str):
        """Background video processing task - runs in its own container."""
        import time
        logger.info(f"[Job {job_id}] Starting processing for {filename}")
        
        # TODO: Add actual video processing here
        try:
            time.sleep(5)
        except Exception as e:
            logger.error(f"[Job {job_id}] Processing failed: {e}")
            return {"job_id": job_id, "status": "failed", "error": str(e)}
        
        logger.info(f"[Job {job_id}] Finished processing {filename}")
        
        return {
            "job_id": job_id, 
            "status": "completed", 
            "filename": filename,
            "preprocessing": "yay"
        }

    @modal.fastapi_endpoint(method="POST")
    async def upload(self, file: UploadFile = None):
        import uuid
        
        if file is None:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file contents
        contents = await file.read()
        file_size = len(contents)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # log file details
        logger.info(f"Received file: {file.filename}")
        logger.info(f"Content-Type: {file.content_type}")
        logger.info(f"Size: {file_size} bytes")
        logger.info(f"Spawning background job: {job_id}")
        
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

