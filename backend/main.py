from fastapi import UploadFile, HTTPException
import modal

# import all necessary modules in the image
image = (
            modal.Image.debian_slim(python_version="3.12")
            .uv_sync()
            .add_local_python_source(
                "preprocessing"
            )
        )
app = modal.App(name="ClipABit", image=image)


@app.cls()
class Server:
    @modal.enter()
    def startup(self):
        """
            Startup logic. This runs once when the container starts.
            Here is where you would instantiate classes and load models that are
            reused across multiple requests to avoid reloading them each time.
        """
        print("Container starting up!")

        from datetime import datetime, timezone
        from preprocessing.preprocessor import Preprocessor  # Import local module inside class
        
        self.start_time = datetime.now(timezone.utc)

        # Instantiate classes
        self.preprocessor = Preprocessor()
        
        print("✅ Preprocessor initialized and ready!")

    @modal.method()
    async def process_video(self, video_bytes: bytes, filename: str, job_id: str):
        """Background video processing task - runs in its own container."""
        import time
        print(f"[Job {job_id}] Starting processing for {filename}")
        
        # TODO: Add actual video processing here
        try:
            time.sleep(5)
        except Exception as e:
            print(f"[Job {job_id}] Processing failed: {e}")
            return {"job_id": job_id, "status": "failed", "error": str(e)}
        
        print(f"[Job {job_id}] Finished processing {filename}")
        
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
        print(f"Received file: {file.filename}")
        print(f"Content-Type: {file.content_type}")
        print(f"Size: {file_size} bytes")
        print(f"Spawning background job: {job_id}")
        
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

