from fastapi import UploadFile, HTTPException
import modal

# Configure Modal app and image
# dependencies found in pyproject.toml
image = (
            modal.Image.debian_slim(python_version="3.12")
            .uv_sync(extra_options="--no-dev") 
            .add_local_python_source(  # add all local modules here
                "preprocessing",
                "embeddings",
                "models",
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

        self.preprocessor = Preprocessor(
            min_chunk_duration=1.0,
            max_chunk_duration=10.0,
            scene_threshold=13.0,
        )

        print("✅ Preprocessor initialized and ready!")

    @modal.method()
    async def process_video(self, video_bytes: bytes, filename: str, job_id: str):
        """Background video processing task - runs in its own container."""
        import time
        print(f"[Job {job_id}] Starting processing for {filename}")
        
        try:
            # Process video through preprocessing pipeline
            processed_chunks = self.preprocessor.process_video_from_bytes(
                video_bytes=video_bytes,
                video_id=job_id,
                filename=filename,
                s3_url=""  # TODO: Add S3 URL when storage is implemented
            )
            
            print(f"[Job {job_id}] Preprocessing complete: {len(processed_chunks)} chunks")
            
            # Log summary statistics
            total_frames = sum(chunk['metadata']['frame_count'] for chunk in processed_chunks)
            total_memory = sum(chunk['memory_mb'] for chunk in processed_chunks)
            avg_complexity = sum(chunk['metadata']['complexity_score'] for chunk in processed_chunks) / len(processed_chunks) if processed_chunks else 0
            
            print(f"[Job {job_id}] Statistics:")
            print(f"  - Total frames: {total_frames}")
            print(f"  - Total memory: {total_memory:.2f} MB")
            print(f"  - Avg complexity: {avg_complexity:.3f}")
            
            # TODO: Send chunks to embedding module
            # TODO: Store results in database
            # TODO: Upload processed data to S3
            
            return {
                "job_id": job_id, 
                "status": "completed", 
                "filename": filename,
                "chunks": len(processed_chunks),
                "total_frames": total_frames,
                "total_memory_mb": total_memory,
                "avg_complexity": avg_complexity
            }
            
        except Exception as e:
            print(f"[Job {job_id}] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return {"job_id": job_id, "status": "failed", "error": str(e)}

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

