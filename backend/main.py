from fastapi import UploadFile, HTTPException
import modal

# Configure Modal app and image
# dependencies found in pyproject.toml
image = (
            modal.Image.debian_slim(python_version="3.12")
            .uv_sync(extra_options="--no-dev")  # exclude dev dependencies to avoid package conflicts
            .add_local_python_source(  # add all local modules here
                "preprocessing",
                "embeddings",
                "models",
            )
        )
app = modal.App(name="ClipABit", image=image)

# Shared storage for job results across containers
job_store = modal.Dict.from_name("clipabit-jobs", create_if_missing=True)


@app.cls()
class Server:

    @modal.enter()
    def startup(self):
        """
            Startup logic. This runs once when the container starts.
            Here is where you would instantiate classes and load models that are
            reused across multiple requests to avoid reloading them each time.
        """
        from datetime import datetime, timezone
        from preprocessing.preprocessor import Preprocessor  # Import local module inside class

        self.start_time = datetime.now(timezone.utc)

        self.preprocessor = Preprocessor(
            min_chunk_duration=1.0,
            max_chunk_duration=10.0,
            scene_threshold=13.0,
        )

        print(f"[Container] Started at {self.start_time.isoformat()}")

    @modal.method()
    async def process_video(self, video_bytes: bytes, filename: str, job_id: str):
        """Background video processing task - runs in its own container."""
        print(f"[Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes)")
        
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

            # Store result for polling endpoint in shared storage
            job_store[job_id] = result
            return result

        except Exception as e:
            print(f"[Job {job_id}] Processing failed: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
            error_result = {"job_id": job_id, "status": "failed", "error": str(e)}

            # Store error result for polling endpoint in shared storage
            job_store[job_id] = error_result
            return error_result

    @modal.fastapi_endpoint(method="GET")
    async def status(self, job_id: str):
        """Poll job status - returns processing status and results when complete."""
        if job_id not in job_store:
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Job is still processing or not found"
            }

        return job_store[job_id]

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

        # Log upload details
        print(f"[Upload] {job_id}: {file.filename} ({file_size} bytes, {file.content_type})")

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

