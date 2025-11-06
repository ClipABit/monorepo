from fastapi import UploadFile, HTTPException
import modal

# import all necessary modules in the image
image = (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install(
                "fastapi[standard]",
                "python-multipart", 
                "ffmpeg-python", 
                "opencv-python-headless", 
                "numpy",
                "torch",
                "transformers",
                "Pillow",
                "pinecone",
                "python-dotenv"
            )
            # Package specific modules explicitly ("." is not a valid package name)
            .add_local_python_source("pinecone_connect")
            .add_local_python_source("embedding_service")
            .add_local_python_source(
                "preprocessing"
            )
        )
app = modal.App(name="ClipABit", image=image)


@app.cls(secrets=[modal.Secret.from_name("pinecone")])
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
        from dotenv import load_dotenv
        import pinecone_connect as pc
        import embedding_service as embedding_service  # preload on first use
        from preprocessing.preprocessor import Preprocessor  # Import local module inside class
        
        load_dotenv()
        self.start_time = datetime.now(timezone.utc)

        # Instantiate classes
        self.preprocessor = Preprocessor()
        # Ensure Pinecone index exists (auto-create if missing)
        try:
            client = pc.connect_to_pinecone()
            pc.ensure_index(client)
            print("✅ Pinecone index ensured")
        except Exception as e:
            print(f"⚠️ Pinecone ensure_index failed: {e}")
        # Warm up CLIP text model to avoid first-request timeout
        try:
            _ = embedding_service.embed_text("warmup")
            print("✅ CLIP text model warmed")
        except Exception as e:
            print(f"⚠️ CLIP warmup failed: {e}")
        
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

        # Auto-ingest: create a simple audio-modality record using filename as placeholder transcript
        # This enables immediate semantic search testing without full processing.
        try:
            import embedding_service as embedding_service
            import pinecone_connect as pc
            client = pc.connect_to_pinecone()
            text_for_ingest = (file.filename or "uploaded media").rsplit('.', 1)[0]
            qv = embedding_service.embed_text(text_for_ingest)
            pc.upsert_audio_chunk(
                client,
                video_id=job_id,
                chunk_id="1",
                start_ts=0.0,
                end_ts=3.0,
                text_vec_512=qv,
                transcript_snippet=text_for_ingest,
                language=None,
            )
            auto_ingest = {"ok": True, "id": f"chunk:{job_id}:1:audio", "text": text_for_ingest}
        except Exception as e:
            auto_ingest = {"ok": False, "error": str(e)}

        return {
            "job_id": job_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "status": "processing",
            "message": "Media uploaded successfully; processing in background; auto-ingest added for quick search",
            "auto_ingest": auto_ingest,
        }

    @modal.fastapi_endpoint(method="POST")
    async def search(self, payload: dict):
        try:
            import time
            t_start = time.perf_counter()
            query = payload.get("query", "").strip()
            if not query:
                raise HTTPException(status_code=400, detail="Missing 'query'")
            top_k = int(payload.get("top_k", 3))
            if top_k <= 0:
                top_k = 3
            filters = payload.get("filters")
            print(f"[search] start | top_k={top_k} | filters={filters}")
            t_embed0 = time.perf_counter()
            import embedding_service as embedding_service
            import pinecone_connect as pc

            qv = embedding_service.embed_text(query)
            t_embed1 = time.perf_counter()
            print(f"[search] embed ok | secs={t_embed1 - t_embed0:.2f}")
            t_query0 = time.perf_counter()
            client = pc.connect_to_pinecone()
            res = pc.query_vectors(
                client,
                qv,
                top_k=top_k,
                metadata_filter=filters,
                include_metadata=True,
                include_values=False,
            )
            t_query1 = time.perf_counter()
            print(f"[search] pinecone query ok | secs={t_query1 - t_query0:.2f}")

            results = []
            for m in getattr(res, "matches", []) or []:
                md = m.metadata or {}
                results.append({
                    "id": m.id,
                    "score": m.score,
                    "video_id": md.get("video_id"),
                    "chunk_id": md.get("chunk_id"),
                    "modality": md.get("modality"),
                    "start_ts": md.get("start_ts"),
                    "end_ts": md.get("end_ts"),
                    "transcript_snippet": md.get("transcript_snippet"),
                    "preview_frame_uri": md.get("preview_frame_uri"),
                })
            t_done = time.perf_counter()
            print(f"[search] done | matches={len(results)} | total_secs={t_done - t_start:.2f}")
            return {"results": results}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @modal.fastapi_endpoint(method="POST")
    async def ingest_chunk(self, payload: dict):
        """
        Minimal ingest endpoint for demo:
        - Embeds provided text with CLIP text encoder (512-d)
        - Upserts as an audio chunk into Pinecone
        Body:
        {
          "video_id": "vidA",
          "chunk_id": "1",
          "start_ts": 0.0,
          "end_ts": 3.2,
          "text": "hello world transcript",
          "language": "en"
        }
        """
        try:
            video_id = payload.get("video_id")
            chunk_id = payload.get("chunk_id")
            start_ts = payload.get("start_ts")
            end_ts = payload.get("end_ts")
            text = (payload.get("text") or "").strip()
            language = payload.get("language")

            # Basic validation
            if not video_id or not chunk_id or start_ts is None or end_ts is None or not text:
                raise HTTPException(status_code=400, detail="Missing one of: video_id, chunk_id, start_ts, end_ts, text")

            import embedding_service as embedding_service
            import pinecone_connect as pc
            client = pc.connect_to_pinecone()

            text_vec = embedding_service.embed_text(text)
            pc.upsert_audio_chunk(
                client,
                video_id=video_id,
                chunk_id=chunk_id,
                start_ts=float(start_ts),
                end_ts=float(end_ts),
                text_vec_512=text_vec,
                transcript_snippet=text,
                language=language,
            )
            return {"ok": True, "id": f"chunk:{video_id}:{chunk_id}:audio"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

