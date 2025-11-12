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
            .add_local_python_source("preprocessing")
            .add_local_python_source("search")      # Add search module
            .add_local_python_source("database")    # Add database module
            .add_local_python_source("models")      # Add models module
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
        from preprocessing.preprocessor import Preprocessor
        from search.searcher import Searcher
        import os
        
        load_dotenv()
        self.start_time = datetime.now(timezone.utc)
        # Isolate vectors per run using a unique namespace
        import uuid as _uuid
        self.namespace = f"session:{_uuid.uuid4().hex[:8]}"
        print(f"✅ Session namespace: {self.namespace}")

        # Get Pinecone API key from environment
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        # Instantiate classes
        self.preprocessor = Preprocessor()
        
        # Initialize searcher with session namespace for isolation
        self.searcher = Searcher(
            api_key=PINECONE_API_KEY,
            index_name="chunks-index",
            namespace=self.namespace  # Use session namespace for isolation
        )
        print(f"✅ Searcher initialized (device: {self.searcher.device})")
        
        # TODO: Ensure Pinecone index exists (requires refactoring old pinecone_connect module)
        # For now, assuming index already exists
        
        # Warm up CLIP model through searcher
        try:
            _ = self.searcher.embedder.embed_text("warmup")
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
        
        # Clear prior vectors for this session so searches only see the next upload
        try:
            import pinecone_connect as pc
            client = pc.connect_to_pinecone()
            pc.delete_namespace(client, namespace=self.namespace)
            print(f"[upload] cleared namespace {self.namespace}")
        except Exception as e:
            print(f"[upload] namespace clear failed: {e}")
        
        # Spawn background processing (non-blocking - returns immediately)
        self.process_video.spawn(contents, file.filename, job_id)

        # Auto-ingest a placeholder vector so search works immediately after upload.
        # Uses filename (without extension) as a simple text snippet.
        auto_ingest = {"ok": False}
        try:
            import numpy as _np
            import pinecone_connect as pc
            client = pc.connect_to_pinecone()
            text_for_ingest = (file.filename or "uploaded media").rsplit(".", 1)[0]
            qv = self.searcher.embedder.embed_text(text_for_ingest)
            pc.upsert_audio_chunk(
                client,
                video_id=job_id,
                chunk_id="1",
                start_ts=0.0,
                end_ts=3.0,
                text_vec_512=qv,
                transcript_snippet=text_for_ingest,
                language=None,
                namespace=self.namespace,
            )
            new_id = f"chunk:{job_id}:1:audio"
            # Fetch back to verify storage and report vector info
            fetched = pc.fetch_by_id(client, new_id, namespace=self.namespace)
            # pinecone fetch returns a dict-like; handle defensively
            vectors = getattr(fetched, "vectors", None) or getattr(fetched, "to_dict", lambda: {})()
            # Support both dict object and .vectors mapping
            vec_entry = None
            if isinstance(fetched, dict):
                vec_entry = (fetched.get("vectors") or {}).get(new_id)
            elif hasattr(fetched, "vectors"):
                vec_entry = fetched.vectors.get(new_id)  # type: ignore
            if not vec_entry:
                # Fallback: report what we upserted
                dim = int(qv.shape[0]) if hasattr(qv, "shape") else len(qv)
                auto_ingest = {
                    "ok": True,
                    "id": new_id,
                    "text": text_for_ingest,
                    "namespace": self.namespace,
                    "vector": {"dim": dim, "l2_norm": float(_np.linalg.norm(qv))},
                    "verified": False,
                }
            else:
                values = vec_entry.get("values") or []
                dim = len(values)
                l2 = float(_np.linalg.norm(_np.array(values))) if values else float(_np.linalg.norm(qv))
                auto_ingest = {
                    "ok": True,
                    "id": new_id,
                    "text": text_for_ingest,
                    "namespace": self.namespace,
                    "vector": {
                        "dim": dim,
                        "l2_norm": l2,
                    },
                    "metadata": vec_entry.get("metadata") or {},
                    "verified": True,
                }
            print(f"[upload] auto-ingested placeholder for job {job_id} in {self.namespace}")
        except Exception as e:
            auto_ingest = {"ok": False, "error": str(e)}
            print(f"[upload] auto-ingest failed: {e}")

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
            events = []
            query = payload.get("query", "").strip()
            if not query:
                raise HTTPException(status_code=400, detail="Missing 'query'")
            top_k = int(payload.get("top_k", 1))
            if top_k <= 0:
                top_k = 1
            filters = payload.get("filters")
            print(f"[search] start | top_k={top_k} | filters={filters}")
            events.append({"stage": "start", "desc": "Search received", "t_rel_s": 0.0})
            
            # Use the clean searcher module instead of raw calls
            t_embed0 = time.perf_counter()
            results = self.searcher.search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            t_done = time.perf_counter()
            
            # Add timing events
            events.append({
                "stage": "search",
                "desc": "Executed modular search (embed + query)",
                "t_rel_s": round(t_done - t_start, 3),
                "dur_s": round(t_done - t_embed0, 3),
            })
            
            print(f"[search] done | matches={len(results)} | total_secs={t_done - t_start:.2f}")
            
            timing = {
                "total_s": round(t_done - t_start, 3),
                "namespace": self.namespace,
                "stages": events,
            }
            return {"results": results, "timing": timing}
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

            import numpy as _np
            import pinecone_connect as pc
            client = pc.connect_to_pinecone()

            text_vec = self.searcher.embedder.embed_text(text)
            pc.upsert_audio_chunk(
                client,
                video_id=video_id,
                chunk_id=chunk_id,
                start_ts=float(start_ts),
                end_ts=float(end_ts),
                text_vec_512=text_vec,
                transcript_snippet=text,
                language=language,
                namespace=self.namespace,
            )
            new_id = f"chunk:{video_id}:{chunk_id}:audio"
            fetched = pc.fetch_by_id(client, new_id, namespace=self.namespace)
            vectors = getattr(fetched, "vectors", None) or getattr(fetched, "to_dict", lambda: {})()
            vec_entry = None
            if isinstance(fetched, dict):
                vec_entry = (fetched.get("vectors") or {}).get(new_id)
            elif hasattr(fetched, "vectors"):
                vec_entry = fetched.vectors.get(new_id)  # type: ignore
            info = {"id": new_id, "namespace": self.namespace}
            if vec_entry:
                values = vec_entry.get("values") or []
                info["vector"] = {"dim": len(values), "l2_norm": float(_np.linalg.norm(_np.array(values))) if values else None}
                info["metadata"] = vec_entry.get("metadata") or {}
                info["verified"] = True
            else:
                dim = int(text_vec.shape[0]) if hasattr(text_vec, "shape") else len(text_vec)
                info["vector"] = {"dim": dim, "l2_norm": float(_np.linalg.norm(text_vec))}
                info["verified"] = False
            return {"ok": True, **info}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


