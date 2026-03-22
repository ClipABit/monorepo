"""
ProcessingService class - base class shared between processing_app.py and dev_combined.py
"""

import logging
import os
import modal

from shared.config import get_environment, get_env_var, get_pinecone_index

logger = logging.getLogger(__name__)


class ProcessingService:
    """
    Processing service base class.
    
    Loads full CLIP model and preprocessing pipeline on startup.
    """

    @modal.enter()
    def startup(self):
        """Load CLIP model, face recognition, and initialize all connectors."""
        from preprocessing.preprocessor import Preprocessor
        from embeddings.video_embedder import VideoEmbedder
        from database.pinecone_connector import PineconeConnector
        from database.cache.job_store_connector import JobStoreConnector
        from database.r2_connector import R2Connector
        from face_recognition import (
            FaceDetector,
            FaceRepository,
            FaceMetadataRepository,
            FaceAppearanceRepository,
        )
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import firebase_admin
        import json
        from firebase_admin import credentials, firestore

        # Configure TensorFlow / Keras behavior for DeepFace (match main.py)
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        env = get_environment()
        logger.info(f"[{self.__class__.__name__}] Starting up in '{env}' environment")

        # Get environment variables
        PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
        R2_ACCOUNT_ID = get_env_var("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = get_env_var("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = get_env_var("R2_SECRET_ACCESS_KEY")

        pinecone_index = get_pinecone_index()
        logger.info(f"[{self.__class__.__name__}] Using Pinecone index: {pinecone_index}")
        pinecone_face_index = f"{env}-faces"
        logger.info(f"[{self.__class__.__name__}] Using Pinecone face index: {pinecone_face_index}")

        # Initialize preprocessor and embedder
        self.preprocessor = Preprocessor(
            min_chunk_duration=1.0,
            max_chunk_duration=10.0,
            scene_threshold=13.0
        )
        self.video_embedder = VideoEmbedder()
        logger.info(f"[{self.__class__.__name__}] CLIP image encoder and preprocessor loaded")

        # Initialize connectors
        self.pinecone_connector = PineconeConnector(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index
        )
        self.face_recognition_pinecone_connector = PineconeConnector(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_face_index,
        )
        self.job_store = JobStoreConnector(dict_name="clipabit-jobs")
        self.r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=env
        )

        firebase_credentials = json.loads(get_env_var("FIREBASE_ADMIN_KEY"))
        cred = credentials.Certificate(firebase_credentials)
        try:
            firebase_admin.initialize_app(cred)
        except ValueError:
            pass  # Already initialized
        firestore_client = firestore.client()

        self.face_metadata_repository = FaceMetadataRepository(firestore_client)
        self.face_appearance_repository = FaceAppearanceRepository(firestore_client)

        # Initialize MediaPipe face landmarker and face recognition helpers
        base_options = python.BaseOptions(model_asset_path="/models/face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        landmarker_instance = vision.FaceLandmarker.create_from_options(options)

        self.face_detector = FaceDetector(landmarker_instance=landmarker_instance)
        self.face_repository = FaceRepository(
            self.face_recognition_pinecone_connector,
            self.r2_connector,
            threshold=0.5,
        )

        logger.info(f"[{self.__class__.__name__}] Initialized and ready!")

    @modal.method()
    def process_video_background(
        self,
        video_bytes: bytes,
        filename: str,
        job_id: str,
        namespace: str = "",
        parent_batch_id: str = None
    ):
        """Process an uploaded video through the full pipeline."""
        logger.info(
            f"[{self.__class__.__name__}][Job {job_id}] Processing started: {filename} ({len(video_bytes)} bytes) "
            f"| namespace='{namespace}' | batch={parent_batch_id or 'None'}"
        )

        hashed_identifier = None
        upserted_chunk_ids = []
        created_face_embedding_ids: list[str] = []
        created_face_image_ids: list[str] = []
        created_face_metadata_ids: list[str] = []
        stored_face_appearance_ids: list[tuple[str, str]] = []

        try:
            # Stage 1: Upload original video to R2 bucket
            success, hashed_identifier = self.r2_connector.upload_video(
                video_data=video_bytes,
                filename=filename,
                namespace=namespace
            )
            if not success:
                upload_error_details = hashed_identifier
                hashed_identifier = None
                raise Exception(f"Failed to upload video to R2 storage: {upload_error_details}")

            # Stage 2: Process video through preprocessing pipeline
            processed_chunks = self.preprocessor.process_video_from_bytes(
                video_bytes=video_bytes,
                video_id=job_id,
                filename=filename,
                hashed_identifier=hashed_identifier
            )

            # Calculate summary statistics
            total_frames = sum(chunk['metadata']['frame_count'] for chunk in processed_chunks)
            total_memory = sum(chunk['memory_mb'] for chunk in processed_chunks)
            avg_complexity = (
                sum(chunk['metadata']['complexity_score'] for chunk in processed_chunks)
                / len(processed_chunks)
                if processed_chunks else 0
            )

            logger.info(
                f"[{self.__class__.__name__}][Job {job_id}] Complete: {len(processed_chunks)} chunks, "
                f"{total_frames} frames, {total_memory:.2f} MB, avg_complexity={avg_complexity:.3f}"
            )

            # Stage 3-4: Embed frames and store in Pinecone
            logger.info(f"[{self.__class__.__name__}][Job {job_id}] Embedding and upserting {len(processed_chunks)} chunks")

            chunk_details = []
            for chunk in processed_chunks:
                embedding = self.video_embedder._generate_clip_embedding(
                    chunk["frames"],
                    num_frames=8
                )

                logger.info(f"[{self.__class__.__name__}][Job {job_id}] Generated CLIP embedding for chunk {chunk['chunk_id']}")

                # Transform metadata for Pinecone compatibility
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

            # After chunk processing, run face recognition pipeline on sampled frames
            try:
                from face_recognition.video_face_pipeline import VideoFacePipeline

                face_map: dict[str, str] = {}
                pipeline = VideoFacePipeline(
                    namespace=namespace,
                    face_detector=self.face_detector,
                    face_repository=self.face_repository,
                    face_metadata_repository=self.face_metadata_repository,
                )
                mapping, vec_ids, img_ids, face_ids = pipeline.process_video_chunks(processed_chunks)
                for k, v in mapping.items():
                    face_map.setdefault(k, v)

                # Collect created resource ids for potential rollback
                if vec_ids:
                    created_face_embedding_ids.extend(vec_ids)
                if img_ids:
                    created_face_image_ids.extend(img_ids)
                if face_ids:
                    created_face_metadata_ids.extend(face_ids)
            except Exception as e:
                logger.error(
                    "[%s][Job %s] Failed to initialize or run VideoFacePipeline: %s",
                    self.__class__.__name__,
                    job_id,
                    e,
                )
                face_map = {}

            for face_id in getattr(face_map, "keys", lambda: [])():
                success = self.face_appearance_repository.set_face_appearance(
                    user_id=namespace,
                    face_id=face_id,
                    video_chunk_id=job_id,
                )
                if success:
                    stored_face_appearance_ids.append((face_id, job_id))
                    logger.info(
                        "[%s][Job %s] Stored face appearance for face_id=%s, video_id=%s",
                        self.__class__.__name__,
                        job_id,
                        face_id,
                        job_id,
                    )
                else:
                    logger.info(
                        "[%s][Job %s] Failed to store face appearance for face_id=%s, video_id=%s",
                        self.__class__.__name__,
                        job_id,
                        face_id,
                        job_id,
                    )

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

            logger.info(f"[{self.__class__.__name__}][Job {job_id}] Finished processing {filename}")

            # Stage 5: Store result
            self.job_store.set_job_completed(job_id, result)

            # Update parent batch if exists
            if parent_batch_id:
                update_success = self.job_store.update_batch_on_child_completion(
                    parent_batch_id,
                    job_id,
                    result
                )
                if update_success:
                    logger.info(f"[{self.__class__.__name__}][Job {job_id}] Updated parent batch {parent_batch_id}")
                else:
                    logger.error(
                        f"[{self.__class__.__name__}][Job {job_id}] CRITICAL: Failed to update parent batch {parent_batch_id}"
                    )

            # Invalidate cache
            try:
                self.r2_connector.clear_cache(namespace or "__default__")
            except Exception as cache_exc:
                logger.error(f"[{self.__class__.__name__}][Job {job_id}] Failed to clear cache: {cache_exc}")

            return result

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}][Job {job_id}] Processing failed: {e}")

            # Rollback
            if hashed_identifier:
                logger.info(f"[{self.__class__.__name__}][Job {job_id}] Rolling back: Deleting video from R2")
                self.r2_connector.delete_video(hashed_identifier)

            if upserted_chunk_ids:
                logger.info(f"[{self.__class__.__name__}][Job {job_id}] Rolling back: Deleting chunks from Pinecone")
                self.pinecone_connector.delete_chunks(upserted_chunk_ids, namespace=namespace)

            # Rollback face embeddings from face-recognition Pinecone index
            if created_face_embedding_ids:
                logger.info(
                    "[%s][Job %s] Rolling back: Deleting %d face embeddings from Pinecone",
                    self.__class__.__name__,
                    job_id,
                    len(created_face_embedding_ids),
                )
                try:
                    self.face_recognition_pinecone_connector.delete_chunks(
                        created_face_embedding_ids,
                        namespace=namespace,
                    )
                except Exception as del_err:
                    logger.error(
                        "[%s][Job %s] Rollback failed for face embeddings deletion: %s",
                        self.__class__.__name__,
                        job_id,
                        del_err,
                    )

            # Rollback face images from R2
            if created_face_image_ids:
                logger.info(
                    "[%s][Job %s] Rolling back: Deleting %d face images from R2",
                    self.__class__.__name__,
                    job_id,
                    len(created_face_image_ids),
                )
                for ident in created_face_image_ids:
                    try:
                        ok = self.r2_connector.delete_image(ident)
                        if not ok:
                            logger.error(
                                "[%s][Job %s] Rollback failed for R2 face image deletion: %s",
                                self.__class__.__name__,
                                job_id,
                                ident,
                            )
                    except Exception as del_err:
                        logger.error(
                            "[%s][Job %s] Exception during face image rollback for %s: %s",
                            self.__class__.__name__,
                            job_id,
                            ident,
                            del_err,
                        )

            # Rollback face metadata from Firestore
            if created_face_metadata_ids:
                logger.info(
                    "[%s][Job %s] Rolling back: Deleting %d face metadata records from Firestore",
                    self.__class__.__name__,
                    job_id,
                    len(created_face_metadata_ids),
                )
                for face_id in created_face_metadata_ids:
                    ok = self.face_metadata_repository.delete_face_metadata(
                        user_id=namespace,
                        face_id=face_id,
                    )
                    if not ok:
                        logger.error(
                            "[%s][Job %s] Rollback failed for face metadata deletion: %s",
                            self.__class__.__name__,
                            job_id,
                            face_id,
                        )

            # Rollback face appearance records from Firestore
            if stored_face_appearance_ids:
                logger.info(
                    "[%s][Job %s] Rolling back: Deleting %d face appearance records from Firestore",
                    self.__class__.__name__,
                    job_id,
                    len(stored_face_appearance_ids),
                )
                for face_id, chunk_id in stored_face_appearance_ids:
                    ok = self.face_appearance_repository.delete_face_appearance(
                        user_id=namespace,
                        face_id=face_id,
                        video_chunk_id=chunk_id,
                    )
                    if not ok:
                        logger.error(
                            "[%s][Job %s] Rollback failed for face appearance deletion: face_id=%s, chunk_id=%s",
                            self.__class__.__name__,
                            job_id,
                            face_id,
                            chunk_id,
                        )

            import traceback
            traceback.print_exc()

            self.job_store.set_job_failed(job_id, str(e))

            if parent_batch_id:
                error_result = {
                    "job_id": job_id,
                    "status": "failed",
                    "filename": filename,
                    "error": str(e)
                }
                self.job_store.update_batch_on_child_completion(parent_batch_id, job_id, error_result)

            return {"job_id": job_id, "status": "failed", "error": str(e)}
