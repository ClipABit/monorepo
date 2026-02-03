from .face_detector import FaceDetector
from .face_repository import FaceRepository
from .face_metadata_repository import FaceMetadataRepository
from .recent_faces_cache import RecentFacesCache, FaceData
import numpy as np
import logging
from typing import Literal
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameFacePipeline:
    """Pipeline to process frames for face recognition.

    This class handles the detection of faces in video frames and computes their embeddings.
    It utilizes a face detector and an embedding model to extract features from detected faces.

    Attributes:
        detector (FaceDetector): The face detector used to find faces in frames.
    """

    def __init__(self, namespace: str, face_detector: FaceDetector, face_repository: FaceRepository, face_metadata_repository: FaceMetadataRepository):
        self.namespace = namespace
        self.face_detector = face_detector
        self.face_repository = face_repository
        self.face_metadata_repository = face_metadata_repository

    def process_frame(
            self, 
            frame: np.ndarray, 
            chunk_id: str,
            recent_faces_cache: RecentFacesCache) -> tuple[dict[str, str], list[str], list[str], list[str]]:
        """Process a video frame to detect faces and compute embeddings.

        Args:
            frame (np.ndarray): The input video frame.
            chunk_id (str): The video chunk ID associated with this frame.
            recent_faces_cache (RecentFacesCache): Cache of recently detected faces in case write to pinecone not available

        Returns:
            tuple:
                - dict mapping face_id to img_access_id for each detected face.
                - list of created vector ids (from upserts) for this frame
                - list of created image identifiers (uploaded to R2) for this frame
                - list of created face identifiers for this frame
        """
        # if frame.dtype != np.uint8:
        #     frame = frame.astype(np.uint8)
        
        # if frame_format == "RGB":
        #     # Convert RGB to BGR for OpenCV processing
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # frame = np.ascontiguousarray(frame)

        face_access_id_map: dict[str, str] = {}
        created_vector_ids: list[str] = []
        created_image_ids: list[str] = []
        created_face_ids: list[str] = []

        faces = self.face_detector.detect_and_embed(frame)
        for face in faces:
            # first try cache lookup
            cache_hit = recent_faces_cache.query_similar_face(face.embedding)
            if cache_hit is not None:
                face_data, similarity = cache_hit
                logger.info("FrameFacePipeline: Cache hit for face_id=%s with similarity=%.4f", face_data.face_id, similarity)
                face_id = face_data.face_id
                face_img_identifier = face_data.img_access_id
            else:
                # cache miss, proceed to repository lookup
                logger.info("FrameFacePipeline: Cache miss for detected face, proceeding to repository lookup")
                # The FaceRepository expects an embedding (np.ndarray) for lookup
                detected_face_data = self.face_repository.get_face_identity(self.namespace, face.embedding)

                if detected_face_data is None:
                    face_id = self.face_repository.generate_face_id()
                    logger.info("FrameFacePipeline: New face detected, generated face_id=%s", face_id)
                    face_img_identifier = self.face_repository.upload_face_image(
                        face_image=face.face_image,
                        face_id=face_id,
                    )
                    if face_img_identifier is None:
                        logger.error("FrameFacePipeline: Failed to upload face image for face_id=%s", face_id)
                        # Skip this face but continue processing others
                        continue

                    # Store face metadata
                    metadata = {
                        "face_id": face_id,
                        "image_url": face_img_identifier,
                        "given_name": None,
                    }
                    ok = self.face_metadata_repository.set_face_metadata(
                        user_id=self.namespace,
                        face_id=face_id,
                        metadata=metadata,
                    )
                    if not ok:
                        logger.error("FrameFacePipeline: Failed to store face metadata for face_id=%s", face_id)
                        # Rollback uploaded image to avoid leaving orphaned images in R2
                        deleted = self.face_repository.delete_face_image(face_img_identifier)
                        if deleted:
                            logger.info("FrameFacePipeline: Rolled back uploaded image for face_id=%s", face_id)
                        else:
                            logger.warning("FrameFacePipeline: Failed to delete uploaded image for face_id=%s; orphaned image may remain: %s", face_id, face_img_identifier)
                        
                        # Skip this face but continue processing others
                        continue
                    created_image_ids.append(face_img_identifier)
                    created_face_ids.append(face_id)
                else:
                    face_id = detected_face_data.face_id
                    face_img_identifier = detected_face_data.img_access_id

            # Use the FaceRepository upsert API (video_chunk_id, face_embedding)
            vector_id = self.face_repository.upsert_identified_face_embedding(
                namespace=self.namespace,
                face_id=face_id,
                img_access_id=face_img_identifier,
                video_chunk_id=chunk_id,
                face_embedding=face.embedding,
            )

            if vector_id:
                recent_faces_cache.add_face(
                    face_id,
                    face_data=FaceData(
                        face_id=face_id,
                        embedding=face.embedding,
                        img_access_id=face_img_identifier,
                    )
                )
                created_vector_ids.append(vector_id)
            else:
                logger.error("FrameFacePipeline: Failed to upsert face embedding for face_id=%s in chunk=%s", face_id, chunk_id)

            face_access_id_map[face_id] = face_img_identifier

        return face_access_id_map, created_vector_ids, created_image_ids, created_face_ids