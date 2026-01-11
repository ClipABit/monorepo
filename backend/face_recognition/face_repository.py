import numpy as np
from .face import Face
import logging
from database import PineconeConnector, R2Connector
import uuid
from io import BytesIO
from PIL import Image
from dataclasses import dataclass
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectedFaceData:
    face_id: str
    img_access_id: str

class FaceRepository:
    def __init__(
            self, 
            pinecone_connector: PineconeConnector,
            r2_connector: R2Connector,
            threshold = 0.35,
            image_serializer: Optional[Callable[[np.ndarray], bytes]] = None,
        ):
        self.threshold = threshold
        
        # Pinecone connector for vector database operations
        self.pinecone_connector = pinecone_connector

        self.r2_connector = r2_connector

        # image_serializer: callable that converts ndarray -> bytes
        self.image_serializer = image_serializer or self._ndarray_to_png_bytes
        
        logging.debug(f"FaceRepository: Initialized FaceRepository with pinecone_index={pinecone_connector.index} and threshold={threshold}")

    # convert ndarray image to PNG bytes for uploading to r2 storage
    def _ndarray_to_png_bytes(self, img_ndarray):
        img = Image.fromarray(img_ndarray)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()  # bytes

    # similarity search in pinecone to find closest face embedding (similarity has to be above threshold)
    # if found, return face identity, else return None
    def get_face_identity(self, namespace: str, face: Face) -> DetectedFaceData | None:
        embedding = face.embedding

        # find closest face from pinecone vector db
        best_match = self.pinecone_connector.query_chunks(
            query_embedding=embedding,
            namespace=namespace,
            top_k=1
        )

        if not best_match or len(best_match) == 0:
            return None

        match = best_match[0]
        if match.get('score', 0) < self.threshold:
            return None
        else:
            metadata = match.get("metadata", {})
            face_id = metadata.get("face_id")
            img_access_id = metadata.get("img_access_id")
            if not face_id or not img_access_id:
                logger.error(f"FaceRepository: Found matching face in pinecone but missing metadata face_id or img_access_id. Metadata: {metadata}")
                raise Exception("Corrupted metadata in pinecone for matched face.")
            return DetectedFaceData(face_id=str(face_id), img_access_id=str(img_access_id))
        
    def generate_face_id(self) -> str:
        return str(uuid.uuid4())

    # store the face image in r2 storage, return the storage identifier / url
    def upload_face_image(self, face: Face, face_id: str = None) -> str | None:
        # Use the injected serializer (or default) to convert ndarray -> bytes
        face_img_bytes = self.image_serializer(face.face_image)
        if not face_id:
            face_id = self.generate_face_id()
            logger.warning("FaceRepository: No face_id provided for upload_face_image, gnerating new face_id {face_id}")
        face_file_name = f"face_{face_id}.png"
        upload_success, identifier = self.r2_connector.upload_image(
            image_data=face_img_bytes,
            filename=face_file_name,
            namespace="face_images"
        )
        if not upload_success:
            logger.error(f"FaceRepository: Failed to upload face image to R2 with face_id {face_id}. Error message: {identifier}")
            return None
        
        return identifier

    # upsert the face embedding with metadata into pinecone vector database
    def upsert_identified_face_embedding(self, namespace: str, face_id: str, img_access_id: str, video_chunk_id: str, face: Face):
        try:
            success = self.pinecone_connector.upsert_chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_embedding=face.embedding,
                namespace=namespace,
                metadata={"face_id": face_id, "img_access_id": img_access_id, "chunk_id": video_chunk_id}
            )
        except Exception as e:
            logger.error(f"FaceRepository: Error upserting face embedding for face_id {face_id} in chunk_id {video_chunk_id}: {e}")
            return False
        
        if not success:
            logger.error(f"FaceRepository: Failed to upsert face embedding for face_id {face_id} in chunk_id {video_chunk_id}.")
            return False
        return True