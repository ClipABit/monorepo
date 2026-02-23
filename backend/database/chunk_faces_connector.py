import logging
import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkFacesConnector:
    """
    Modal dict connector for storing face_id appeared in each chunk and number of times it appears.
    """
    
    DEFAULT_DICT_NAME = "chunk-faces-store"

    def __init__(self, dict_name: str = DEFAULT_DICT_NAME):
        self.dict_name = dict_name
        self.chunk_faces_store = modal.Dict.from_name(dict_name, create_if_missing=True)
        logger.info(f"Initialized ChunkFacesConnector with Dict: {dict_name}")

    def add_chunk_faces(self, chunk_id: str, face_counts: dict[str, int]) -> bool:
        """Add or update face counts for a given chunk."""
        try:
            if chunk_id in self.chunk_faces_store:
                existing_counts = self.chunk_faces_store[chunk_id]
                for face_id, count in face_counts.items():
                    existing_counts[face_id] = existing_counts.get(face_id, 0) + count
                self.chunk_faces_store[chunk_id] = existing_counts
            else:
                self.chunk_faces_store[chunk_id] = face_counts
            logger.info(f"Added/Updated faces for chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding/updating faces for chunk {chunk_id}: {e}")
            return False

    def get_chunk_faces(self, chunk_id: str) -> dict:
        """Retrieve face counts for a given chunk."""
        try:
            if chunk_id in self.chunk_faces_store:
                face_counts = self.chunk_faces_store[chunk_id]
                logger.info(f"Retrieved faces for chunk {chunk_id}")
                return face_counts
            else:
                logger.info(f"No faces found for chunk {chunk_id}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving faces for chunk {chunk_id}: {e}")
            return {}
        
    def delete_chunk_faces(self, chunk_id: str) -> bool:
        """Delete face counts for a given chunk."""
        try:
            if chunk_id in self.chunk_faces_store:
                del self.chunk_faces_store[chunk_id]
                logger.info(f"Deleted faces for chunk {chunk_id}")
                return True
            else:
                logger.info(f"No faces to delete for chunk {chunk_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting faces for chunk {chunk_id}: {e}")
            return False