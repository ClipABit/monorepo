from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class FaceData:
    face_id: str
    embedding: np.ndarray
    img_access_id: str

class RecentFacesCache:
    def __init__(self, max_size: int = 100, confidente_threshold: float = 0.9):
        self.max_size = max_size
        self.confidente_threshold = confidente_threshold
        self.cache: Dict[str, FaceData] = {}
        self.order: List[str] = []

    def add_face(self, face_id: str, face_data: FaceData):
        if face_id in self.cache:
            self.order.remove(face_id)
        elif len(self.cache) >= self.max_size:
            oldest_face_id = self.order.pop(0)
            del self.cache[oldest_face_id]

        self.cache[face_id] = face_data
        self.order.append(face_id)

    def get_face(self, face_id: str) -> Optional[FaceData]:
        return self.cache.get(face_id, None)

    def query_similar_face(self, embedding: np.ndarray, similarity_funct: Literal["cosine"] = "cosine") -> Optional[tuple[FaceData, float]]:
        if not self.cache:
            return None

        ordered_faces = [self.cache[face_id] for face_id in self.order]
        embeddings = np.stack([fd.embedding for fd in ordered_faces])

        if similarity_funct == "cosine":
            similarities = cosine_similarity([embedding], embeddings)[0]
        else:
            raise ValueError(f"Unsupported similarity function: {similarity_funct}")

        max_index = np.argmax(similarities)
        max_similarity = similarities[max_index]

        if max_similarity >= self.confidente_threshold:
            face_id = self.order[max_index]
            return self.cache[face_id], max_similarity
        else:
            return None

    def clear(self):
        self.cache.clear()
        self.order.clear()