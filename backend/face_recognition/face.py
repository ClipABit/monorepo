import numpy as np
import cv2

class Face:
    def __init__(self, embedding: np.ndarray, face_image: np.ndarray):
        self.embedding = embedding
        self.face_image = face_image

    @classmethod
    def from_original_image(cls, embedding: np.ndarray, orig_image: np.ndarray | str, bbox: tuple[int, int, int, int]):
        if (type(orig_image) == str):
            image_np = cv2.imread(orig_image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif (type(orig_image) == np.ndarray):
            image_np = orig_image
        else:
            raise ValueError("orig_image must be either a file path (str) or a numpy ndarray.")

        x, y, w, h = bbox

        # Crop (OpenCV uses NumPy slicing)
        try:
            face_image = image_np[y:y+h, x:x+w]
        except Exception as e:
            raise ValueError(f"Error cropping face image with bbox {bbox}: {e}")
        return cls(embedding, face_image)