"""Face detector: detection and embedding faces from images.

This module provides the FaceDetector class which wraps face detection
and embedding (via DeepFace) 

Responsibilities:
- detect faces in images and compute embeddings.
"""

# from incdbscan import IncrementalDBSCAN
from deepface import DeepFace
import numpy as np
# from sklearn.cluster import *
from .face import Face
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    """Class used to detect faces and compute embeddings

    Attributes:
        detector_backend (str): Face detection backend to use.
        embedding_model_name (str): Name of the embedding model.
        enforce_detection (bool): Whether to enforce that at least one face is detected.
        align (bool): Whether to align faces before embedding.
    Class attributes:
        all_detector_backends (list[str]): List of all supported detector backends.
        all_embed_models (list[str]): List of all supported embedding models.
    """

    # list of all detector backends used for face detection
    all_detector_backends = [
        "mtcnn"
    ]

    # list of embedding models (informational)
    all_embed_models = [
        "ArcFace"
    ]

    def __init__(
            self, 
            detector_backend="mtcnn", 
            embedding_model_name="ArcFace", 
            enforce_detection=True, 
            align=True):
        # parameters for face detection and embedding with deepface
        self.detector_backend = detector_backend
        self.embedding_model_name = embedding_model_name
        self.enforce_detection = enforce_detection
        self.align = align
        
        logging.debug(f"FaceDetector: Initialized FaceDetector with detector_backend={detector_backend}, "
                     f"embedding_model_name={embedding_model_name}, enforce_detection={enforce_detection}, "
                     f"align={align}")

    def detect_and_embed(self, img) -> list[Face]:
        """Detect faces in `img` and compute embeddings.

        Args:
            img (str | np.ndarray): Path to the image or an image as a NumPy array.

        Returns:
            list[Face]: A list of Face objects. Each Face contains the embedding
                (np.ndarray) and the cropped face image as a NumPy array.

        Notes:
            This method calls ``DeepFace.represent`` which returns a list of
            dictionaries with keys such as ``embedding`` and ``facial_area``.
            We convert each result into a ``Face`` using
            ``Face.from_original_image`` with the reported bounding box.
        """
        faces: list[Face] = []

        try:
            rep = DeepFace.represent(
                img_path=img,
                model_name=self.embedding_model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=self.align,
            )
        except ValueError:
            logger.warning(f"FaceDetector: No face detected in image {img}\nreturning empty face list.")
            return []

        for r in rep:
            try:
                face = Face.from_original_image(
                    embedding=np.array(r["embedding"]),
                    orig_image=img,
                    bbox=(
                        r["facial_area"]["x"],
                        r["facial_area"]["y"],
                        r["facial_area"]["w"],
                        r["facial_area"]["h"],
                    ),
                )
            except Exception as e:
                logger.error(f"FaceDetector: Error creating Face object from representation {r} on image {img}: {e}\nskipping this face.")
                continue
            faces.append(face)

        return faces