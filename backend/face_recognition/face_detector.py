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
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define paths and options
model_path = '/Users/yifanzhang/workspace/ClipABit/monorepo/backend/face_recognition/face_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for np.ndarray
    output_facial_transformation_matrixes=True, # Required for frontal score
    num_faces=1
)

# Create the landmarker instance
landmarker_instance = vision.FaceLandmarker.create_from_options(options)


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
        "mtcnn",
        "retinaface",
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
        align=True,
        # quality thresholds
        min_face_size: int = 48,
        min_area_ratio: float = 0.002,
        max_aspect_ratio: float = 3.0):
        # parameters for face detection and embedding with deepface
        self.detector_backend = detector_backend
        self.embedding_model_name = embedding_model_name
        self.enforce_detection = enforce_detection
        self.align = align
        # quality thresholds used to filter out bad faces
        self.min_face_size = min_face_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio

        logging.debug(
            f"FaceDetector: Initialized FaceDetector with detector_backend={detector_backend}, "
            f"embedding_model_name={embedding_model_name}, enforce_detection={enforce_detection}, "
            f"align={align}"
        )

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

        # determine original image dimensions for area-based checks (best-effort)
        orig_h = orig_w = None
        try:
            if isinstance(img, str):
                _img = cv2.imread(img)
                if _img is not None:
                    orig_h, orig_w = _img.shape[:2]
            elif isinstance(img, np.ndarray):
                orig_h, orig_w = img.shape[:2]
        except Exception:
            orig_h = orig_w = None

        try:
            rep = DeepFace.represent(
                img_path=img,
                model_name=self.embedding_model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=self.align,
            )
        except ValueError as e:
            logger.warning(f"FaceDetector: No face detected in image: {e}\nreturning empty face list.")
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
            # quality filtering: drop faces that are too small, too blurry,
            # or have an extreme aspect ratio
            try:
                if not self._is_good_quality(face, orig_h, orig_w):
                    logger.info("FaceDetector: Skipping low-quality face.")
                    continue
            except Exception as e:
                logger.warning(f"FaceDetector: Quality check failed for face: {e} -- keeping face by default")

            faces.append(face)

        logger.info(f"FaceDetector: Detected {len(faces)} face(s) in image.")
        return faces

    def _frontal_score(self, image_array: np.ndarray):
        """
        Inputs: 
        - image_array: NumPy array (BGR)
        - landmarker_instance: An initialized MediaPipe FaceLandmarker
        Returns:
        - float: Frontal deviation score (0 is perfectly frontal)
        """
        # 1. Convert BGR to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # 2. Convert NumPy array to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 3. Perform detection (using IMAGE mode for single np arrays)
        result = landmarker_instance.detect(mp_image)

        if not result.facial_transformation_matrixes:
            return None # No face detected

        # 4. Extract 4x4 matrix and decompose
        matrix = result.facial_transformation_matrixes[0]
        
        # Extract rotation components
        pitch = np.arcsin(-matrix[1][2])
        yaw = np.arctan2(matrix[0][2], matrix[2][2])
        roll = np.arctan2(matrix[1][0], matrix[1][1])

        # Convert to absolute degrees and sum
        total_score = abs(np.degrees(pitch)) + abs(np.degrees(yaw)) + abs(np.degrees(roll))
        
        return total_score

    def _is_good_quality(self, face: Face, orig_h: int | None, orig_w: int | None) -> bool:
        """Return True if the face passes basic quality checks.

        Checks performed:
        - face image present and not empty
        - width and height greater than min_face_size
        - area ratio relative to original image greater than min_area_ratio (if orig dims available)
        - aspect ratio within reasonable bounds
        """
        if face.face_image is None:
            logger.warning("FaceDetector: Face image is None, rejecting face.")
            return False

        h_w = face.face_image.shape[:2]
        if len(h_w) < 2:
            logger.warning("FaceDetector: Face image has invalid shape, rejecting face.")
            return False
        h, w = h_w

        # size checks
        if w < self.min_face_size or h < self.min_face_size:
            logger.warning(f"FaceDetector: Face size too small (w={w}, h={h}), rejecting face.")
            return False
        
        # frontal check
        score = self._frontal_score(face.face_image)
        if score is None or score > 25: # threshold is somewhat arbitrary, based on manual testing
            logger.warning(f"FaceDetector: Face does not appear frontal enough with score {score}, rejecting face.")
            return False

        # area ratio check (if original dims are known)
        try:
            if orig_h and orig_w:
                area_ratio = (w * h) / (orig_h * orig_w)
                if area_ratio < self.min_area_ratio:
                    logger.warning(f"FaceDetector: Face area ratio too small ({area_ratio:.6f}), rejecting face.")
                    return False
        except Exception:
            # don't fail on computation error
            pass

        # aspect ratio check
        ar = max(w / max(h, 1), h / max(w, 1))
        if ar > self.max_aspect_ratio:
            logger.warning(f"FaceDetector: Face aspect ratio too extreme ({ar:.2f}), rejecting face.")
            return False

        return True