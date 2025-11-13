"""Face repository: detection, embedding and incremental clustering utilities.

This module provides the FaceRepository class which wraps face detection
and embedding (via DeepFace) and incremental clustering (via IncrementalDBSCAN).

Responsibilities:
- Detect faces in images and compute embeddings.
- Maintain a global list of embeddings and an incremental clustering model.
- Map clip IDs to the clusters (face identities) observed in that clip.
- Provide utilities to retrieve example face crops per cluster and per-clip face images.

The implementation stores example face crops in `cluster_example_face` so the UI
can render representative face images for each cluster.
"""

from incdbscan import IncrementalDBSCAN
from deepface import DeepFace
import numpy as np
from sklearn.cluster import *
from .face import Face
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRepository:
    """Repository that detects faces, computes embeddings and clusters them.

    Attributes:
        detector_backend (str): DeepFace detector backend name.
        embedding_model_name (str): Embedding model name used by DeepFace.
        enforce_detection (bool): Whether DeepFace should enforce detection.
        align (bool): Whether to align faces before embedding.
        all_embeddings (list[np.ndarray]): Flattened list of all stored embeddings.
        clip_faces_map (dict[int, set[int]]): Map from clip_id to set of cluster labels seen in that clip.
        clustering (IncrementalDBSCAN): Incremental clustering model used to assign embeddings to clusters.
        cluster_example_face (dict[int, Face]): Representative Face object for each cluster .

    Class attributes:
        all_detector_backends (list[str]): Supported detector backends (informational).
        all_embed_models (list[str]): Supported embedding models (informational).
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
            align=True,
            cluster_metric = "cosine",
            cluster_eps = 0.6,
            cluster_min_pts = 2):
        # parameters for face detection and embedding with deepface
        self.detector_backend = detector_backend
        self.embedding_model_name = embedding_model_name
        self.enforce_detection = enforce_detection
        self.align = align

        self.all_embeddings: list[np.ndarray] = []
        self.clip_faces_map: dict[int, set[int]] = {}    # map from clip_id to set of face indices in that clip
        self.clustering = IncrementalDBSCAN(metric=cluster_metric, eps=cluster_eps, min_pts = cluster_min_pts)

        self.cluster_example_face: dict[int, Face] = {} # list of example face images in cluster = key

        logging.debug(f"FaceRepository: Initialized FaceRepository with detector_backend={detector_backend}, "
                     f"embedding_model_name={embedding_model_name}, enforce_detection={enforce_detection}, "
                     f"align={align}, cluster_metric={cluster_metric}, cluster_eps={cluster_eps}, "
                     f"cluster_min_pts={cluster_min_pts}")

    def __detect_and_embed(self, img):
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
        except Exception as e:
            logger.error(f"FaceRepository: Error during face detection and embedding on image {img}: {e}\nreturning empty face list.")
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
                logger.error(f"FaceRepository: Error creating Face object from representation {r} on image {img}: {e}\nskipping this face.")
                continue
            faces.append(face)

        return faces

    # add a list of faces to the cluster
    def add_faces(self, clip_id: int, faces: list[Face]):
        """Add a batch of Face objects to the clustering model.

        This method inserts the provided embeddings into the incremental
        clustering model, updates the global embedding list, and records which
        clusters appear in `clip_id`.

        Args:
            clip_id (int): Identifier for the video/audio clip the images belong to.
            faces (list[Face]): List of Face objects (must contain `.embedding`).

        Returns:
            list[int]: A list of cluster labels for the provided faces. Labels
                are integers >= 0 for cluster assignments. (Outliers may be
                filtered out by the clustering model and not returned here.)
        """
        logging.debug(f"FaceRepository: Adding {len(faces)} faces to clustering for clip_id {clip_id}.")

        try:
            # collect and stack embeddings from face objects
            face_embeddings = [f.embedding for f in faces]
            X = np.stack(face_embeddings)
        except Exception as e:
            logger.error(f"FaceRepository: Error extracting embeddings from faces for clip_id {clip_id}: {e}\nreturning empty label list.")
            return []

        try:
            self.clustering.insert(X)
        except Exception as e:
            logger.error(f"FaceRepository: Error inserting embeddings into clustering model for clip_id {clip_id}: {e}\nreturning empty label list.")
            return []

        # Persist embeddings and retrieve labels for the inserted batch
        self.all_embeddings += face_embeddings
        labels = [int(l) for l in self.clustering.get_cluster_labels(X)]

        # Update mapping from clip_id to set of seen cluster labels
        if clip_id not in self.clip_faces_map:
            self.clip_faces_map[clip_id] = set([l for l in labels if l > -1])
        else:
            self.clip_faces_map[clip_id].update([l for l in labels if l > -1])

        # Ensure we have a representative Face object for any newly created cluster
        for i, label in enumerate(labels):
            if label not in self.cluster_example_face.keys() and label > -1:
                self.cluster_example_face[label] = faces[i]
                logging.debug(f"FaceRepository: New face detected with cluster label {label}. Stored example face.")

        logging.debug(f"FaceRepository: Added {len(faces)} faces for clip_id {clip_id}, assigned labels: {labels}. Outlier faces (label -1) will be ignored.")
        return labels
    
    # get all face cluster labels in a given clip
    def get_faces_in_clip(self, clip_id: int):
        """Return the set of cluster labels observed in a clip.

        Args:
            clip_id (int): Clip identifier.

        Returns:
            set[int]: A set of cluster labels (may be empty if no faces observed).
        """
        if clip_id not in self.clip_faces_map:
            logging.warning(f"FaceRepository: No faces recorded for clip_id {clip_id}. Empty set will be returned.")
        return self.clip_faces_map.get(clip_id, set())
        
    def get_face_images_in_clip(self, clip_id: int):
        """Retrieve example face crop images for all clusters in a clip.

        Args:
            clip_id (int): Clip identifier.

        Returns:
            list[np.ndarray]: List of face crop images ( NumPy arrays ). The
                order is determined by the set iteration over cluster labels.
        """
        face_images = []
        if (clip_id not in self.clip_faces_map):
            logging.warning(f"FaceRepository: No faces recorded for clip_id {clip_id}. Empty face image list will be returned.")
            return []
        
        face_labels = self.clip_faces_map.get(clip_id, set())
        for label in face_labels:
            if label in self.cluster_example_face.keys():
                face_images.append(self.cluster_example_face[label].face_image)
            else:
                logging.warning(f"FaceRepository: Cluster label {label} for clip_id {clip_id} has no example face stored.")
        return face_images

    def add_images(self, clip_id: int, img_lst: list):
        """Detect faces and add their embeddings for a list of images.

        This is a convenience wrapper that runs detection+embedding for each
        image in ``img_lst`` and then calls :meth:`add_faces` to insert the
        resulting embeddings into the clustering model.

        Args:
            clip_id (int): Clip identifier to associate the detected faces with.
            img_lst (list[str|np.ndarray]): List of image paths or NumPy arrays.

        Returns:
            list[int]: Cluster labels assigned to the detected faces.
        """
        logging.debug(f"FaceRepository: Adding frame images for clip_id {clip_id} for facial recognition, number of images: {len(img_lst)}")
        if not img_lst:
            logging.warning(f"FaceRepository: empty img_lst provided for clip_id {clip_id}")

        embedded_faces = []
        for img in img_lst:
            faces = self.__detect_and_embed(img)
            embedded_faces += faces

        face_cluster = self.add_faces(clip_id, embedded_faces)
        logging.debug(f"FaceRepository: Completed processing {len(img_lst)} images for clip_id {clip_id}")

        return face_cluster