"""Face repository: detection, embedding and incremental clustering utilities.

This module provides the FaceRepository class which wraps face detection
and embedding (via DeepFace) and clustring (via Nearest-Neighbor + Threshold).

Responsibilities:
- detect faces in images and compute embeddings.
- for each face, find the face_id in database that it belongs to (or create a new face_id).
- add new face embeddings to Pinecone vector database, with metadata for face ID and chunk ID.
"""

# from incdbscan import IncrementalDBSCAN
from deepface import DeepFace
import numpy as np
# from sklearn.cluster import *
from .face import Face
import logging
from database import PineconeConnector
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRepository:
    """Repository that detects faces, computes embeddings and clusters them.

    Attributes:
        pinecone_connector (PineconeConnector): Connector to Pinecone vector database.
        detector_backend (str): Backend to use for face detection (default: "mtcnn").
        embedding_model_name (str): Model name to use for face embedding (default: "ArcFace").
        enforce_detection (bool): Whether to enforce face detection (default: True).
        align (bool): Whether to align faces before embedding (default: True).
        threshold (float): Similarity threshold for face recognition (default: 0.35).
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
            pinecone_api_key: str,
            index_name: str,
            detector_backend="mtcnn", 
            embedding_model_name="ArcFace", 
            enforce_detection=True, 
            align=True,
            threshold = 0.35):
        # parameters for face detection and embedding with deepface
        self.detector_backend = detector_backend
        self.embedding_model_name = embedding_model_name
        self.enforce_detection = enforce_detection
        self.align = align
        self.threshold = threshold  # threshold for face recognition matching

        self.pinecone_connector = PineconeConnector(api_key=pinecone_api_key, index_name=index_name)
        
        self.cluster_example_face: dict[int, Face] = {} # list of example face images in cluster = key

        logging.debug(f"FaceRepository: Initialized FaceRepository with detector_backend={detector_backend}, "
                     f"embedding_model_name={embedding_model_name}, enforce_detection={enforce_detection}, "
                     f"align={align}, index_name={index_name}, threshold={threshold}")

    def _detect_and_embed(self, img):
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

    def _upsert_face_embedding(self, face_ids_count: dict, namespace: str, face_id: str, chunk_id: str, face_embedding: np.ndarray):
        """Upsert a face embedding into the Pinecone index.

        Args:
            face_ids_count (dict): Dictionary mapping face IDs to their counts.
            namespace (str): Namespace to upsert the face embedding into.
            face_id (str): Unique identifier for the face.
            chunk_id (str): Unique identifier for the clip chunk.
            face_embedding (np.ndarray): The face embedding to upsert.

        Returns:
            bool: True if upsert was successful, False otherwise.
        """
        try:
            success = self.pinecone_connector.upsert_chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_embedding=face_embedding,
                namespace=namespace,
                metadata={"face_id": face_id, "chunk_id": chunk_id}
            )
        except Exception as e:
            logger.error(f"FaceRepository: Error upserting face embedding for face_id {face_id} in chunk_id {chunk_id}: {e}")
            return False
        
        if success:
            if face_id in face_ids_count:
                face_ids_count[face_id] += 1
            else:
                face_ids_count[face_id] = 1
        return success

    # add a list of faces to the cluster
    def add_faces(self, namespace: str, chunk_id: str, faces: list[Face]):
        """Add a batch of Face objects to the clustering model.

        This method inserts the provided embeddings into the incremental
        clustering model, updates the global embedding list, and records which
        clusters appear in `clip_id`.

        Args:
            namespace (str): Namespace to upsert the face embeddings into.
            chunk_id (str): Unique identifier for the clip chunk.
            faces (list[Face]): List of Face objects to add.

        Returns:
            dict: A dictionary mapping face IDs to the number of times they
                appear in this chunk.
        """
        logging.debug(f"FaceRepository: Adding {len(faces)} faces to clustering for chunk_id {chunk_id}.")

        try:
            # collect and stack embeddings from face objects
            face_embeddings = [f.embedding for f in faces]
        except Exception as e:
            logger.error(f"FaceRepository: Error extracting embeddings from faces for chunk_id {chunk_id}: {e}\nreturning empty label list.")
            return []
        
        # dict where key = face_id, value = number of times appear in this chunk
        face_ids_count: dict = {}
        
        for e in face_embeddings:
            # find closest face from pinecone vector db
            best_match = self.pinecone_connector.query_chunks(
                query_embedding=e,
                namespace=namespace,
                top_k=1
            )
            print(best_match)
            if not best_match or len(best_match) == 0:
                # no match found, insert as new cluster
                new_id = str(uuid.uuid4())
                upsert_success = self._upsert_face_embedding(
                    face_ids_count=face_ids_count,
                    namespace=namespace,
                    face_id=new_id,
                    chunk_id=chunk_id,
                    face_embedding=e
                )
                if not upsert_success:
                    logger.error(f"FaceRepository: Failed to upsert new face embedding for new_id {new_id} in chunk_id {chunk_id}.")
                    continue
                continue

            best_match = best_match[0]

            # if score above threshold, group new embedding into existing cluster
            if best_match['score'] > self.threshold:
                face_id = str(best_match["metadata"].get("face_id", None))
                print(face_id)
                if face_id is not None:
                    upsert_success = self._upsert_face_embedding(
                        face_ids_count=face_ids_count,
                        namespace=namespace,
                        face_id=face_id,
                        chunk_id=chunk_id,
                        face_embedding=e
                    )
                    if not upsert_success:
                        logger.error(f"FaceRepository: Failed to upsert face embedding for existing face_id {face_id} in chunk_id {chunk_id}.")
                        continue
                else:
                    logger.error(f"FaceRepository: Best match from Pinecone for chunk_id {chunk_id} has no face_id in metadata. Skipping.")
                    continue
            else:
                # otherwise, insert as new cluster
                new_id = str(uuid.uuid4())
                upsert_success = self._upsert_face_embedding(
                    face_ids_count=face_ids_count,
                    namespace=namespace,
                    face_id=new_id,
                    chunk_id=chunk_id,
                    face_embedding=e
                )
                if not upsert_success:
                    logger.error(f"FaceRepository: Failed to upsert new face embedding for new_id {new_id} in chunk_id {chunk_id}.")
                    continue
        
        return face_ids_count

    def add_images(self, namespace: str, chunk_id: str, img_lst: list):
        """Detect faces and add their embeddings for a list of images.

        This is a convenience wrapper that runs detection+embedding for each
        image in ``img_lst`` and then calls :meth:`add_faces` to insert the
        resulting embeddings into the clustering model.

        Args:
            clip_id (int): Clip identifier to associate the detected faces with.
            img_lst (list[str|np.ndarray]): List of image paths or NumPy arrays.

        Returns:
            dict: A dictionary mapping face IDs to the number of times they
                appear in this chunk.
        """
        logging.debug(f"FaceRepository: Adding frame images for chunk_id {chunk_id} for facial recognition, number of images: {len(img_lst)}")
        if not img_lst:
            logging.warning(f"FaceRepository: empty img_lst provided for chunk_id {chunk_id}")

        embedded_faces = []
        for img in img_lst:
            faces = self._detect_and_embed(img)
            embedded_faces += faces

        face_cluster = self.add_faces(namespace, chunk_id, embedded_faces)
        logging.debug(f"FaceRepository: Completed processing {len(img_lst)} images for chunk_id {chunk_id}")

        return face_cluster