from .face_detector import FaceDetector
from .face_repository import FaceRepository
from .face_metadata_repository import FaceMetadataRepository
from .recent_faces_cache import RecentFacesCache, FaceData
import numpy as np
import logging
from typing import Literal
import cv2
from typing import List, Dict, Any, Optional
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# for each cluster, find the core samples and select up to 3 representatives based on proximity to the cluster centroid
def get_cluster_representatives_from_dbscan(
    embeddings,
    dbscan_model,
    face_record=None,
    top_k=3
):
    labels = dbscan_model.labels_
    core_indices = dbscan_model.core_sample_indices_

    core_mask = np.zeros(len(embeddings), dtype=bool)
    core_mask[core_indices] = True

    representatives = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)

    for label in unique_labels:
        cluster_mask = (labels == label)
        strong_mask = cluster_mask & core_mask
        cluster_indices = np.where(strong_mask)[0]

        if len(cluster_indices) < top_k:
            cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_embeddings = embeddings[cluster_indices]

        if len(cluster_embeddings) <= top_k:
            reps_idx = cluster_indices
        else:
            centroid = cluster_embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid)

            sims = cluster_embeddings @ centroid
            top_local = np.argsort(-sims)[:top_k]
            reps_idx = cluster_indices[top_local]

        rep_list = [
            (embeddings[idx], face_record[idx][0], int(idx))
            for idx in reps_idx
        ]

        representatives[label] = rep_list

    return representatives

class VideoFacePipeline:
    """Pipeline to process video frames for face recognition.

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

    def process_video_chunks(self, processed_chunks: List[Dict[str, Any]]):
        all_face_embeddings = []
        face_record = []

        face_access_id_map: dict[str, str] = {}
        created_vector_ids: list[str] = []
        created_image_ids: list[str] = []
        created_face_ids: list[str] = []

        for chunk in processed_chunks:
            # sample up to 8 frames evenly from the chunk for face processing
            frames = chunk.get('frames')
            sampled = []
            if frames is not None and len(frames) > 0:
                n = min(8, len(frames))
                if len(frames) <= n:
                    sampled = [frame.copy() for frame in frames]
                else:
                    idx = np.linspace(0, len(frames) - 1, n).astype(int)
                    sampled = [frames[i].copy() for i in idx]
            

            # Aggregate face mappings for the chunk (face_id -> img_access_id)
            face_map = {}
            for f in sampled:
                faces = self.face_detector.detect_and_embed(f)

                for face in faces:
                    # Normalize embedding safely (avoid division by zero)
                    emb = np.asarray(face.embedding, dtype=float)
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm > 0:
                        normalized_embedding = emb / emb_norm
                    else:
                        normalized_embedding = emb

                    all_face_embeddings.append(normalized_embedding)
                    face_record.append([chunk["chunk_id"], face.face_image])

        X = np.vstack(all_face_embeddings)
        # use cosine metric which often helps with normalized embeddings
        db = DBSCAN(eps=0.4, min_samples=3, metric='cosine').fit(X)
        representatives = get_cluster_representatives_from_dbscan(X, db, face_record=face_record, top_k=1)
        logger.info("DBSCAN clustering complete. labels=%s", db.labels_)

        class FaceEmbeddingData:
            def __init__(self, face_id: str, img_access_id: str, embedding: np.ndarray):
                self.face_id = face_id
                self.img_access_id = img_access_id
                self.embedding = embedding
        face_embedding_data_list: list[FaceEmbeddingData] = []
        # save the image of each representative
        for cluster_label, reps in representatives.items():
            for i, (embedding, chunk_id, original_idx) in enumerate(reps):
                face_img = face_record[original_idx][1]
                # img.save(f"cluster_representatives/cluster{cluster_label}_rep{i}_chunk{chunk_id}.png")

                detected_face_data = self.face_repository.get_face_identity(self.namespace, embedding)
                if detected_face_data is None:
                    # create new face record in repository
                    face_id = self.face_repository.generate_face_id()
                    face_img_identifier = self.face_repository.upload_face_image(
                        face_image=face_img,
                        face_id=face_id,
                    )
                    if face_img_identifier is None:
                        logger.error("VideoFacePipeline: Failed to upload face image for face_id=%s", face_id)
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
                        logger.error("VideoFacePipeline: Failed to store face metadata for face_id=%s", face_id)
                        # Rollback uploaded image to avoid leaving orphaned images in R2
                        deleted = self.face_repository.delete_face_image(face_img_identifier)
                        if deleted:
                            logger.info("VideoFacePipeline: Rolled back uploaded image for face_id=%s", face_id)
                        else:
                            logger.warning("VideoFacePipeline: Failed to delete uploaded image for face_id=%s; orphaned image may remain: %s", face_id, face_img_identifier)
                        
                        # Skip this face but continue processing others
                        continue
                    created_image_ids.append(face_img_identifier)
                    created_face_ids.append(face_id)
                    logger.info("Created new face_id=%s for cluster_label=%d", face_id, cluster_label)
                else:
                    face_id = detected_face_data.face_id
                    face_img_identifier = detected_face_data.img_access_id
                    logger.info("Found existing face_id=%s for cluster_label=%d", face_id, cluster_label)

                # save in face_embedding_data_list for later upsert to FaceRepository
                face_embedding_data_list.append(FaceEmbeddingData(face_id, face_img_identifier, embedding))

        # Upsert all face embeddings for the chunk to the FaceRepository and build the face_id -> img_access_id map
        for data in face_embedding_data_list:
            face_id = data.face_id
            face_img_identifier = data.img_access_id
            embedding = data.embedding
            # Use the FaceRepository upsert API (video_chunk_id, face_embedding)
            vector_id = self.face_repository.upsert_identified_face_embedding(
                namespace=self.namespace,
                face_id=face_id,
                img_access_id=face_img_identifier,
                video_chunk_id=chunk_id,
                face_embedding=embedding,
            )

            if vector_id:
                created_vector_ids.append(vector_id)
            else:
                logger.error("FrameFacePipeline: Failed to upsert face embedding for face_id=%s in chunk=%s", face_id, chunk_id)

            face_access_id_map[face_id] = face_img_identifier

        return face_access_id_map, created_vector_ids, created_image_ids, created_face_ids
