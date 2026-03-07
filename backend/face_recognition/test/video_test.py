import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pathlib import Path
import cv2
from deepface import DeepFace
from typing import List


from PIL import Image
import numpy as np

import numpy as np
# Import classes here
from preprocessing.preprocessor import Preprocessor
from face_recognition import FaceDetector
from sklearn.metrics.pairwise import cosine_similarity

# video = "scene_0002__00h00m19s399__00h01m12s033.mp4"
# video = "whiplash.mp4"
video = "scene_0001__00h00m00s000__00h03m59s672.mp4"
# video = "scene_0002__00h00m18s320__00h00m37s759.mp4"
# namespace = "video-test-namespace-2"

preprocessor = Preprocessor(min_chunk_duration=1.0, max_chunk_duration=10.0, scene_threshold=13.0)

face_detector = FaceDetector(
    detector_backend="mtcnn",
    # detector_backend="yolov8n",
    # detector_backend="retinaface",
    embedding_model_name="ArcFace",
    # embedding_model_name="Facenet512",
)

# os.makedirs("frames", exist_ok=True)
os.makedirs("detected_faces", exist_ok=True)
os.makedirs("cluster_representatives", exist_ok=True)

threshold = 0.32

counter = 0
face_counter = 0

all_face_embeddings = []
face_record = []

with open(video, "rb") as video:
    video_bytes = video.read()

    processed_chunks = preprocessor.process_video_from_bytes(
        video_bytes=video_bytes,
        video_id="video-test-id",
        filename=video,
        hashed_identifier="test-hash-identifier",
    )

    # Embed frames and store in Pinecone
    print(f"Job Embedding and upserting {len(processed_chunks)} chunks")

    # Prepare chunk details for response (without frame arrays)
    chunk_details = []
    for chunk in processed_chunks:
        # sample up to 8 frames evenly from the chunk for face processing
        frames = chunk.get('frames')
        sampled = []
        if frames is not None and len(frames) > 0:
            n = min(8, len(frames))
            if len(frames) <= n:
                sampled = [frame.copy() for frame in frames]
            else:
                import numpy as _np
                idx = _np.linspace(0, len(frames) - 1, n).astype(int)
                sampled = [frames[i].copy() for i in idx]
        

        # Aggregate face mappings for the chunk (face_id -> img_access_id)
        face_map = {}
        for f in sampled:
            # image = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            # image.save(f"frames/frame{counter}.png")
            # print("Processing frame ", counter)

            faces = face_detector.detect_and_embed(f)
            print(f"Detected {len(faces)} faces in frame {counter}")

            for face in faces:
                # Normalize embedding safely (avoid division by zero)
                emb = np.asarray(face.embedding, dtype=float)
                emb_norm = np.linalg.norm(emb)
                if emb_norm > 0:
                    normalized_embedding = emb / emb_norm
                else:
                    normalized_embedding = emb

                # If there are no previous embeddings, treat as a new face
                # if len(all_face_embeddings) == 0:
                #     max_similarity = -1
                # else:
                #     # stack previous embeddings into an array for vectorized similarity
                #     try:
                #         prev_matrix = np.stack(all_face_embeddings)
                #         similarities = cosine_similarity([normalized_embedding], prev_matrix)[0]
                #         max_similarity = float(np.max(similarities)) if similarities.size > 0 else -1
                #     except Exception:
                #         # fallback to iterative computation if stacking fails
                #         max_similarity = -1

                # print("Max similarity with existing faces:", max_similarity)
                # if max_similarity < threshold:
                #     print("new face in frame", counter)
                #     image = Image.fromarray(face.face_image)
                #     image.save(f"detected_faces/frame{counter}_face{face_counter}.png")
                #     face_counter += 1

                image = Image.fromarray(face.face_image)
                # image.save(f"detected_faces/frame{counter}_face{face_counter}.png")
                all_face_embeddings.append(normalized_embedding)
                face_record.append([chunk["chunk_id"], image])

            counter = counter + 1


print(f"Job Finished processing {video}")


print(len(all_face_embeddings), type(all_face_embeddings[0]) if len(all_face_embeddings) > 0 else "N/A")

# cluster the embeddings using DBSCAN
from sklearn.cluster import DBSCAN
X = np.vstack(all_face_embeddings)
# use cosine metric which often helps with normalized embeddings
db = DBSCAN(eps=0.35, min_samples=3, metric='cosine').fit(X)
labels = db.labels_
print(labels)
# print(face_record)
for i, (chunk_id, image) in enumerate(face_record):
    cluster_id = labels[i]
    image.save(f"detected_faces/cluster{cluster_id}_{chunk_id}_{i}.png")

core_indices = db.core_sample_indices_
print(core_indices)

import numpy as np

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

representatives = get_cluster_representatives_from_dbscan(X, db, face_record=face_record, top_k=3)
# print(f"Cluster representatives (up to 3 per cluster): {representatives}")

# save the image of each representative
for cluster_label, reps in representatives.items():
    for i, (embedding, chunk_id, original_idx) in enumerate(reps):
        img = face_record[original_idx][1]
        img.save(f"cluster_representatives/cluster{cluster_label}_rep{i}_chunk{chunk_id}.png")