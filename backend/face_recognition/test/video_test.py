import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pathlib import Path
import cv2
from deepface import DeepFace


from PIL import Image
import numpy as np

import numpy as np
# Import classes here
from preprocessing.preprocessor import Preprocessor
from face_recognition import FaceDetector
from sklearn.metrics.pairwise import cosine_similarity

video = "scene_0002__00h00m19s399__00h01m12s033.mp4"
# video = "whiplash.mp4"
# namespace = "video-test-namespace-2"


preprocessor = Preprocessor(min_chunk_duration=1.0, max_chunk_duration=10.0, scene_threshold=13.0)

face_detector = FaceDetector(
    detector_backend="mtcnn",
    embedding_model_name="ArcFace",

)

os.makedirs("frames", exist_ok=True)
os.makedirs("detected_faces", exist_ok=True)

threshold = 0.32

counter = 0
face_counter = 0

faces_count = {}
all_face_embeddings = []

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
            image = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            image.save(f"frames/frame{counter}.png")
            print("Processing frame ", counter)

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
                if len(all_face_embeddings) == 0:
                    max_similarity = -1
                else:
                    # stack previous embeddings into an array for vectorized similarity
                    try:
                        prev_matrix = np.stack(all_face_embeddings)
                        similarities = cosine_similarity([normalized_embedding], prev_matrix)[0]
                        max_similarity = float(np.max(similarities)) if similarities.size > 0 else -1
                    except Exception:
                        # fallback to iterative computation if stacking fails
                        max_similarity = -1

                print("Max similarity with existing faces:", max_similarity)
                if max_similarity < threshold:
                    print("new face in frame", counter)
                    image = Image.fromarray(face.face_image)
                    image.save(f"detected_faces/frame{counter}_face{face_counter}.png")
                    face_counter += 1

                all_face_embeddings.append(normalized_embedding)

            counter = counter + 1


print(f"Job Finished processing {video}")

print(faces_count)