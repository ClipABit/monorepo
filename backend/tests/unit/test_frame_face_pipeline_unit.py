import numpy as np
from unittest.mock import MagicMock

from face_recognition.frame_face_pipeline import FrameFacePipeline


class SimpleFace:
    def __init__(self, embedding, img=None):
        self.embedding = embedding
        self.img = img


def test_process_frame_with_no_faces():
    detector = MagicMock()
    detector.detect_and_embed.return_value = []

    repo = MagicMock()
    pipeline = FrameFacePipeline(namespace='ns', face_detector=detector, face_repository=repo)

    res = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), chunk_id='c1')
    assert res == {}
    detector.detect_and_embed.assert_called_once()
    repo.get_face_identity.assert_not_called()

def test_process_frame_with_existing_face():
    # Create a fake face embedding and image
    fake_embedding = np.ones(128, dtype=float)
    fake_face = SimpleFace(embedding=fake_embedding, img=b'IMG')

    detector = MagicMock()
    detector.detect_and_embed.return_value = [fake_face]

    repo = MagicMock()
    # Simulate existing face in pinecone
    repo.get_face_identity.return_value = MagicMock(face_id='face-456', img_access_id='r2-888')

    pipeline = FrameFacePipeline(namespace='ns', face_detector=detector, face_repository=repo)

    out = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), chunk_id='chunk-1')

    assert out == {'face-456': 'r2-888'}
    repo.get_face_identity.assert_called_once()
    repo.upload_face_image.assert_not_called()
    repo.upsert_identified_face_embedding.assert_called_once()


def test_process_frame_with_new_face():
    # Create a fake face embedding and image
    fake_embedding = np.ones(128, dtype=float)
    fake_face = SimpleFace(embedding=fake_embedding, img=b'IMG')

    detector = MagicMock()
    detector.detect_and_embed.return_value = [fake_face]

    repo = MagicMock()
    # No match in pinecone
    repo.get_face_identity.return_value = None
    repo.generate_face_id.return_value = 'face-123'
    repo.upload_face_image.return_value = 'r2-999'
    repo.upsert_identified_face_embedding.return_value = True

    pipeline = FrameFacePipeline(namespace='ns', face_detector=detector, face_repository=repo)

    out = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), chunk_id='chunk-1')

    assert out == {'face-123': 'r2-999'}
    repo.generate_face_id.assert_called_once()
    repo.upload_face_image.assert_called_once()
    repo.upsert_identified_face_embedding.assert_called_once()
