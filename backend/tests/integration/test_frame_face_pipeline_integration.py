import numpy as np
from unittest.mock import MagicMock

from face_recognition.frame_face_pipeline import FrameFacePipeline
from face_recognition.face_repository import FaceRepository


class SimpleFace:
    def __init__(self, embedding, img=None):
        self.embedding = embedding
        self.img = img


def test_pipeline_integration_new_and_existing(mock_pinecone_connector, mock_r2_connector):
    # Unpack fixtures
    connector, mock_index, mock_client, mock_pine = mock_pinecone_connector
    r2_conn, mock_r2_client, mock_boto3 = mock_r2_connector

    # Create a face repository wired to mocked connectors
    serializer = lambda img: b'PNGBYTES'
    repo = FaceRepository(pinecone_connector=connector, r2_connector=r2_conn, image_serializer=serializer)

    # Prepare two faces: first is existing (pinecone match), second is new
    face_existing = SimpleFace(embedding=np.array([0.1] * 128), img=b'EXIST')
    face_new = SimpleFace(embedding=np.array([0.9] * 128), img=b'NEW')

    detector = MagicMock()
    detector.detect_and_embed.return_value = [face_existing, face_new]

    # Configure connector.query_chunks to return a match for the first face and no match for the second
    def query_side_effect(query_embedding, namespace, top_k=1):
        # Simple check on first value to decide
        if float(query_embedding[0]) == 0.1:
            return [{'score': 0.95, 'metadata': {'face_id': 'existing-id', 'img_access_id': 'r2-exist'}}]
        return []
    connector.query_chunks = MagicMock(side_effect=query_side_effect)

    # r2 upload returns new id for new face
    r2_conn.upload_image = MagicMock(return_value=(True, 'r2-new'))

    # The repo.upload_face_image implementation now matches the pipeline's
    # public API and uses the injected `r2_conn` mock. No monkeypatching required.

    # Make connector.upsert_chunk a MagicMock so we can assert call counts
    connector.upsert_chunk = MagicMock(return_value=True)

    # The repo.upsert_identified_face_embedding implementation now matches the
    # pipeline's expected parameter names (video_chunk_id, face_embedding) and
    # delegates to the injected `connector` mock. No monkeypatching required.

    pipeline = FrameFacePipeline(namespace='ns', face_detector=detector, face_repository=repo)

    mapping = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), chunk_id='c-1')

    # Expect existing-id present 
    assert 'existing-id' in mapping
    assert mapping['existing-id'] == 'r2-exist'

    # Expect new face mapping returned and mapped to r2-new
    assert 'r2-new' in mapping.values()
    assert len(mapping) == 2

    # Upload should have been called once for the new face
    r2_conn.upload_image.assert_called_once()
    # Upsert should have been called twice (once per face)
    assert connector.upsert_chunk.call_count == 2

def test_pipeline_integration_two_same_faces(mock_pinecone_connector, mock_r2_connector):
    # Two identical faces in the same frame should dedupe to a single mapping entry.
    connector, mock_index, mock_client, mock_pine = mock_pinecone_connector
    r2_conn, mock_r2_client, mock_boto3 = mock_r2_connector

    serializer = lambda img: b'PNGBYTES'
    repo = FaceRepository(pinecone_connector=connector, r2_connector=r2_conn, image_serializer=serializer)

    # Both faces have the same embedding
    emb = np.array([0.2] * 128)
    face_a = SimpleFace(embedding=emb, img=b'A')
    face_b = SimpleFace(embedding=emb, img=b'A')

    detector = MagicMock()
    detector.detect_and_embed.return_value = [face_a, face_b]

    # First query returns no match (new face). After the first upsert we record
    # the face_id and make the second query return that same face_id so the
    # pipeline deduplicates both detections to a single identity.
    call = {'n': 0}
    stored = {'face_id': None}

    def query_side_effect(query_embedding, namespace, top_k=1):
        call['n'] += 1
        if call['n'] == 1:
            return []
        # return match using the face_id captured from the upsert
        return [{'score': 0.95, 'metadata': {'face_id': stored['face_id'], 'img_access_id': 'r2-same'}}]

    connector.query_chunks = MagicMock(side_effect=query_side_effect)

    # r2 upload should only be called once for the new face
    r2_conn.upload_image = MagicMock(return_value=(True, 'r2-same'))

    # Capture face_id from the upsert metadata so query can return it on second call
    def upsert_side_effect(chunk_id, chunk_embedding, namespace, metadata):
        stored['face_id'] = metadata.get('face_id')
        return True

    connector.upsert_chunk = MagicMock(side_effect=upsert_side_effect)

    pipeline = FrameFacePipeline(namespace='ns', face_detector=detector, face_repository=repo)

    mapping = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), chunk_id='c-1')

    # Mapping should contain a single entry and contain the r2 id
    assert isinstance(mapping, dict)
    assert len(mapping) == 1
    assert 'r2-same' in mapping.values()

    r2_conn.upload_image.assert_called_once()
    assert connector.upsert_chunk.call_count >= 1

