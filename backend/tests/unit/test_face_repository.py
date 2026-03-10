import numpy as np
import pytest
from unittest.mock import MagicMock

from face_recognition import face_repository
from face_recognition.face import Face


def _make_face(embedding_dim: int = 128) -> Face:
    embedding = np.zeros(embedding_dim, dtype=float)
    # simple white square RGB image
    img = (np.ones((16, 16, 3), dtype='uint8') * 255)
    return Face(embedding=embedding, face_image=img)


def test_get_face_identity_no_match(mock_pinecone_connector, mock_r2_connector):
    """When Pinecone returns no matches, get_face_identity should return None."""
    connector, mock_index, mock_client, mock_pine = mock_pinecone_connector
    # ensure the pinecone connector we inject returns no matches
    connector.query_chunks = MagicMock(return_value=[])

    r2_conn, _, _ = mock_r2_connector
    # construct repository with fixture-provided connectors
    repo = face_repository.FaceRepository(pinecone_connector=connector, r2_connector=r2_conn)
    f = _make_face()

    assert repo.get_face_identity('ns', f) is None

def test_get_face_identity_match_below_threshold():
    """When Pinecone returns a low-scoring match, get_face_identity should return None."""
    mock_pine = MagicMock()
    mock_pine.query_chunks.return_value = [
        {'score': 0.01, 'metadata': {'face_id': 'fid-123', 'img_access_id': 'img-abc'}}
    ]

    # construct repository with a local mocked pinecone connector
    repo = face_repository.FaceRepository(pinecone_connector=mock_pine, r2_connector=MagicMock())
    f = _make_face()

    assert repo.get_face_identity('ns', f) is None


def test_get_face_identity_match_success(mock_pinecone_connector, mock_r2_connector):
    """When Pinecone returns a high-scoring match, the returned DetectedFaceData should contain IDs."""
    connector, mock_index, mock_client, mock_pine = mock_pinecone_connector
    connector.query_chunks = MagicMock(return_value=[
        {'score': 0.9, 'metadata': {'face_id': 'fid-123', 'img_access_id': 'img-abc'}}
    ])

    r2_conn, _, _ = mock_r2_connector
    repo = face_repository.FaceRepository(pinecone_connector=connector, r2_connector=r2_conn)
    f = _make_face()

    result = repo.get_face_identity('ns', f)
    assert result is not None
    assert result.face_id == 'fid-123'
    assert result.img_access_id == 'img-abc'


def test_upload_face_image_success(mock_pinecone_connector, mock_r2_connector):
    """upload_face_image should call R2Connector.upload_image and return the identifier on success.

    Note: the class defines a helper to convert numpy->png; patch it to avoid exercising PIL in unit test.
    """
    # Use an injected serializer stub so tests do not rely on PIL behavior
    serializer = lambda img: b'PNGBYTES'

    r2_conn, mock_client, mock_boto3 = mock_r2_connector
    # stub upload_image to return the expected identifier
    r2_conn.upload_image = MagicMock(return_value=(True, 'r2-id-42'))

    # Provide a mocked pinecone connector for construction and pass serializer
    connector, _, _, _ = mock_pinecone_connector
    repo = face_repository.FaceRepository(pinecone_connector=connector, r2_connector=r2_conn, image_serializer=serializer)
    f = _make_face()

    identifier = repo.upload_face_image(f, face_id='my-face-1')
    assert identifier == 'r2-id-42'

    # verify upload_image was called with (bytes, filename, namespace)
    assert r2_conn.upload_image.called
    called_kwargs = r2_conn.upload_image.call_args[1]
    # upload_image is invoked with keyword args in the implementation
    assert isinstance(called_kwargs.get('image_data'), (bytes, bytearray))
    assert 'face_my-face-1' in called_kwargs.get('filename')
    assert called_kwargs.get('namespace') == 'face_images'


def test_upsert_identified_face_embedding_calls_pinecone(mock_pinecone_connector):
    """upsert_identified_face_embedding should call PineconeConnector.upsert_chunk with the provided metadata."""
    connector, mock_index, mock_client, mock_pine = mock_pinecone_connector
    connector.upsert_chunk = MagicMock(return_value=True)

    repo = face_repository.FaceRepository(pinecone_connector=connector, r2_connector=MagicMock())
    f = _make_face()

    ok = repo.upsert_identified_face_embedding(namespace='ns', face_id='face-1', img_access_id='img-1', video_chunk_id='chunk-x', face_embedding=f.embedding)
    assert ok is True

    # Verify upsert_chunk was called and metadata contains our keys
    assert connector.upsert_chunk.called
    called_kwargs = connector.upsert_chunk.call_args[1]
    assert 'metadata' in called_kwargs
    md = called_kwargs['metadata']
    assert md['face_id'] == 'face-1'
    assert md['img_access_id'] == 'img-1'
    assert md['chunk_id'] == 'chunk-x'
