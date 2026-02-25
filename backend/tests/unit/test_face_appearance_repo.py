from unittest.mock import MagicMock

from face_recognition import (
    FaceAppearanceRepository,
)


def make_repo_with_collection(collection_mock):
    db = MagicMock()
    users_collection = MagicMock()
    user_doc = MagicMock()
    # chain: db.collection("users").document(user_id).collection("face_appearances") -> collection_mock
    db.collection.return_value = users_collection
    users_collection.document.return_value = user_doc
    user_doc.collection.return_value = collection_mock
    return FaceAppearanceRepository(db)


def test_set_face_appearance_success():
    collection_mock = MagicMock()
    doc_ref = MagicMock()
    collection_mock.document.return_value = doc_ref

    repo = make_repo_with_collection(collection_mock)

    result = repo.set_face_appearance("user1", "face1", "chunk1")
    
    assert result is True
    doc_ref.set.assert_called_once_with({
        "face_id": "face1",
        "video_chunk_id": "chunk1",
    })


def test_set_face_appearance_failure():
    collection_mock = MagicMock()
    doc_ref = MagicMock()
    doc_ref.set.side_effect = Exception("boom")
    collection_mock.document.return_value = doc_ref

    repo = make_repo_with_collection(collection_mock)

    result = repo.set_face_appearance("user1", "face1", "chunk1")

    assert result is False


def test_get_faces_for_chunk_returns_appearances():
    collection_mock = MagicMock()
    # create fake document objects returned by query.stream()
    doc = MagicMock()
    doc.id = "face1_chunk1"
    doc.to_dict.return_value = {"face_id": "face1", "video_chunk_id": "chunk1"}

    query_mock = MagicMock()
    query_mock.stream.return_value = [doc]
    collection_mock.where.return_value = query_mock

    repo = make_repo_with_collection(collection_mock)

    appearances = repo.get_faces_for_chunk("user1", "chunk1")

    assert isinstance(appearances, dict)
    assert "face1_chunk1" in appearances
    assert appearances["face1_chunk1"]["face_id"] == "face1"


def test_get_chunks_for_face_returns_appearances():
    collection_mock = MagicMock()

    doc = MagicMock()
    doc.id = "face1_chunk1"
    doc.to_dict.return_value = {"face_id": "face1", "video_chunk_id": "chunk1"}

    query_mock = MagicMock()
    query_mock.stream.return_value = [doc]
    collection_mock.where.return_value = query_mock

    repo = make_repo_with_collection(collection_mock)

    appearances = repo.get_chunks_for_face("user1", "face1")

    assert isinstance(appearances, dict)
    assert "face1_chunk1" in appearances
    assert appearances["face1_chunk1"]["video_chunk_id"] == "chunk1"


def test_get_faces_for_chunk_handles_exception():
    collection_mock = MagicMock()
    collection_mock.where.side_effect = Exception("boom")

    repo = make_repo_with_collection(collection_mock)

    result = repo.get_faces_for_chunk("user1", "chunk1")
    assert result is None


def test_delete_face_appearance_success():
    collection_mock = MagicMock()
    doc_ref = MagicMock()
    collection_mock.document.return_value = doc_ref

    repo = make_repo_with_collection(collection_mock)

    result = repo.delete_face_appearance("user1", "face1", "chunk1")
    assert result is True
    doc_ref.delete.assert_called_once()


def test_delete_face_appearance_failure():
    collection_mock = MagicMock()
    doc_ref = MagicMock()
    doc_ref.delete.side_effect = Exception("boom")
    collection_mock.document.return_value = doc_ref

    repo = make_repo_with_collection(collection_mock)

    result = repo.delete_face_appearance("user1", "face1", "chunk1")
    assert result is False


def test_update_face_appearance_success():
    collection_mock = MagicMock()
    doc_ref = MagicMock()
    collection_mock.document.return_value = doc_ref

    repo = make_repo_with_collection(collection_mock)

    result = repo.update_face_appearance("user1", "face1", "chunk1", {"foo": "bar"})
    assert result is True
    doc_ref.update.assert_called_once_with({"foo": "bar"})


def test_update_face_appearance_failure():
    collection_mock = MagicMock()
    doc_ref = MagicMock()
    doc_ref.update.side_effect = Exception("boom")
    collection_mock.document.return_value = doc_ref

    repo = make_repo_with_collection(collection_mock)

    result = repo.update_face_appearance("user1", "face1", "chunk1", {"foo": "bar"})
    assert result is False
