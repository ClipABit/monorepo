"""
Extensive tests for UserStoreConnector quota, namespace pool, and video registration features.
"""

import pytest
from unittest.mock import MagicMock

from database.firebase.user_store_connector import UserStoreConnector


@pytest.fixture
def mock_firestore():
    """Mock Firestore client with collection/document chain."""
    return MagicMock()


@pytest.fixture
def connector(mock_firestore):
    """UserStoreConnector with mocked Firestore client."""
    conn = UserStoreConnector(firestore_client=mock_firestore)
    # Patch _assign_namespace for tests that create users
    conn._assign_namespace = MagicMock(return_value="ns_00")
    return conn


def _mock_doc(exists: bool, data: dict = None):
    """Helper to create a mock Firestore document snapshot."""
    doc = MagicMock()
    doc.exists = exists
    doc.to_dict.return_value = data
    return doc


# =============================================================================
# Namespace Pool Assignment Tests
# =============================================================================


class TestAssignNamespace:
    """Tests for the pool-based namespace assignment."""

    def test_picks_namespace_with_most_remaining_capacity(self, mock_firestore):
        """Assigns to the namespace with lowest vector count."""
        conn = UserStoreConnector(firestore_client=mock_firestore)
        conn._namespace_docs_initialized = True

        ns_col = MagicMock()
        mock_firestore.collection.return_value = ns_col

        # Setup: ns_00 has 80k vectors, ns_01 has 20k, rest have 50k
        def make_ns_doc(i):
            if i == 0:
                return _mock_doc(True, {"vector_count": 80_000, "user_count": 8})
            elif i == 1:
                return _mock_doc(True, {"vector_count": 20_000, "user_count": 2})
            else:
                return _mock_doc(True, {"vector_count": 50_000, "user_count": 5})

        ns_col.document.return_value.get.side_effect = [
            make_ns_doc(i) for i in range(20)
        ]
        ns_col.document.return_value.update = MagicMock()

        result = conn._assign_namespace()

        assert result == "ns_01"

    def test_skips_full_namespaces(self, mock_firestore):
        """Namespaces at max user_count are skipped."""
        conn = UserStoreConnector(firestore_client=mock_firestore)
        conn._namespace_docs_initialized = True

        ns_col = MagicMock()

        # ns_00 is full (10 users), ns_01 has room
        docs = []
        for i in range(20):
            if i == 0:
                docs.append(_mock_doc(True, {"vector_count": 90_000, "user_count": 10}))
            elif i == 1:
                docs.append(_mock_doc(True, {"vector_count": 30_000, "user_count": 3}))
            else:
                docs.append(_mock_doc(True, {"vector_count": 100_000, "user_count": 10}))

        # Each call to ns_col.document(ns_id).get() needs its own mock chain
        doc_mocks = {}
        for i in range(20):
            ns_id = f"ns_{i:02d}"
            dm = MagicMock()
            dm.get.return_value = docs[i]
            doc_mocks[ns_id] = dm

        ns_col.document.side_effect = lambda ns_id: doc_mocks[ns_id]
        mock_firestore.collection.side_effect = lambda name: ns_col if name == "namespaces" else MagicMock()

        result = conn._assign_namespace()

        assert result == "ns_01"

    def test_all_full_raises(self, mock_firestore):
        """RuntimeError when every namespace is at max users."""
        conn = UserStoreConnector(firestore_client=mock_firestore)
        conn._namespace_docs_initialized = True

        ns_col = MagicMock()
        full_doc = _mock_doc(True, {"vector_count": 100_000, "user_count": 10})

        doc_mocks = {}
        for i in range(20):
            ns_id = f"ns_{i:02d}"
            dm = MagicMock()
            dm.get.return_value = full_doc
            doc_mocks[ns_id] = dm

        ns_col.document.side_effect = lambda ns_id: doc_mocks[ns_id]
        mock_firestore.collection.side_effect = lambda name: ns_col if name == "namespaces" else MagicMock()

        with pytest.raises(RuntimeError, match="All namespaces are full"):
            conn._assign_namespace()

    def test_permanent_binding(self, connector, mock_firestore):
        """User keeps their namespace on subsequent calls."""
        existing = {
            "user_id": "auth0|bound",
            "namespace": "ns_07",
            "vector_count": 500,
            "vector_quota": 10_000,
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=existing)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result1 = connector.get_or_create_user("auth0|bound")
        result2 = connector.get_or_create_user("auth0|bound")

        assert result1["namespace"] == "ns_07"
        assert result2["namespace"] == "ns_07"
        connector._assign_namespace.assert_not_called()


# =============================================================================
# User Creation with Quota Fields Tests
# =============================================================================


class TestGetOrCreateUserQuotaFields:
    """Test that new users are created with quota fields."""

    def test_new_user_has_pool_namespace(self, connector, mock_firestore):
        """New user doc gets a pool namespace."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new1")

        assert result["namespace"] == "ns_00"
        connector._assign_namespace.assert_called_once()

    def test_new_user_has_vector_count_zero(self, connector, mock_firestore):
        """New user starts with vector_count of 0."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new2")

        assert result["vector_count"] == 0

    def test_new_user_has_vector_quota(self, connector, mock_firestore):
        """New user gets default vector quota of 10,000."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new3")

        assert result["vector_quota"] == 10_000

    def test_existing_user_returns_existing_data(self, connector, mock_firestore):
        """Existing user with pool namespace is returned as-is."""
        existing = {
            "user_id": "auth0|old1",
            "namespace": "ns_03",
            "vector_count": 500,
            "vector_quota": 10_000,
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=existing)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|old1")

        assert result == existing
        mock_doc_ref.set.assert_not_called()

    def test_new_user_saved_data_matches_returned_data(self, connector, mock_firestore):
        """Data saved to Firestore matches what is returned."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new4")

        saved_data = mock_doc_ref.set.call_args[0][0]
        assert saved_data == result


# =============================================================================
# Quota Check Tests
# =============================================================================


class TestCheckQuota:
    """Tests for the check_quota method."""

    def test_under_limit(self, connector, mock_firestore):
        """Returns (True, count, quota) when under limit."""
        user_data = {
            "user_id": "u1",
            "vector_count": 5000,
            "vector_quota": 10_000,
            "namespace": "ns_00",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u1")

        assert ok is True
        assert count == 5000
        assert quota == 10_000

    def test_at_limit(self, connector, mock_firestore):
        """Returns (False, 10000, 10000) when exactly at limit."""
        user_data = {
            "user_id": "u2",
            "vector_count": 10_000,
            "vector_quota": 10_000,
            "namespace": "ns_00",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u2")

        assert ok is False
        assert count == 10_000
        assert quota == 10_000

    def test_over_limit(self, connector, mock_firestore):
        """Returns (False, count, quota) when over limit."""
        user_data = {
            "user_id": "u3",
            "vector_count": 11_000,
            "vector_quota": 10_000,
            "namespace": "ns_00",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u3")

        assert ok is False
        assert count == 11_000

    def test_zero_count_passes(self, connector, mock_firestore):
        """Fresh user with zero vectors passes quota check."""
        user_data = {
            "user_id": "u5",
            "vector_count": 0,
            "vector_quota": 10_000,
            "namespace": "ns_00",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u5")

        assert ok is True
        assert count == 0

    def test_one_below_limit_passes(self, connector, mock_firestore):
        """User at 9999/10000 still passes."""
        user_data = {
            "user_id": "u6",
            "vector_count": 9_999,
            "vector_quota": 10_000,
            "namespace": "ns_00",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u6")

        assert ok is True
        assert count == 9_999

    def test_custom_quota(self, connector, mock_firestore):
        """Respects custom quota values (e.g., premium user with 50k)."""
        user_data = {
            "user_id": "u7",
            "vector_count": 15_000,
            "vector_quota": 50_000,
            "namespace": "ns_00",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u7")

        assert ok is True
        assert quota == 50_000


# =============================================================================
# Increment Vector Count Tests
# =============================================================================


class TestIncrementVectorCount:
    """Tests for the increment_vector_count method."""

    def test_increments_both_user_and_namespace(self, connector, mock_firestore):
        """Verifies both user and namespace docs are updated."""
        user_doc_ref = MagicMock()
        ns_doc_ref = MagicMock()

        def collection_router(name):
            col = MagicMock()
            if name == "users":
                col.document.return_value = user_doc_ref
            elif name == "namespaces":
                col.document.return_value = ns_doc_ref
            return col

        mock_firestore.collection.side_effect = collection_router

        connector.increment_vector_count("u1", 10, "ns_05")

        user_doc_ref.update.assert_called_once()
        ns_doc_ref.update.assert_called_once()

    def test_no_namespace_skips_namespace_update(self, connector, mock_firestore):
        """When namespace is empty, only user doc is updated."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_vector_count("u1", 10, "")

        mock_doc_ref.update.assert_called_once()

    def test_zero_count_no_op(self, connector, mock_firestore):
        """Zero count does nothing."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_vector_count("u1", 0, "ns_00")

        mock_doc_ref.update.assert_not_called()

    def test_negative_count_no_op(self, connector, mock_firestore):
        """Negative count does nothing (use decrement instead)."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_vector_count("u1", -5, "ns_00")

        mock_doc_ref.update.assert_not_called()


# =============================================================================
# Decrement Vector Count Tests
# =============================================================================


class TestDecrementVectorCount:
    """Tests for the decrement_vector_count method."""

    def test_decrements_both_user_and_namespace(self, connector, mock_firestore):
        """Both user and namespace docs are decremented."""
        user_doc_ref = MagicMock()
        user_doc_ref.get.return_value = _mock_doc(exists=True, data={"vector_count": 50})
        ns_doc_ref = MagicMock()
        ns_doc_ref.get.return_value = _mock_doc(exists=True, data={"vector_count": 5000})

        def collection_router(name):
            col = MagicMock()
            if name == "users":
                col.document.return_value = user_doc_ref
            elif name == "namespaces":
                col.document.return_value = ns_doc_ref
            return col

        mock_firestore.collection.side_effect = collection_router

        connector.decrement_vector_count("u1", 10, "ns_05")

        user_doc_ref.update.assert_called_once()
        ns_doc_ref.update.assert_called_once()

    def test_floors_user_at_zero(self, connector, mock_firestore):
        """Floors negative user vector count to 0."""
        user_doc_ref = MagicMock()
        user_doc_ref.get.return_value = _mock_doc(exists=True, data={"vector_count": -5})

        def collection_router(name):
            col = MagicMock()
            if name == "users":
                col.document.return_value = user_doc_ref
            else:
                ns_ref = MagicMock()
                ns_ref.get.return_value = _mock_doc(exists=True, data={"vector_count": 50})
                col.document.return_value = ns_ref
            return col

        mock_firestore.collection.side_effect = collection_router

        connector.decrement_vector_count("u1", 10, "ns_00")

        calls = user_doc_ref.update.call_args_list
        assert len(calls) == 2
        floor_call = calls[1][0][0]
        assert floor_call == {"vector_count": 0}

    def test_zero_count_no_op(self, connector, mock_firestore):
        """Zero count does nothing."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.decrement_vector_count("u1", 0, "ns_00")

        mock_doc_ref.update.assert_not_called()

    def test_negative_count_no_op(self, connector, mock_firestore):
        """Negative count does nothing."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.decrement_vector_count("u1", -5, "ns_00")

        mock_doc_ref.update.assert_not_called()


# =============================================================================
# Video Registration Tests (Subcollection)
# =============================================================================


class TestRegisterVideo:
    """Tests for video registration in subcollection."""

    def test_creates_subdocument(self, connector, mock_firestore):
        """Writes to users/{id}/videos/{hash}."""
        mock_video_ref = MagicMock()
        mock_videos_collection = MagicMock()
        mock_videos_collection.document.return_value = mock_video_ref
        mock_user_ref = MagicMock()
        mock_user_ref.collection.return_value = mock_videos_collection
        mock_firestore.collection.return_value.document.return_value = mock_user_ref

        connector.register_video("u1", "hash123", 15, "video.mp4")

        mock_firestore.collection.assert_called_with("users")
        mock_firestore.collection.return_value.document.assert_called_with("u1")
        mock_user_ref.collection.assert_called_with("videos")
        mock_videos_collection.document.assert_called_with("hash123")
        mock_video_ref.set.assert_called_once()

    def test_stores_chunk_count(self, connector, mock_firestore):
        """Chunk count is persisted."""
        mock_video_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value = mock_video_ref

        connector.register_video("u1", "hash123", 42, "video.mp4")

        saved_data = mock_video_ref.set.call_args[0][0]
        assert saved_data["chunk_count"] == 42

    def test_stores_filename(self, connector, mock_firestore):
        """Filename is persisted."""
        mock_video_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value = mock_video_ref

        connector.register_video("u1", "hash123", 10, "my_video.mp4")

        saved_data = mock_video_ref.set.call_args[0][0]
        assert saved_data["filename"] == "my_video.mp4"

    def test_stores_hashed_identifier(self, connector, mock_firestore):
        """Hashed identifier is persisted."""
        mock_video_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value = mock_video_ref

        connector.register_video("u1", "abc_hash", 5, "v.mp4")

        saved_data = mock_video_ref.set.call_args[0][0]
        assert saved_data["hashed_identifier"] == "abc_hash"

    def test_stores_created_at(self, connector, mock_firestore):
        """Created_at timestamp is stored."""
        mock_video_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value = mock_video_ref

        connector.register_video("u1", "hash123", 10, "v.mp4")

        saved_data = mock_video_ref.set.call_args[0][0]
        assert "created_at" in saved_data


class TestGetVideoChunkCount:
    """Tests for reading video chunk count from subcollection."""

    def test_returns_correct_count(self, connector, mock_firestore):
        """Reads back what was written."""
        mock_doc = _mock_doc(exists=True, data={"chunk_count": 25, "filename": "v.mp4"})
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value = mock_doc

        count = connector.get_video_chunk_count("u1", "hash123")

        assert count == 25

    def test_missing_video_returns_zero(self, connector, mock_firestore):
        """Returns 0 if video not found."""
        mock_doc = _mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value = mock_doc

        count = connector.get_video_chunk_count("u1", "nonexistent")

        assert count == 0

    def test_missing_chunk_count_field_returns_zero(self, connector, mock_firestore):
        """Returns 0 if chunk_count field is missing from doc."""
        mock_doc = _mock_doc(exists=True, data={"filename": "v.mp4"})
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value = mock_doc

        count = connector.get_video_chunk_count("u1", "hash123")

        assert count == 0


class TestDeregisterVideo:
    """Tests for video deregistration (subcollection delete)."""

    def test_deletes_subdocument(self, connector, mock_firestore):
        """Delete is called on the subcollection document."""
        mock_video_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value = mock_video_ref

        connector.deregister_video("u1", "hash123")

        mock_video_ref.delete.assert_called_once()

    def test_nonexistent_no_error(self, connector, mock_firestore):
        """Doesn't crash when deleting a non-existent video."""
        mock_video_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value.collection.return_value.document.return_value = mock_video_ref

        connector.deregister_video("u1", "nonexistent_hash")

        mock_video_ref.delete.assert_called_once()

    def test_uses_correct_subcollection_path(self, connector, mock_firestore):
        """Verifies the correct Firestore path is used."""
        mock_videos_collection = MagicMock()
        mock_user_ref = MagicMock()
        mock_user_ref.collection.return_value = mock_videos_collection
        mock_firestore.collection.return_value.document.return_value = mock_user_ref

        connector.deregister_video("auth0|u1", "vid_hash_abc")

        mock_firestore.collection.assert_called_with("users")
        mock_firestore.collection.return_value.document.assert_called_with("auth0|u1")
        mock_user_ref.collection.assert_called_with("videos")
        mock_videos_collection.document.assert_called_with("vid_hash_abc")
