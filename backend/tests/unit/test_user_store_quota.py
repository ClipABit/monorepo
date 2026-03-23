"""
Extensive tests for UserStoreConnector quota, namespace, and video registration features.
"""

import pytest
from unittest.mock import MagicMock, call, patch

from database.firebase.user_store_connector import UserStoreConnector


@pytest.fixture
def mock_firestore():
    """Mock Firestore client with collection/document chain."""
    return MagicMock()


@pytest.fixture
def connector(mock_firestore):
    """UserStoreConnector with mocked Firestore client."""
    return UserStoreConnector(firestore_client=mock_firestore)


def _mock_doc(exists: bool, data: dict = None):
    """Helper to create a mock Firestore document snapshot."""
    doc = MagicMock()
    doc.exists = exists
    doc.to_dict.return_value = data
    return doc


# =============================================================================
# Namespace Resolution Tests
# =============================================================================


class TestResolveNamespace:
    """Tests for the static resolve_namespace method."""

    def test_deterministic(self):
        """Same user_id always produces the same namespace."""
        ns1 = UserStoreConnector.resolve_namespace("auth0|abc123")
        ns2 = UserStoreConnector.resolve_namespace("auth0|abc123")
        assert ns1 == ns2

    def test_different_users_produce_different_namespaces(self):
        """Different user_ids produce different namespaces."""
        ns1 = UserStoreConnector.resolve_namespace("auth0|user1")
        ns2 = UserStoreConnector.resolve_namespace("auth0|user2")
        assert ns1 != ns2

    def test_url_safe_no_special_chars(self):
        """Namespace contains only alphanumeric chars and underscore."""
        ns = UserStoreConnector.resolve_namespace("auth0|user@example.com")
        # Should start with user_ and only contain safe chars
        assert ns.startswith("user_")
        remainder = ns[5:]  # after "user_"
        assert remainder.isalnum(), f"Non-alphanumeric chars in namespace: {remainder}"

    def test_handles_auth0_format(self):
        """Works correctly with Auth0 pipe-separated user IDs."""
        ns = UserStoreConnector.resolve_namespace("auth0|abc123")
        assert ns.startswith("user_")
        assert "|" not in ns
        assert "@" not in ns
        assert "/" not in ns

    def test_handles_google_oauth_format(self):
        """Works with Google OAuth style IDs."""
        ns = UserStoreConnector.resolve_namespace("google-oauth2|1234567890")
        assert ns.startswith("user_")
        assert "|" not in ns

    def test_handles_email_format(self):
        """Works with email-style user IDs."""
        ns = UserStoreConnector.resolve_namespace("user@example.com")
        assert ns.startswith("user_")
        assert "@" not in ns

    def test_namespace_length_is_consistent(self):
        """Namespace is always the same length (user_ + 16 hex chars = 21 chars)."""
        ns1 = UserStoreConnector.resolve_namespace("short")
        ns2 = UserStoreConnector.resolve_namespace("a" * 1000)
        assert len(ns1) == len(ns2) == 21  # "user_" (5) + 16 hex chars

    def test_empty_user_id(self):
        """Handles empty string user_id without error."""
        ns = UserStoreConnector.resolve_namespace("")
        assert ns.startswith("user_")
        assert len(ns) == 21


# =============================================================================
# User Creation with Quota Fields Tests
# =============================================================================


class TestGetOrCreateUserQuotaFields:
    """Test that new users are created with quota fields."""

    def test_new_user_has_namespace(self, connector, mock_firestore):
        """New user doc includes namespace field."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new1")

        assert "namespace" in result
        assert result["namespace"] == UserStoreConnector.resolve_namespace("auth0|new1")

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
        """Existing user data is returned as-is, not overwritten."""
        existing = {
            "user_id": "auth0|old1",
            "namespace": "user_existingns",
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
            "namespace": "user_abc",
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
            "namespace": "user_abc",
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
            "namespace": "user_abc",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u3")

        assert ok is False
        assert count == 11_000

    def test_missing_fields_backfill(self, connector, mock_firestore):
        """Handles pre-existing users without quota fields, backfills them."""
        # User from before quota system existed
        user_data = {
            "user_id": "u4",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u4")

        assert ok is True
        assert count == 0
        assert quota == 10_000
        # Should have backfilled the missing fields
        mock_doc_ref.update.assert_called_once()
        update_args = mock_doc_ref.update.call_args[0][0]
        assert "vector_count" in update_args
        assert "vector_quota" in update_args
        assert "namespace" in update_args

    def test_zero_count_passes(self, connector, mock_firestore):
        """Fresh user with zero vectors passes quota check."""
        user_data = {
            "user_id": "u5",
            "vector_count": 0,
            "vector_quota": 10_000,
            "namespace": "user_abc",
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
            "namespace": "user_abc",
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
            "namespace": "user_abc",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        ok, count, quota = connector.check_quota("u7")

        assert ok is True
        assert quota == 50_000

    def test_no_backfill_when_fields_present(self, connector, mock_firestore):
        """Doesn't update Firestore when all fields are present."""
        user_data = {
            "user_id": "u8",
            "vector_count": 100,
            "vector_quota": 10_000,
            "namespace": "user_abc",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=user_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.check_quota("u8")

        mock_doc_ref.update.assert_not_called()


# =============================================================================
# Increment Vector Count Tests
# =============================================================================


class TestIncrementVectorCount:
    """Tests for the increment_vector_count method."""

    def test_calls_firestore_increment(self, connector, mock_firestore):
        """Verifies Firestore Increment transform is used."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_vector_count("u1", 10)

        mock_doc_ref.update.assert_called_once()
        update_args = mock_doc_ref.update.call_args[0][0]
        assert "vector_count" in update_args

    def test_positive_value(self, connector, mock_firestore):
        """Increment with positive count calls update."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_vector_count("u1", 50)

        mock_doc_ref.update.assert_called_once()

    def test_zero_count_no_op(self, connector, mock_firestore):
        """Zero count does nothing."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_vector_count("u1", 0)

        mock_doc_ref.update.assert_not_called()

    def test_negative_count_no_op(self, connector, mock_firestore):
        """Negative count does nothing (use decrement instead)."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_vector_count("u1", -5)

        mock_doc_ref.update.assert_not_called()

    def test_uses_correct_collection_and_document(self, connector, mock_firestore):
        """Verifies correct Firestore path."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_vector_count("auth0|abc", 10)

        mock_firestore.collection.assert_called_with("users")
        mock_firestore.collection.return_value.document.assert_called_with("auth0|abc")


# =============================================================================
# Decrement Vector Count Tests
# =============================================================================


class TestDecrementVectorCount:
    """Tests for the decrement_vector_count method."""

    def test_calls_firestore_increment_negative(self, connector, mock_firestore):
        """Verifies Firestore Increment(-n) is used."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data={"vector_count": 50})
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.decrement_vector_count("u1", 10)

        # First call is the decrement, second would be flooring if needed
        mock_doc_ref.update.assert_called()

    def test_floors_at_zero(self, connector, mock_firestore):
        """Floors negative result to 0."""
        mock_doc_ref = MagicMock()
        # After decrement, the count went negative
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data={"vector_count": -5})
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.decrement_vector_count("u1", 10)

        # Should have an update call to set vector_count to 0
        calls = mock_doc_ref.update.call_args_list
        assert len(calls) == 2  # First: Increment(-10), Second: set to 0
        floor_call = calls[1][0][0]
        assert floor_call == {"vector_count": 0}

    def test_no_floor_when_positive(self, connector, mock_firestore):
        """No flooring needed when result is positive."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data={"vector_count": 50})
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.decrement_vector_count("u1", 10)

        # Only one update call (the decrement), no flooring
        assert mock_doc_ref.update.call_count == 1

    def test_decrement_more_than_current_floors(self, connector, mock_firestore):
        """Decrement 100 when count is 50 should floor at 0."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data={"vector_count": -50})
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.decrement_vector_count("u1", 100)

        calls = mock_doc_ref.update.call_args_list
        assert len(calls) == 2
        floor_call = calls[1][0][0]
        assert floor_call == {"vector_count": 0}

    def test_zero_count_no_op(self, connector, mock_firestore):
        """Zero count does nothing."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.decrement_vector_count("u1", 0)

        mock_doc_ref.update.assert_not_called()

    def test_negative_count_no_op(self, connector, mock_firestore):
        """Negative count does nothing."""
        mock_doc_ref = MagicMock()
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.decrement_vector_count("u1", -5)

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

        # Should not raise
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
