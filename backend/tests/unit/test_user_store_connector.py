"""
Unit tests for UserStoreConnector.

Tests JIT user creation, retrieval, and existence checks.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from database.firebase.user_store_connector import UserStoreConnector


@pytest.fixture
def mock_firestore():
    """Mock Firestore client with collection/document chain."""
    client = MagicMock()
    return client


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


class TestUserStoreConnectorInitialization:
    """Test connector initialization."""

    def test_initializes_with_firestore_client(self, mock_firestore):
        """Verify connector stores client and default collection."""
        connector = UserStoreConnector(firestore_client=mock_firestore)

        assert connector.db is mock_firestore
        assert connector.collection == "users"

    def test_initializes_with_custom_collection(self, mock_firestore):
        """Verify connector accepts custom collection name."""
        connector = UserStoreConnector(firestore_client=mock_firestore, collection="custom-users")

        assert connector.collection == "custom-users"


class TestGetOrCreateUser:
    """Test JIT user creation logic."""

    def test_returns_existing_user(self, connector, mock_firestore):
        """Verify existing user is returned without creating a new one."""
        existing_data = {"user_id": "auth0|abc123", "created_at": "2024-01-01T00:00:00+00:00"}
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data=existing_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|abc123")

        assert result == existing_data
        mock_doc_ref.set.assert_not_called()

    def test_creates_new_user_when_not_found(self, connector, mock_firestore):
        """Verify new user is created with default fields when not found."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new456")

        assert result["user_id"] == "auth0|new456"
        assert "created_at" in result
        mock_doc_ref.set.assert_called_once()
        saved_data = mock_doc_ref.set.call_args[0][0]
        assert saved_data["user_id"] == "auth0|new456"

    def test_uses_correct_collection_and_document_id(self, connector, mock_firestore):
        """Verify Firestore is queried with correct collection and document ID."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data={"user_id": "auth0|abc123"})
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.get_or_create_user("auth0|abc123")

        mock_firestore.collection.assert_called_with("users")
        mock_firestore.collection.return_value.document.assert_called_with("auth0|abc123")

    def test_created_at_is_utc_iso_format(self, connector, mock_firestore):
        """Verify created_at timestamp is ISO format UTC."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new789")

        created_at = datetime.fromisoformat(result["created_at"])
        assert created_at.tzinfo is not None  # timezone-aware


class TestGetUser:
    """Test user retrieval."""

    def test_returns_user_data_when_found(self, connector, mock_firestore):
        """Verify existing user data is returned."""
        user_data = {"user_id": "auth0|abc123", "created_at": "2024-01-01T00:00:00+00:00"}
        mock_firestore.collection.return_value.document.return_value.get.return_value = (
            _mock_doc(exists=True, data=user_data)
        )

        result = connector.get_user("auth0|abc123")

        assert result == user_data

    def test_returns_none_when_not_found(self, connector, mock_firestore):
        """Verify None is returned for non-existent user."""
        mock_firestore.collection.return_value.document.return_value.get.return_value = (
            _mock_doc(exists=False)
        )

        result = connector.get_user("auth0|nonexistent")

        assert result is None


class TestUserExists:
    """Test user existence check."""

    def test_returns_true_for_existing_user(self, connector, mock_firestore):
        """Verify True for existing user."""
        mock_firestore.collection.return_value.document.return_value.get.return_value = (
            _mock_doc(exists=True)
        )

        assert connector.user_exists("auth0|abc123") is True

    def test_returns_false_for_missing_user(self, connector, mock_firestore):
        """Verify False for non-existent user."""
        mock_firestore.collection.return_value.document.return_value.get.return_value = (
            _mock_doc(exists=False)
        )

        assert connector.user_exists("auth0|nonexistent") is False


class TestIncrementUsage:
    """Test increment_usage updates Firestore with Increment sentinels."""

    def test_calls_get_or_create_user_then_update(self, connector, mock_firestore):
        """Increment usage ensures user exists then updates with increments."""
        from google.cloud.firestore import Increment

        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data={"user_id": "auth0|u1"})
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_usage(
            user_id="auth0|u1",
            uploaded_bytes=1000,
            upload_count=1,
            frames=20,
        )

        mock_doc_ref.update.assert_called_once()
        updates = mock_doc_ref.update.call_args[0][0]
        assert updates["total_upload_bytes"] == Increment(1000)
        assert updates["total_upload_count"] == Increment(1)
        assert updates["total_frames_uploaded"] == Increment(20)

    def test_skips_update_when_all_zero(self, connector, mock_firestore):
        """Increment usage does not call update when all deltas are zero."""
        update_mock = MagicMock()
        mock_firestore.collection.return_value.document.return_value.update = update_mock
        connector.increment_usage(user_id="auth0|u1", uploaded_bytes=0, upload_count=0, frames=0)
        update_mock.assert_not_called()

    def test_update_only_includes_non_zero_fields(self, connector, mock_firestore):
        """Only non-zero arguments are added to the update dict."""
        from google.cloud.firestore import Increment

        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = _mock_doc(exists=True, data={"user_id": "auth0|u1"})
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.increment_usage(
            user_id="auth0|u1",
            uploaded_bytes=500,
            upload_count=0,
            frames=0,
        )

        updates = mock_doc_ref.update.call_args[0][0]
        assert list(updates.keys()) == ["total_upload_bytes"]
        assert updates["total_upload_bytes"] == Increment(500)
