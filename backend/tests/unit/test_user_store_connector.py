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
    """UserStoreConnector with mocked Firestore client and patched namespace assignment."""
    conn = UserStoreConnector(firestore_client=mock_firestore)
    # Patch _assign_namespace so tests don't need full namespace collection mocking
    conn._assign_namespace = MagicMock(return_value="ns_00")
    return conn


class TestUserStoreConnectorInitialization:
    """Test connector initialization."""

    def test_initializes_with_firestore_client(self, mock_firestore):
        """Verify connector stores client and default collection."""
        connector = UserStoreConnector(firestore_client=mock_firestore)

        assert connector.db is mock_firestore
        assert connector.collection == "users"

    def test_initializes_with_custom_collection(self, mock_firestore):
        """Verify connector accepts custom collection name."""
        connector = UserStoreConnector(
            firestore_client=mock_firestore, collection="custom-users"
        )

        assert connector.collection == "custom-users"


class TestGetOrCreateUser:
    """Test JIT user creation logic."""

    def test_returns_existing_user_with_pool_namespace(
        self, connector, mock_firestore, mock_doc
    ):
        """Verify existing user with ns_XX namespace is returned without backfill."""
        existing_data = {
            "user_id": "auth0|abc123",
            "created_at": "2024-01-01T00:00:00+00:00",
            "namespace": "ns_05",
            "vector_count": 200,
            "vector_quota": 10_000,
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc(exists=True, data=existing_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|abc123")

        assert result == existing_data
        mock_doc_ref.set.assert_not_called()
        mock_doc_ref.update.assert_not_called()

    def test_creates_new_user_with_pool_namespace(
        self, connector, mock_firestore, mock_doc, default_vector_quota
    ):
        """Verify new user is created with pool namespace from _assign_namespace."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new456")

        assert result["user_id"] == "auth0|new456"
        assert result["namespace"] == "ns_00"
        assert result["vector_count"] == 0
        assert result["vector_quota"] == default_vector_quota
        mock_doc_ref.set.assert_called_once()
        connector._assign_namespace.assert_called_once()

    def test_backfills_old_hash_namespace_to_pool(
        self, connector, mock_firestore, mock_doc
    ):
        """Existing user with old user_XXX namespace gets reassigned to pool."""
        existing_data = {
            "user_id": "auth0|old",
            "namespace": "user_abc123def456",
            "vector_count": 500,
            "vector_quota": 10_000,
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc(exists=True, data=existing_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|old")

        assert result["namespace"] == "ns_00"
        mock_doc_ref.update.assert_called_once()
        connector._assign_namespace.assert_called_once()

    def test_backfills_missing_fields(
        self, connector, mock_firestore, mock_doc, default_vector_quota
    ):
        """Existing user missing quota fields gets them backfilled."""
        existing_data = {
            "user_id": "auth0|legacy",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc(exists=True, data=existing_data)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|legacy")

        assert result["namespace"] == "ns_00"
        assert result["vector_count"] == 0
        assert result["vector_quota"] == default_vector_quota
        mock_doc_ref.update.assert_called_once()

    def test_uses_correct_collection_and_document_id(
        self, connector, mock_firestore, mock_doc
    ):
        """Verify Firestore is queried with correct collection and document ID."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc(
            exists=True,
            data={
                "user_id": "auth0|abc123",
                "namespace": "ns_02",
                "vector_count": 0,
                "vector_quota": 10_000,
            },
        )
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        connector.get_or_create_user("auth0|abc123")

        mock_firestore.collection.assert_called_with("users")
        mock_firestore.collection.return_value.document.assert_called_with(
            "auth0|abc123"
        )

    def test_created_at_is_utc_iso_format(self, connector, mock_firestore, mock_doc):
        """Verify created_at timestamp is ISO format UTC."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get.return_value = mock_doc(exists=False)
        mock_firestore.collection.return_value.document.return_value = mock_doc_ref

        result = connector.get_or_create_user("auth0|new789")

        created_at = datetime.fromisoformat(result["created_at"])
        assert created_at.tzinfo is not None


class TestGetUser:
    """Test user retrieval."""

    def test_returns_user_data_when_found(self, connector, mock_firestore, mock_doc):
        """Verify existing user data is returned."""
        user_data = {
            "user_id": "auth0|abc123",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        mock_firestore.collection.return_value.document.return_value.get.return_value = mock_doc(
            exists=True, data=user_data
        )

        result = connector.get_user("auth0|abc123")

        assert result == user_data

    def test_returns_none_when_not_found(self, connector, mock_firestore, mock_doc):
        """Verify None is returned for non-existent user."""
        mock_firestore.collection.return_value.document.return_value.get.return_value = mock_doc(
            exists=False
        )

        result = connector.get_user("auth0|nonexistent")

        assert result is None


class TestUserExists:
    """Test user existence check."""

    def test_returns_true_for_existing_user(self, connector, mock_firestore, mock_doc):
        """Verify True for existing user."""
        mock_firestore.collection.return_value.document.return_value.get.return_value = mock_doc(
            exists=True
        )

        assert connector.user_exists("auth0|abc123") is True

    def test_returns_false_for_missing_user(self, connector, mock_firestore, mock_doc):
        """Verify False for non-existent user."""
        mock_firestore.collection.return_value.document.return_value.get.return_value = mock_doc(
            exists=False
        )

        assert connector.user_exists("auth0|nonexistent") is False
