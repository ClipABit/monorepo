"""
Tests for delete flow vector quota decrement.

Verifies that vector count is decremented and video deregistered on successful deletion.
"""

import pytest
from unittest.mock import MagicMock


class TestDeleteQuota:
    """Tests for vector count decrement in delete_video_background."""

    def _create_service_with_mocks(self):
        """Create a ServerService instance with all connectors mocked."""
        from services.http_server import ServerService

        service = ServerService.__new__(ServerService)
        service.pinecone_connector = MagicMock()
        service.r2_connector = MagicMock()
        service.job_store = MagicMock()
        service.user_store = MagicMock()

        # Default: deletion succeeds
        service.pinecone_connector.delete_by_identifier.return_value = True
        service.r2_connector.delete_video.return_value = True

        return service

    def test_decrements_vector_count(self):
        """Decrement called with correct chunk count from video registration."""
        service = self._create_service_with_mocks()
        service.user_store.get_video_chunk_count.return_value = 15

        service.delete_video_background("job1", "hash123", "user_ns", "auth0|user1")

        service.user_store.decrement_vector_count.assert_called_once_with("auth0|user1", 15)

    def test_deregisters_video(self):
        """Subcollection entry removed on successful delete."""
        service = self._create_service_with_mocks()
        service.user_store.get_video_chunk_count.return_value = 10

        service.delete_video_background("job1", "hash123", "user_ns", "auth0|user1")

        service.user_store.deregister_video.assert_called_once_with("auth0|user1", "hash123")

    def test_no_user_id_skips_decrement(self):
        """No quota operations when user_id is empty (backward compat)."""
        service = self._create_service_with_mocks()

        service.delete_video_background("job1", "hash123", "user_ns", "")

        service.user_store.get_video_chunk_count.assert_not_called()
        service.user_store.decrement_vector_count.assert_not_called()
        service.user_store.deregister_video.assert_not_called()

    def test_no_user_id_default_skips_decrement(self):
        """Default user_id="" skips quota operations."""
        service = self._create_service_with_mocks()

        service.delete_video_background("job1", "hash123", "user_ns")

        service.user_store.get_video_chunk_count.assert_not_called()

    def test_unknown_video_decrements_zero(self):
        """Video not in subcollection means decrement by 0 (no-op)."""
        service = self._create_service_with_mocks()
        service.user_store.get_video_chunk_count.return_value = 0

        service.delete_video_background("job1", "hash123", "user_ns", "auth0|user1")

        # decrement_vector_count won't be called because chunk_count is 0
        # (the implementation checks if chunk_count > 0)
        service.user_store.deregister_video.assert_called_once()

    def test_quota_failure_does_not_crash_deletion(self):
        """If quota update fails, deletion still completes."""
        service = self._create_service_with_mocks()
        service.user_store.get_video_chunk_count.side_effect = Exception("Firestore down")

        result = service.delete_video_background("job1", "hash123", "user_ns", "auth0|user1")

        # Deletion itself should succeed; quota failure logged but not fatal
        assert result["status"] == "completed"

    def test_result_includes_vectors_removed(self):
        """Result dict includes vectors_removed count."""
        service = self._create_service_with_mocks()
        service.user_store.get_video_chunk_count.return_value = 20

        result = service.delete_video_background("job1", "hash123", "user_ns", "auth0|user1")

        assert result["vectors_removed"] == 20

    def test_result_vectors_removed_zero_when_no_user(self):
        """vectors_removed is 0 when no user_id."""
        service = self._create_service_with_mocks()

        result = service.delete_video_background("job1", "hash123", "user_ns")

        assert result["vectors_removed"] == 0

    def test_pinecone_failure_skips_quota(self):
        """If Pinecone deletion fails, quota is not decremented."""
        service = self._create_service_with_mocks()
        service.pinecone_connector.delete_by_identifier.return_value = False

        result = service.delete_video_background("job1", "hash123", "user_ns", "auth0|user1")

        assert result["status"] == "failed"
        service.user_store.decrement_vector_count.assert_not_called()
        service.user_store.deregister_video.assert_not_called()
