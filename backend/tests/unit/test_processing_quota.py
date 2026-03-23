"""
Tests for ProcessingService vector quota tracking.

Verifies that vector count is incremented and video is registered after successful processing,
and that no increment happens on failure or when user_id is None.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np


class TestProcessingQuota:
    """Tests for vector count tracking in process_video_background."""

    def _create_service_with_mocks(self):
        """Create a ProcessingService instance with all connectors mocked."""
        from services.processing_service import ProcessingService

        service = ProcessingService.__new__(ProcessingService)

        # Mock all connectors (no R2 — processing pipeline doesn't use it)
        service.preprocessor = MagicMock()
        service.video_embedder = MagicMock()
        service.pinecone_connector = MagicMock()
        service.job_store = MagicMock()
        service.user_store = MagicMock()

        # Setup preprocessor to return mock chunks
        mock_chunk = {
            "chunk_id": "job1_chunk_0000",
            "frames": [np.zeros((480, 640, 3), dtype=np.uint8)],
            "metadata": {
                "frame_count": 8,
                "complexity_score": 0.5,
                "timestamp_range": (0.0, 5.0),
                "file_info": {"filename": "test.mp4", "type": "video/mp4", "hashed_identifier": "hashed_id_123"},
            },
            "memory_mb": 1.0,
        }
        service.preprocessor.process_video_from_bytes.return_value = [mock_chunk]

        # Setup embedder to return mock embedding
        mock_embedding = MagicMock()
        mock_embedding.numpy.return_value = np.zeros(512)
        service.video_embedder._generate_clip_embedding.return_value = mock_embedding

        # Setup pinecone to return success
        service.pinecone_connector.upsert_chunk.return_value = True

        return service

    def test_increments_count_after_upsert(self):
        """increment_vector_count called with correct chunk count after successful upsert."""
        service = self._create_service_with_mocks()

        service.process_video_background(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id="auth0|user1",
            hashed_identifier="hashed_id_123",
        )

        service.user_store.increment_vector_count.assert_called_once_with("auth0|user1", 1)

    def test_registers_video_after_upsert(self):
        """register_video called with hashed_identifier after successful upsert."""
        service = self._create_service_with_mocks()

        service.process_video_background(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id="auth0|user1",
            hashed_identifier="hashed_id_123",
        )

        service.user_store.register_video.assert_called_once_with(
            "auth0|user1", "hashed_id_123", 1, "test.mp4"
        )

    def test_no_increment_on_failure(self):
        """Rollback path doesn't increment vector count."""
        service = self._create_service_with_mocks()
        # Make upsert fail
        service.pinecone_connector.upsert_chunk.return_value = False

        result = service.process_video_background(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id="auth0|user1",
            hashed_identifier="hashed_id_123",
        )

        assert result["status"] == "failed"
        service.user_store.increment_vector_count.assert_not_called()
        service.user_store.register_video.assert_not_called()

    def test_no_user_id_skips_quota(self):
        """When user_id is None, quota operations are skipped."""
        service = self._create_service_with_mocks()

        service.process_video_background(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id=None,
            hashed_identifier="hashed_id_123",
        )

        service.user_store.increment_vector_count.assert_not_called()
        service.user_store.register_video.assert_not_called()

    def test_increment_count_matches_upserted_chunks(self):
        """Count matches actual vectors upserted, not estimated."""
        service = self._create_service_with_mocks()

        # Setup 3 chunks
        mock_chunks = []
        for i in range(3):
            mock_chunks.append({
                "chunk_id": f"job1_chunk_{i:04d}",
                "frames": [np.zeros((480, 640, 3), dtype=np.uint8)],
                "metadata": {
                    "frame_count": 8,
                    "complexity_score": 0.5,
                    "timestamp_range": (i * 5.0, (i + 1) * 5.0),
                    "file_info": {"filename": "test.mp4", "type": "video/mp4", "hashed_identifier": "hashed_id_123"},
                },
                "memory_mb": 1.0,
            })
        service.preprocessor.process_video_from_bytes.return_value = mock_chunks

        service.process_video_background(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id="auth0|user1",
            hashed_identifier="hashed_id_123",
        )

        service.user_store.increment_vector_count.assert_called_once_with("auth0|user1", 3)
        service.user_store.register_video.assert_called_once_with(
            "auth0|user1", "hashed_id_123", 3, "test.mp4"
        )

    def test_quota_failure_does_not_crash_processing(self):
        """If increment fails, processing still returns completed (with critical log)."""
        service = self._create_service_with_mocks()
        service.user_store.increment_vector_count.side_effect = Exception("Firestore down")

        result = service.process_video_background(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id="auth0|user1",
            hashed_identifier="hashed_id_123",
        )

        # Processing still completes successfully even if quota update fails
        assert result["status"] == "completed"

    def test_preprocessing_failure_skips_quota(self):
        """If preprocessing fails, quota is not touched."""
        service = self._create_service_with_mocks()
        service.preprocessor.process_video_from_bytes.side_effect = Exception("Bad video")

        result = service.process_video_background(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id="auth0|user1",
            hashed_identifier="hashed_id_123",
        )

        assert result["status"] == "failed"
        service.user_store.increment_vector_count.assert_not_called()

    def test_hashed_identifier_passed_to_register_video(self):
        """The client-provided hashed_identifier flows through to register_video."""
        service = self._create_service_with_mocks()

        service.process_video_background(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id="auth0|user1",
            hashed_identifier="client_generated_hash_abc",
        )

        service.user_store.register_video.assert_called_once_with(
            "auth0|user1", "client_generated_hash_abc", 1, "test.mp4"
        )
