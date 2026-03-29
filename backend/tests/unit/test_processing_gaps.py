"""
Tests for ProcessingService edge cases — rollback failures, batch parent paths,
embedding failures, empty chunks, and exception vs return-false distinctions.
"""

import pytest
from unittest.mock import MagicMock
import numpy as np


def _make_mock_chunks(n):
    """Create n mock processed chunks."""
    chunks = []
    for i in range(n):
        chunks.append({
            "chunk_id": f"job1_chunk_{i:04d}",
            "frames": [np.zeros((480, 640, 3), dtype=np.uint8)],
            "metadata": {
                "frame_count": 8,
                "complexity_score": 0.5,
                "timestamp_range": (i * 5.0, (i + 1) * 5.0),
                "file_info": {"filename": "test.mp4", "type": "video/mp4", "hashed_identifier": "hash123"},
            },
            "memory_mb": 1.0,
        })
    return chunks


class TestProcessingEdgeCases:
    """Tests for processing service edge cases and error paths."""

    def _create_service_with_mocks(self, n_chunks=1):
        from services.processing_service import ProcessingService

        service = ProcessingService.__new__(ProcessingService)
        service.preprocessor = MagicMock()
        service.video_embedder = MagicMock()
        service.pinecone_connector = MagicMock()
        service.job_store = MagicMock()
        service.user_store = MagicMock()

        service.preprocessor.process_video_from_bytes.return_value = _make_mock_chunks(n_chunks)

        mock_embedding = MagicMock()
        mock_embedding.numpy.return_value = np.zeros(512)
        service.video_embedder._generate_clip_embedding.return_value = mock_embedding

        service.pinecone_connector.upsert_chunk.return_value = True
        service.user_store.reserve_quota.return_value = (True, 0, 10_000)

        return service

    def _run(self, service, **overrides):
        defaults = dict(
            video_bytes=b"fake_video",
            filename="test.mp4",
            job_id="job1",
            namespace="user_ns",
            parent_batch_id=None,
            user_id="auth0|user1",
            hashed_identifier="hash123",
            project_id="",
        )
        defaults.update(overrides)
        return service.process_video_background(**defaults)

    def test_reserve_quota_exception_leaves_quota_unreleased(self):
        """If reserve_quota raises (vs returning False), quota_reserved stays False — no decrement."""
        service = self._create_service_with_mocks()
        service.user_store.reserve_quota.side_effect = Exception("Transaction contention exhausted")

        result = self._run(service)

        assert result["status"] == "failed"
        # quota_reserved never became True, so decrement is skipped
        service.user_store.decrement_vector_count.assert_not_called()
        # No upserts happened
        service.pinecone_connector.upsert_chunk.assert_not_called()

    def test_pinecone_delete_failure_blocks_quota_decrement(self):
        """If delete_chunks raises in rollback, the exception propagates — decrement never executes."""
        service = self._create_service_with_mocks(n_chunks=3)
        # First 2 upserts succeed, third fails
        service.pinecone_connector.upsert_chunk.side_effect = [True, True, False]
        # delete_chunks raises inside the except block
        service.pinecone_connector.delete_chunks.side_effect = Exception("Pinecone unreachable")

        with pytest.raises(Exception, match="Pinecone unreachable"):
            self._run(service)

        # delete_chunks was called and raised
        service.pinecone_connector.delete_chunks.assert_called_once()
        # Because delete_chunks raised inside the except block, decrement was NOT reached
        service.user_store.decrement_vector_count.assert_not_called()

    def test_set_job_completed_failure_triggers_rollback(self):
        """If set_job_completed raises after successful upserts, rollback deletes vectors and releases quota."""
        service = self._create_service_with_mocks()
        service.job_store.set_job_completed.side_effect = Exception("Job store down")

        result = self._run(service)

        assert result["status"] == "failed"
        # Vectors were upserted then rolled back
        service.pinecone_connector.upsert_chunk.assert_called_once()
        service.pinecone_connector.delete_chunks.assert_called_once()
        # Quota reservation was released
        service.user_store.decrement_vector_count.assert_called_once_with("auth0|user1", 1, "user_ns")

    def test_batch_parent_update_success(self):
        """Success path with parent_batch_id: both job and batch are updated."""
        service = self._create_service_with_mocks()
        service.job_store.update_batch_on_child_completion.return_value = True

        result = self._run(service, parent_batch_id="batch_001")

        assert result["status"] == "completed"
        service.job_store.set_job_completed.assert_called_once()
        service.job_store.update_batch_on_child_completion.assert_called_once()
        args = service.job_store.update_batch_on_child_completion.call_args[0]
        assert args[0] == "batch_001"
        assert args[1] == "job1"

    def test_batch_parent_update_returns_false(self):
        """Batch update returning False logs error but result is still completed."""
        service = self._create_service_with_mocks()
        service.job_store.update_batch_on_child_completion.return_value = False

        result = self._run(service, parent_batch_id="batch_002")

        assert result["status"] == "completed"

    def test_batch_parent_update_raises_triggers_rollback_but_propagates(self):
        """Batch update raising exception triggers rollback, but the error-path batch update
        also raises (same side_effect), so the function propagates the exception.
        Vectors are deleted and quota released before the second raise."""
        service = self._create_service_with_mocks()
        # First call (success path line 216) raises → enters except block
        # Second call (error path line 261) also raises → propagates out
        service.job_store.update_batch_on_child_completion.side_effect = Exception("Batch store broken")

        with pytest.raises(Exception, match="Batch store broken"):
            self._run(service, parent_batch_id="batch_003")

        # Rollback DID execute before the second raise
        service.pinecone_connector.delete_chunks.assert_called_once()
        service.user_store.decrement_vector_count.assert_called_once()

    def test_batch_parent_update_in_error_path(self):
        """When processing fails with parent_batch_id, batch is updated with error result."""
        service = self._create_service_with_mocks()
        service.pinecone_connector.upsert_chunk.return_value = False

        result = self._run(service, parent_batch_id="batch_004")

        assert result["status"] == "failed"
        service.job_store.update_batch_on_child_completion.assert_called_once()
        error_result = service.job_store.update_batch_on_child_completion.call_args[0][2]
        assert error_result["status"] == "failed"
        assert "error" in error_result

    def test_user_id_and_parent_batch_id_both_provided(self):
        """Real-world batch upload: both quota tracking and batch tracking active."""
        service = self._create_service_with_mocks(n_chunks=2)
        service.job_store.update_batch_on_child_completion.return_value = True

        result = self._run(service, parent_batch_id="batch_005")

        assert result["status"] == "completed"
        service.user_store.reserve_quota.assert_called_once_with("auth0|user1", 2, "user_ns")
        service.user_store.register_video.assert_called_once()
        service.job_store.update_batch_on_child_completion.assert_called_once()

    def test_no_user_id_with_parent_batch_id(self):
        """Batch upload without user_id: quota skipped, batch still tracked."""
        service = self._create_service_with_mocks()
        service.job_store.update_batch_on_child_completion.return_value = True

        result = self._run(service, user_id=None, parent_batch_id="batch_006")

        assert result["status"] == "completed"
        service.user_store.reserve_quota.assert_not_called()
        service.user_store.register_video.assert_not_called()
        service.job_store.update_batch_on_child_completion.assert_called_once()

    def test_embedding_failure_mid_loop_releases_quota(self):
        """Embedding exception after 1 successful upsert: rollback + full quota release."""
        service = self._create_service_with_mocks(n_chunks=3)

        call_count = 0
        mock_embedding = MagicMock()
        mock_embedding.numpy.return_value = np.zeros(512)

        def embedding_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("CLIP encoder OOM")
            return mock_embedding

        service.video_embedder._generate_clip_embedding.side_effect = embedding_side_effect

        result = self._run(service)

        assert result["status"] == "failed"
        # 1 chunk was upserted before embedding failure on chunk 2
        service.pinecone_connector.delete_chunks.assert_called_once()
        # Full reservation (3 chunks) is released
        service.user_store.decrement_vector_count.assert_called_once_with("auth0|user1", 3, "user_ns")

    def test_empty_processed_chunks_completes_with_zero(self):
        """Preprocessor returning [] results in completed job with 0 chunks."""
        service = self._create_service_with_mocks(n_chunks=0)

        result = self._run(service)

        assert result["status"] == "completed"
        assert result["chunks"] == 0
        assert result["total_frames"] == 0
        service.pinecone_connector.upsert_chunk.assert_not_called()
        # reserve_quota called with count=0 (short-circuits)
        service.user_store.reserve_quota.assert_called_once_with("auth0|user1", 0, "user_ns")

    def test_upsert_raises_exception_triggers_rollback(self):
        """upsert_chunk raising (vs returning False) still triggers rollback."""
        service = self._create_service_with_mocks(n_chunks=2)
        service.pinecone_connector.upsert_chunk.side_effect = [True, Exception("Pinecone timeout")]

        result = self._run(service)

        assert result["status"] == "failed"
        # First chunk was upserted, then exception on second
        service.pinecone_connector.delete_chunks.assert_called_once()
        service.user_store.decrement_vector_count.assert_called_once_with("auth0|user1", 2, "user_ns")

    def test_decrement_failure_in_rollback_still_returns_failed(self):
        """If decrement raises during rollback, processing still returns failed (critical log)."""
        service = self._create_service_with_mocks()
        service.pinecone_connector.upsert_chunk.return_value = False
        service.user_store.decrement_vector_count.side_effect = Exception("Firestore down")

        result = self._run(service)

        assert result["status"] == "failed"
        service.user_store.decrement_vector_count.assert_called_once()

    def test_user_id_injected_into_all_chunks(self):
        """Verify user_id is in metadata of every chunk, not just the first."""
        service = self._create_service_with_mocks(n_chunks=3)

        result = self._run(service)

        assert result["status"] == "completed"
        # Check all 3 upsert calls have user_id in metadata
        upsert_calls = service.pinecone_connector.upsert_chunk.call_args_list
        assert len(upsert_calls) == 3
        for c in upsert_calls:
            metadata = c.kwargs["metadata"]
            assert metadata["user_id"] == "auth0|user1"

    def test_project_id_injected_when_provided(self):
        """project_id is added to chunk metadata when non-empty."""
        service = self._create_service_with_mocks()

        self._run(service, project_id="proj_abc")

        metadata = service.pinecone_connector.upsert_chunk.call_args.kwargs["metadata"]
        assert metadata["project_id"] == "proj_abc"

    def test_project_id_empty_string_not_injected(self):
        """project_id="" (falsy) is NOT injected into metadata."""
        service = self._create_service_with_mocks()

        self._run(service, project_id="")

        metadata = service.pinecone_connector.upsert_chunk.call_args.kwargs["metadata"]
        assert "project_id" not in metadata

    def test_project_id_none_not_injected(self):
        """project_id=None is NOT injected into metadata."""
        service = self._create_service_with_mocks()

        self._run(service, project_id=None)

        metadata = service.pinecone_connector.upsert_chunk.call_args.kwargs["metadata"]
        assert "project_id" not in metadata

    def test_set_job_failed_exception_propagates(self):
        """If set_job_failed raises in error handler, exception propagates (not caught)."""
        service = self._create_service_with_mocks()
        service.preprocessor.process_video_from_bytes.side_effect = Exception("Bad video")
        service.job_store.set_job_failed.side_effect = Exception("Job store also down")

        with pytest.raises(Exception, match="Job store also down"):
            self._run(service)

    def test_none_metadata_values_are_stripped(self):
        """Metadata keys with None values are removed before upsert."""
        service = self._create_service_with_mocks()
        chunks = _make_mock_chunks(1)
        chunks[0]["metadata"]["extra_field"] = None
        service.preprocessor.process_video_from_bytes.return_value = chunks

        self._run(service)

        metadata = service.pinecone_connector.upsert_chunk.call_args.kwargs["metadata"]
        assert "extra_field" not in metadata

    def test_timestamp_range_transformed_to_start_end(self):
        """timestamp_range tuple is converted to start_time_s and end_time_s."""
        service = self._create_service_with_mocks()

        self._run(service)

        metadata = service.pinecone_connector.upsert_chunk.call_args.kwargs["metadata"]
        assert metadata["start_time_s"] == 0.0
        assert metadata["end_time_s"] == 5.0
        assert "timestamp_range" not in metadata

    def test_file_info_flattened_with_prefix(self):
        """file_info dict is flattened with file_ prefix."""
        service = self._create_service_with_mocks()

        self._run(service)

        metadata = service.pinecone_connector.upsert_chunk.call_args.kwargs["metadata"]
        assert metadata["file_filename"] == "test.mp4"
        assert metadata["file_type"] == "video/mp4"
        assert "file_info" not in metadata

    def test_completed_result_has_expected_fields(self):
        """Verify all expected fields in completed result."""
        service = self._create_service_with_mocks()

        result = self._run(service)

        assert result["status"] == "completed"
        assert result["job_id"] == "job1"
        assert result["hashed_identifier"] == "hash123"
        assert result["filename"] == "test.mp4"
        assert result["chunks"] == 1
        assert "total_frames" in result
        assert "total_memory_mb" in result
        assert "avg_complexity" in result
        assert "chunk_details" in result
        assert len(result["chunk_details"]) == 1
