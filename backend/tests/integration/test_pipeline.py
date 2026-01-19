import pytest
from unittest.mock import MagicMock


class TestProcessingPipeline:
    """
    Integration tests for the video processing pipeline (ProcessingWorker).
    Covers success paths, failure/rollback scenarios, and edge cases.
    """

    # ==========================================================================
    # SUCCESS SCENARIOS
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_process_video_success(self, processing_worker, sample_video_bytes):
        """
        Scenario: Happy path - everything succeeds.
        Expectation:
            - R2 upload called.
            - Preprocessing called.
            - Embeddings generated.
            - Pinecone upsert called for all chunks.
            - Job marked completed.
            - Result contains correct stats.
        """
        # Setup
        hashed_id = "hash-success"
        processing_worker.r2_connector.upload_video.return_value = (True, hashed_id)
        
        # Mock Preprocessor output
        chunks = [
            {
                "chunk_id": "c1", 
                "frames": [1, 2], 
                "metadata": {"frame_count": 10, "complexity_score": 0.5, "timestamp_range": [0.0, 5.0]}, 
                "memory_mb": 1.0
            },
            {
                "chunk_id": "c2", 
                "frames": [3, 4], 
                "metadata": {"frame_count": 15, "complexity_score": 0.8, "timestamp_range": [5.0, 10.0]}, 
                "memory_mb": 1.5
            }
        ]
        processing_worker.preprocessor.process_video_from_bytes.return_value = chunks
        
        # Mock Embedder
        mock_embedding = MagicMock()
        mock_embedding.numpy.return_value = [0.1, 0.2]
        processing_worker.video_embedder._generate_clip_embedding.return_value = mock_embedding
        
        # Mock Pinecone success
        processing_worker.pinecone_connector.upsert_chunk.return_value = True

        # Execute
        result = processing_worker.process_video_background(
            video_bytes=sample_video_bytes,
            filename="success.mp4",
            job_id="job-success",
            namespace="test-ns"
        )

        # Verify Result
        assert result["status"] == "completed"
        assert result["hashed_identifier"] == hashed_id
        assert result["chunks"] == 2
        assert result["total_frames"] == 25
        assert result["total_memory_mb"] == 2.5
        
        # Verify Interactions
        processing_worker.r2_connector.upload_video.assert_called_once()
        processing_worker.preprocessor.process_video_from_bytes.assert_called_once()
        assert processing_worker.video_embedder._generate_clip_embedding.call_count == 2
        assert processing_worker.pinecone_connector.upsert_chunk.call_count == 2
        
        # Verify Job Store Update
        processing_worker.job_store.set_job_completed.assert_called_once_with("job-success", result)

    @pytest.mark.asyncio
    async def test_process_video_empty_result(self, processing_worker):
        """
        Scenario: Video processed but resulted in 0 chunks (e.g. too short).
        Expectation:
            - Job completes successfully with 0 chunks.
            - No embeddings generated.
            - No Pinecone upserts.
        """
        # Setup
        processing_worker.r2_connector.upload_video.return_value = (True, "hash-empty")
        processing_worker.preprocessor.process_video_from_bytes.return_value = []  # No chunks

        # Execute
        result = processing_worker.process_video_background(
            video_bytes=b"short-video",
            filename="short.mp4",
            job_id="job-empty",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "completed"
        assert result["chunks"] == 0
        
        processing_worker.video_embedder._generate_clip_embedding.assert_not_called()
        processing_worker.pinecone_connector.upsert_chunk.assert_not_called()

    # ==========================================================================
    # ROLLBACK / FAILURE SCENARIOS
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_rollback_on_r2_upload_failure(self, processing_worker):
        """
        Scenario: R2 upload fails immediately.
        Expectation: 
            - Job marked failed.
            - No rollback actions (delete_video, delete_chunks) because nothing was created.
        """
        # Setup
        processing_worker.r2_connector.upload_video.side_effect = Exception("R2 Upload Error")

        # Execute
        result = processing_worker.process_video_background(
            video_bytes=b"fake-video-data",
            filename="test.mp4",
            job_id="job-1",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "failed"
        assert "R2 Upload Error" in result["error"]

        # Rollback checks
        processing_worker.r2_connector.delete_video.assert_not_called()
        processing_worker.pinecone_connector.delete_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_on_preprocessing_failure(self, processing_worker):
        """
        Scenario: R2 upload succeeds, but Preprocessing fails.
        Expectation:
            - Job marked failed.
            - R2 video is deleted (Rollback).
            - Pinecone delete not called (nothing upserted).
        """
        # Setup
        hashed_id = "hash-123"
        processing_worker.r2_connector.upload_video.return_value = (True, hashed_id)
        processing_worker.preprocessor.process_video_from_bytes.side_effect = Exception("Preprocessing Failed")

        # Execute
        result = processing_worker.process_video_background(
            video_bytes=b"fake-video-data",
            filename="test.mp4",
            job_id="job-2",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "failed"
        assert "Preprocessing Failed" in result["error"]

        # Rollback checks
        processing_worker.r2_connector.delete_video.assert_called_once_with(hashed_id)
        processing_worker.pinecone_connector.delete_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_on_embedding_failure(self, processing_worker):
        """
        Scenario: R2 upload & Preprocessing succeed, but Embedding generation fails.
        Expectation:
            - Job marked failed.
            - R2 video is deleted.
            - Pinecone delete not called (nothing upserted).
        """
        # Setup
        hashed_id = "hash-456"
        processing_worker.r2_connector.upload_video.return_value = (True, hashed_id)
        
        # Mock preprocessor to return one chunk
        chunk = {
            "chunk_id": "chunk-1",
            "frames": [1, 2, 3],
            "metadata": {"frame_count": 10, "complexity_score": 0.5},
            "memory_mb": 1.0
        }
        processing_worker.preprocessor.process_video_from_bytes.return_value = [chunk]
        
        # Fail embedding
        processing_worker.video_embedder._generate_clip_embedding.side_effect = Exception("Embedding Model Error")

        # Execute
        result = processing_worker.process_video_background(
            video_bytes=b"fake-video-data",
            filename="test.mp4",
            job_id="job-3",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "failed"
        assert "Embedding Model Error" in result["error"]

        # Rollback checks
        processing_worker.r2_connector.delete_video.assert_called_once_with(hashed_id)
        processing_worker.pinecone_connector.delete_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_on_partial_pinecone_failure(self, processing_worker):
        """
        Scenario: 
            - R2 upload succeeds.
            - Preprocessing succeeds (2 chunks).
            - Chunk 1 upsert succeeds.
            - Chunk 2 upsert fails.
        Expectation:
            - Job marked failed.
            - R2 video is deleted.
            - Pinecone delete called for Chunk 1 (Rollback).
        """
        # Setup
        hashed_id = "hash-789"
        processing_worker.r2_connector.upload_video.return_value = (True, hashed_id)
        
        chunks = [
            {
                "chunk_id": "chunk-1",
                "frames": [],
                "metadata": {"frame_count": 10, "complexity_score": 0.5},
                "memory_mb": 1.0
            },
            {
                "chunk_id": "chunk-2",
                "frames": [],
                "metadata": {"frame_count": 10, "complexity_score": 0.5},
                "memory_mb": 1.0
            }
        ]
        processing_worker.preprocessor.process_video_from_bytes.return_value = chunks
        
        # Mock embedding to succeed
        mock_embedding = MagicMock()
        mock_embedding.numpy.return_value = [0.1, 0.2]
        processing_worker.video_embedder._generate_clip_embedding.return_value = mock_embedding

        # Mock Pinecone upsert: First succeeds, Second fails
        processing_worker.pinecone_connector.upsert_chunk.side_effect = [True, False]

        # Execute
        result = processing_worker.process_video_background(
            video_bytes=b"fake-video-data",
            filename="test.mp4",
            job_id="job-4",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "failed"
        assert "Failed to upsert chunk chunk-2" in result["error"]

        # Rollback checks
        processing_worker.r2_connector.delete_video.assert_called_once_with(hashed_id)
        
        # Should delete the one that succeeded (chunk-1)
        processing_worker.pinecone_connector.delete_chunks.assert_called_once_with(
            ["chunk-1"], 
            namespace="test-ns"
        )

    @pytest.mark.asyncio
    async def test_rollback_best_effort_when_cleanup_fails(self, processing_worker):
        """
        Scenario: 
            - Pipeline fails (triggering rollback).
            - R2 deletion fails (Rollback step 1 fails).
        Expectation:
            - Pinecone deletion (Rollback step 2) should still be attempted.
        """
        # Setup failure in pipeline (Partial Pinecone failure to ensure we have chunks to delete)
        hashed_id = "hash-fail-cleanup"
        processing_worker.r2_connector.upload_video.return_value = (True, hashed_id)
        
        chunks = [
            {"chunk_id": "c1", "frames": [], "metadata": {"frame_count": 10, "complexity_score": 0.5}, "memory_mb": 1},
            {"chunk_id": "c2", "frames": [], "metadata": {"frame_count": 10, "complexity_score": 0.5}, "memory_mb": 1}
        ]
        processing_worker.preprocessor.process_video_from_bytes.return_value = chunks
        processing_worker.video_embedder._generate_clip_embedding.return_value = MagicMock(numpy=lambda: [0.1])
        
        # Upsert: True, False (Trigger rollback)
        processing_worker.pinecone_connector.upsert_chunk.side_effect = [True, False]
        
        # Setup failure in R2 cleanup
        processing_worker.r2_connector.delete_video.return_value = False
        
        # Execute
        result = processing_worker.process_video_background(
            video_bytes=b"data", filename="test.mp4", job_id="job-5", namespace="ns"
        )
            
        # Verify R2 delete was called (and failed)
        processing_worker.r2_connector.delete_video.assert_called_once_with(hashed_id)
        
        # Verify Pinecone delete was called DESPITE R2 delete failure
        processing_worker.pinecone_connector.delete_chunks.assert_called_once_with(["c1"], namespace="ns")
        
        # Verify result is still failed
        assert result["status"] == "failed"

    # ==========================================================================
    # METADATA HANDLING
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_metadata_transformation(self, processing_worker):
        """
        Scenario: Metadata contains complex types (timestamp_range, file_info) that need flattening.
        Expectation:
            - timestamp_range is split into start_time_s and end_time_s.
            - file_info is flattened into file_*.
            - Null values are removed.
        """
        # Setup
        processing_worker.r2_connector.upload_video.return_value = (True, "hash-meta")
        
        raw_metadata = {
            "frame_count": 10,
            "complexity_score": 0.5,
            "timestamp_range": [10.5, 20.5],
            "file_info": {"size": 100, "type": "mp4"},
            "optional_field": None
        }
        
        chunks = [{
            "chunk_id": "c1", 
            "frames": [], 
            "metadata": raw_metadata, 
            "memory_mb": 1.0
        }]
        processing_worker.preprocessor.process_video_from_bytes.return_value = chunks
        
        # Mock embedding
        processing_worker.video_embedder._generate_clip_embedding.return_value = MagicMock(numpy=lambda: [0.1])
        processing_worker.pinecone_connector.upsert_chunk.return_value = True

        # Execute
        processing_worker.process_video_background(b"data", "test.mp4", "job-meta", "ns")

        # Verify Upsert Call Arguments
        call_args = processing_worker.pinecone_connector.upsert_chunk.call_args
        upserted_metadata = call_args.kwargs['metadata']
        
        # Check transformations
        assert upserted_metadata['start_time_s'] == 10.5
        assert upserted_metadata['end_time_s'] == 20.5
        assert upserted_metadata['file_size'] == 100
        assert upserted_metadata['file_type'] == "mp4"
        
        # Check removals
        assert 'timestamp_range' not in upserted_metadata
        assert 'file_info' not in upserted_metadata
        assert 'optional_field' not in upserted_metadata


class TestDeletionPipeline:
    """
    Integration tests for the video deletion pipeline (Server.delete_video_background).
    """

    # ==========================================================================
    # DELETION SCENARIOS
    # ==========================================================================

    def test_delete_video_success(self, server_instance):
        """
        Scenario: Happy path for deletion - everything succeeds.
        Expectation:
            - Pinecone delete_by_identifier is called.
            - R2 delete_video is called.
            - Job is marked as completed.
        """
        # Setup
        hashed_id = "hash-to-delete"
        job_id = "job-delete-success"
        namespace = "test-ns"

        server_instance.pinecone_connector.delete_by_identifier.return_value = True
        server_instance.r2_connector.delete_video.return_value = True

        # Execute
        result = server_instance.delete_video_background(
            job_id=job_id,
            hashed_identifier=hashed_id,
            namespace=namespace
        )

        # Verify Result
        assert result["status"] == "completed"
        assert result["hashed_identifier"] == hashed_id
        assert result["r2"]["deleted"] is True
        assert result["pinecone"]["deleted"] is True

        # Verify Interactions
        server_instance.pinecone_connector.delete_by_identifier.assert_called_once_with(
            hashed_identifier=hashed_id,
            namespace=namespace
        )
        server_instance.r2_connector.delete_video.assert_called_once_with(hashed_id)

        # Verify Job Store Update
        server_instance.job_store.set_job_completed.assert_called_once_with(job_id, result)

    def test_delete_video_pinecone_failure(self, server_instance):
        """
        Scenario: Pinecone deletion fails.
        Expectation:
            - Job is marked as failed.
            - R2 deletion is NOT called.
        """
        # Setup
        hashed_id = "hash-pinecone-fail"
        job_id = "job-delete-pinecone-fail"
        namespace = "test-ns"

        server_instance.pinecone_connector.delete_by_identifier.return_value = False

        # Execute
        result = server_instance.delete_video_background(
            job_id=job_id,
            hashed_identifier=hashed_id,
            namespace=namespace
        )

        # Verify Result
        assert result["status"] == "failed"
        assert "Failed to delete chunks from Pinecone" in result["error"]

        # Verify Interactions
        server_instance.pinecone_connector.delete_by_identifier.assert_called_once()
        server_instance.r2_connector.delete_video.assert_not_called()

        # Verify Job Store Update
        server_instance.job_store.set_job_failed.assert_called_once_with(job_id, "Failed to delete chunks from Pinecone")

    def test_delete_video_r2_failure(self, server_instance):
        """
        Scenario: Pinecone deletion succeeds, but R2 deletion fails.
        Expectation:
            - Job is marked as failed.
            - A critical log should indicate the inconsistency.
        """
        # Setup
        hashed_id = "hash-r2-fail"
        job_id = "job-delete-r2-fail"
        namespace = "test-ns"

        server_instance.pinecone_connector.delete_by_identifier.return_value = True
        server_instance.r2_connector.delete_video.return_value = False

        # Execute
        result = server_instance.delete_video_background(
            job_id=job_id,
            hashed_identifier=hashed_id,
            namespace=namespace
        )

        # Verify Result
        assert result["status"] == "failed"
        assert "Failed to delete video from R2" in result["error"]

        # Verify Interactions
        server_instance.pinecone_connector.delete_by_identifier.assert_called_once()
        server_instance.r2_connector.delete_video.assert_called_once_with(hashed_id)

        # Verify Job Store Update
        error_msg = "Failed to delete video from R2 after deleting chunks. System may be inconsistent."
        server_instance.job_store.set_job_failed.assert_called_once_with(job_id, error_msg)
