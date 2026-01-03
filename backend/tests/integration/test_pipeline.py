import pytest
from unittest.mock import MagicMock

class TestPipeline:
    """
    Integration tests for the video processing pipeline.
    Covers success paths, failure/rollback scenarios, and edge cases.
    """

    # ==========================================================================
    # SUCCESS SCENARIOS
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_process_video_success(self, server_instance, sample_video_bytes):
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
        server_instance.r2_connector.upload_video.return_value = (True, hashed_id)
        
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
        server_instance.preprocessor.process_video_from_bytes.return_value = chunks
        
        # Mock Embedder
        mock_embedding = MagicMock()
        mock_embedding.numpy.return_value = [0.1, 0.2]
        server_instance.video_embedder._generate_clip_embedding.return_value = mock_embedding
        
        # Mock Pinecone success
        server_instance.pinecone_connector.upsert_chunk.return_value = True

        # Execute
        result = await server_instance.process_video_background(
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
        server_instance.r2_connector.upload_video.assert_called_once()
        server_instance.preprocessor.process_video_from_bytes.assert_called_once()
        assert server_instance.video_embedder._generate_clip_embedding.call_count == 2
        assert server_instance.pinecone_connector.upsert_chunk.call_count == 2
        
        # Verify Job Store Update
        server_instance.job_store.set_job_completed.assert_called_once_with("job-success", result)

    @pytest.mark.asyncio
    async def test_process_video_empty_result(self, server_instance):
        """
        Scenario: Video processed but resulted in 0 chunks (e.g. too short).
        Expectation:
            - Job completes successfully with 0 chunks.
            - No embeddings generated.
            - No Pinecone upserts.
        """
        # Setup
        server_instance.r2_connector.upload_video.return_value = (True, "hash-empty")
        server_instance.preprocessor.process_video_from_bytes.return_value = [] # No chunks

        # Execute
        result = await server_instance.process_video_background(
            video_bytes=b"short-video",
            filename="short.mp4",
            job_id="job-empty",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "completed"
        assert result["chunks"] == 0
        
        server_instance.video_embedder._generate_clip_embedding.assert_not_called()
        server_instance.pinecone_connector.upsert_chunk.assert_not_called()

    # ==========================================================================
    # ROLLBACK / FAILURE SCENARIOS
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_rollback_on_r2_upload_failure(self, server_instance):
        """
        Scenario: R2 upload fails immediately.
        Expectation: 
            - Job marked failed.
            - No rollback actions (delete_video, delete_chunks) because nothing was created.
        """
        # Setup
        # Raise exception
        server_instance.r2_connector.upload_video.side_effect = Exception("R2 Upload Error")

        # Execute
        result = await server_instance.process_video_background(
            video_bytes=b"fake-video-data",
            filename="test.mp4",
            job_id="job-1",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "failed"
        assert "R2 Upload Error" in result["error"]

        # Rollback checks
        server_instance.r2_connector.delete_video.assert_not_called()
        server_instance.pinecone_connector.delete_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_on_preprocessing_failure(self, server_instance):
        """
        Scenario: R2 upload succeeds, but Preprocessing fails.
        Expectation:
            - Job marked failed.
            - R2 video is deleted (Rollback).
            - Pinecone delete not called (nothing upserted).
        """
        # Setup
        hashed_id = "hash-123"
        server_instance.r2_connector.upload_video.return_value = (True, hashed_id)
        
        server_instance.preprocessor.process_video_from_bytes.side_effect = Exception("Preprocessing Failed")

        # Execute
        result = await server_instance.process_video_background(
            video_bytes=b"fake-video-data",
            filename="test.mp4",
            job_id="job-2",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "failed"
        assert "Preprocessing Failed" in result["error"]

        # Rollback checks
        server_instance.r2_connector.delete_video.assert_called_once_with(hashed_id)
        server_instance.pinecone_connector.delete_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_on_embedding_failure(self, server_instance):
        """
        Scenario: R2 upload & Preprocessing succeed, but Embedding generation fails.
        Expectation:
            - Job marked failed.
            - R2 video is deleted.
            - Pinecone delete not called (nothing upserted).
        """
        # Setup
        hashed_id = "hash-456"
        server_instance.r2_connector.upload_video.return_value = (True, hashed_id)
        
        # Mock preprocessor to return one chunk
        chunk = {
            "chunk_id": "chunk-1",
            "frames": [1, 2, 3],
            "metadata": {"frame_count": 10, "complexity_score": 0.5},
            "memory_mb": 1.0
        }
        server_instance.preprocessor.process_video_from_bytes.return_value = [chunk]
        
        # Fail embedding
        server_instance.video_embedder._generate_clip_embedding.side_effect = Exception("Embedding Model Error")

        # Execute
        result = await server_instance.process_video_background(
            video_bytes=b"fake-video-data",
            filename="test.mp4",
            job_id="job-3",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "failed"
        assert "Embedding Model Error" in result["error"]

        # Rollback checks
        server_instance.r2_connector.delete_video.assert_called_once_with(hashed_id)
        server_instance.pinecone_connector.delete_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_on_partial_pinecone_failure(self, server_instance):
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
        server_instance.r2_connector.upload_video.return_value = (True, hashed_id)
        
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
        server_instance.preprocessor.process_video_from_bytes.return_value = chunks
        
        # Mock embedding to succeed
        mock_embedding = MagicMock()
        mock_embedding.numpy.return_value = [0.1, 0.2]
        server_instance.video_embedder._generate_clip_embedding.return_value = mock_embedding

        # Mock Pinecone upsert: First succeeds, Second fails
        # side_effect can be an iterable of return values or exceptions
        # Call 1: True (Success)
        # Call 2: False (Failure) OR Raise Exception
        
        # The code checks `if success:` then `else: raise Exception`
        # So we can just return [True, False]
        server_instance.pinecone_connector.upsert_chunk.side_effect = [True, False]

        # Execute
        result = await server_instance.process_video_background(
            video_bytes=b"fake-video-data",
            filename="test.mp4",
            job_id="job-4",
            namespace="test-ns"
        )

        # Verify
        assert result["status"] == "failed"
        assert "Failed to upsert chunk chunk-2" in result["error"]

        # Rollback checks
        server_instance.r2_connector.delete_video.assert_called_once_with(hashed_id)
        
        # Should delete the one that succeeded (chunk-1)
        server_instance.pinecone_connector.delete_chunks.assert_called_once_with(
            ["chunk-1"], 
            namespace="test-ns"
        )

    @pytest.mark.asyncio
    async def test_rollback_best_effort_when_cleanup_fails(self, server_instance):
        """
        Scenario: 
            - Pipeline fails (triggering rollback).
            - R2 deletion fails (Rollback step 1 fails).
        Expectation:
            - Pinecone deletion (Rollback step 2) should still be attempted.
        """
        # Setup failure in pipeline (Partial Pinecone failure to ensure we have chunks to delete)
        hashed_id = "hash-fail-cleanup"
        server_instance.r2_connector.upload_video.return_value = (True, hashed_id)
        
        chunks = [
            {"chunk_id": "c1", "frames": [], "metadata": {"frame_count": 10, "complexity_score": 0.5}, "memory_mb": 1},
            {"chunk_id": "c2", "frames": [], "metadata": {"frame_count": 10, "complexity_score": 0.5}, "memory_mb": 1}
        ]
        server_instance.preprocessor.process_video_from_bytes.return_value = chunks
        server_instance.video_embedder._generate_clip_embedding.return_value = MagicMock(numpy=lambda: [0.1])
        
        # Upsert: True, False (Trigger rollback)
        server_instance.pinecone_connector.upsert_chunk.side_effect = [True, False]
        
        # Setup failure in R2 cleanup
        server_instance.r2_connector.delete_video.return_value = False
        
        # Execute
        result = await server_instance.process_video_background(
            video_bytes=b"data", filename="test.mp4", job_id="job-5", namespace="ns"
        )
            
        # Verify R2 delete was called (and failed)
        server_instance.r2_connector.delete_video.assert_called_once_with(hashed_id)
        
        # Verify Pinecone delete was called DESPITE R2 delete failure
        server_instance.pinecone_connector.delete_chunks.assert_called_once_with(["c1"], namespace="ns")
        
        # Verify result is still failed
        assert result["status"] == "failed"

    # ==========================================================================
    # METADATA HANDLING
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_metadata_transformation(self, server_instance):
        """
        Scenario: Metadata contains complex types (timestamp_range, file_info) that need flattening.
        Expectation:
            - timestamp_range is split into start_time_s and end_time_s.
            - file_info is flattened into file_*.
            - Null values are removed.
        """
        # Setup
        server_instance.r2_connector.upload_video.return_value = (True, "hash-meta")
        
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
        server_instance.preprocessor.process_video_from_bytes.return_value = chunks
        
        # Mock embedding
        server_instance.video_embedder._generate_clip_embedding.return_value = MagicMock(numpy=lambda: [0.1])
        server_instance.pinecone_connector.upsert_chunk.return_value = True

        # Execute
        await server_instance.process_video_background(b"data", "test.mp4", "job-meta", "ns")

        # Verify Upsert Call Arguments
        call_args = server_instance.pinecone_connector.upsert_chunk.call_args
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

