import pytest
from unittest.mock import MagicMock

class TestPipelineRollback:
    """
    Integration tests for the video processing pipeline's atomic rollback logic.
    Verifies that resources are cleaned up (rolled back) when failures occur at different stages.
    """

    @pytest.mark.asyncio
    async def test_rollback_on_r2_upload_failure(self, server_instance):
        """
        Scenario: R2 upload fails immediately.
        Expectation: 
            - Job marked failed.
            - No rollback actions (delete_video, delete_chunks) because nothing was created.
        """
        # Setup
        server_instance.r2_connector.upload_video.return_value = (False, None) # Fail
        # Or raise exception
        server_instance.r2_connector.upload_video.side_effect = Exception("R2 Upload Error")

        # Execute
        result = await server_instance.process_video(
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
        server_instance.r2_connector.upload_video.side_effect = None
        
        server_instance.preprocessor.process_video_from_bytes.side_effect = Exception("Preprocessing Failed")

        # Execute
        result = await server_instance.process_video(
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
        server_instance.r2_connector.upload_video.side_effect = None
        
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
        result = await server_instance.process_video(
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
        server_instance.r2_connector.upload_video.side_effect = None
        
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
        result = await server_instance.process_video(
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
        server_instance.r2_connector.delete_video.side_effect = Exception("R2 Delete Failed")
        
        # Execute
        result = await server_instance.process_video(
            video_bytes=b"data", filename="test.mp4", job_id="job-5", namespace="ns"
        )
            
        # Verify R2 delete was called (and failed)
        server_instance.r2_connector.delete_video.assert_called_once_with(hashed_id)
        
        # Verify Pinecone delete was called DESPITE R2 delete failure
        server_instance.pinecone_connector.delete_chunks.assert_called_once_with(["c1"], namespace="ns")
        
        # Verify result is still failed
        assert result["status"] == "failed"

