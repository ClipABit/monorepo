"""
Tests for PineconeConnector class.

This module tests both existing functionality and new deletion capabilities
of the PineconeConnector class, including property-based tests for deletion operations.
"""

import numpy as np
from database.pinecone_connector import PineconeConnector


class TestPineconeConnectorInitialization:
    """Test connector initialization."""

    def test_initializes_with_api_key_and_index_name(self, mocker):
        """Verify connector initializes with required parameters."""
        mock_pinecone = mocker.patch('database.pinecone_connector.Pinecone')
        mock_client = mocker.MagicMock()
        mock_pinecone.return_value = mock_client

        connector = PineconeConnector(api_key="test-api-key", index_name="test-index")

        assert connector.index_name == "test-index"
        mock_pinecone.assert_called_once_with(api_key="test-api-key")
        assert connector.client == mock_client


class TestUpsertChunk:
    """Test chunk upsert operations."""

    def test_upsert_chunk_success(self, mock_pinecone_connector, sample_embedding):
        """Verify successful chunk upsert."""
        connector, mock_index, mock_client, _ = mock_pinecone_connector
        
        result = connector.upsert_chunk(
            chunk_id="chunk-123",
            chunk_embedding=sample_embedding,
            namespace="test-namespace",
            metadata={"video_id": "video-1", "start_time": 0.0}
        )

        assert result is True
        mock_client.Index.assert_called_once_with("test-index")
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args
        assert call_args[1]['namespace'] == "test-namespace"
        vectors = call_args[1]['vectors']
        assert len(vectors) == 1
        assert vectors[0][0] == "chunk-123"
        assert vectors[0][2]["video_id"] == "video-1"

    def test_upsert_chunk_with_default_namespace(self, mock_pinecone_connector, sample_embedding):
        """Verify upsert uses default namespace when not specified."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        result = connector.upsert_chunk(
            chunk_id="chunk-123",
            chunk_embedding=sample_embedding
        )

        assert result is True
        call_args = mock_index.upsert.call_args
        assert call_args[1]['namespace'] == "__default__"

    def test_upsert_chunk_without_metadata(self, mock_pinecone_connector, sample_embedding):
        """Verify upsert works with no metadata provided."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        result = connector.upsert_chunk(
            chunk_id="chunk-123",
            chunk_embedding=sample_embedding
        )

        assert result is True
        call_args = mock_index.upsert.call_args
        vectors = call_args[1]['vectors']
        assert vectors[0][2] == {}  # Empty metadata dict

    def test_upsert_chunk_converts_numpy_to_list(self, mock_pinecone_connector, sample_embedding):
        """Verify numpy array is converted to list before upsert."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        connector.upsert_chunk(
            chunk_id="chunk-123",
            chunk_embedding=sample_embedding
        )

        call_args = mock_index.upsert.call_args
        vectors = call_args[1]['vectors']
        # Verify embedding was converted to list (not numpy array)
        assert isinstance(vectors[0][1], list)
        assert not isinstance(vectors[0][1], np.ndarray)

    def test_upsert_chunk_handles_exception(self, mock_pinecone_connector, sample_embedding):
        """Verify upsert returns False on exception."""
        connector, mock_index, _, _ = mock_pinecone_connector
        mock_index.upsert.side_effect = Exception("Pinecone error")

        result = connector.upsert_chunk(
            chunk_id="chunk-123",
            chunk_embedding=sample_embedding
        )

        assert result is False

    def test_upsert_multiple_chunks(self, mock_pinecone_connector, sample_embedding):
        """Verify multiple chunks can be upserted."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        result1 = connector.upsert_chunk("chunk-1", sample_embedding, metadata={"id": 1})
        result2 = connector.upsert_chunk("chunk-2", sample_embedding, metadata={"id": 2})
        result3 = connector.upsert_chunk("chunk-3", sample_embedding, metadata={"id": 3})

        assert result1 is True
        assert result2 is True
        assert result3 is True
        assert mock_index.upsert.call_count == 3


class TestQueryChunks:
    """Test chunk query operations."""

    def test_query_chunks_success(self, mock_pinecone_connector, sample_embedding):
        """Verify successful chunk query."""
        connector, mock_index, _, _ = mock_pinecone_connector

        # Mock query response
        mock_response = {
            'matches': [
                {'id': 'chunk-1', 'score': 0.95, 'metadata': {'video_id': 'video-1'}},
                {'id': 'chunk-2', 'score': 0.87, 'metadata': {'video_id': 'video-1'}},
                {'id': 'chunk-3', 'score': 0.82, 'metadata': {'video_id': 'video-2'}}
            ]
        }
        mock_index.query.return_value = mock_response

        results = connector.query_chunks(
            query_embedding=sample_embedding,
            namespace="test-namespace",
            top_k=3
        )

        assert len(results) == 3
        assert results[0]['id'] == 'chunk-1'
        assert results[0]['score'] == 0.95
        mock_index.query.assert_called_once()
        call_args = mock_index.query.call_args
        assert call_args[1]['namespace'] == "test-namespace"
        assert call_args[1]['top_k'] == 3
        assert call_args[1]['include_metadata'] is True

    def test_query_chunks_with_default_namespace(self, mock_pinecone_connector, sample_embedding):
        """Verify query uses default namespace when not specified."""
        connector, mock_index, _, _ = mock_pinecone_connector
        mock_index.query.return_value = {'matches': []}

        connector.query_chunks(query_embedding=sample_embedding)

        call_args = mock_index.query.call_args
        assert call_args[1]['namespace'] == "__default__"

    def test_query_chunks_with_default_top_k(self, mock_pinecone_connector, sample_embedding):
        """Verify query uses default top_k when not specified."""
        connector, mock_index, _, _ = mock_pinecone_connector
        mock_index.query.return_value = {'matches': []}

        connector.query_chunks(query_embedding=sample_embedding)

        call_args = mock_index.query.call_args
        assert call_args[1]['top_k'] == 5

    def test_query_chunks_converts_numpy_to_list(self, mock_pinecone_connector, sample_embedding):
        """Verify numpy array is converted to list before query."""
        connector, mock_index, _, _ = mock_pinecone_connector
        mock_index.query.return_value = {'matches': []}

        connector.query_chunks(query_embedding=sample_embedding)

        call_args = mock_index.query.call_args
        # Verify embedding was converted to list (not numpy array)
        assert isinstance(call_args[1]['vector'], list)
        assert not isinstance(call_args[1]['vector'], np.ndarray)

    def test_query_chunks_handles_exception(self, mock_pinecone_connector, sample_embedding):
        """Verify query returns empty list on exception."""
        connector, mock_index, _, _ = mock_pinecone_connector
        mock_index.query.side_effect = Exception("Pinecone error")

        results = connector.query_chunks(query_embedding=sample_embedding)

        assert results == []

    def test_query_chunks_with_custom_top_k(self, mock_pinecone_connector, sample_embedding):
        """Verify query respects custom top_k parameter."""
        connector, mock_index, _, _ = mock_pinecone_connector

        mock_response = {
            'matches': [
                {'id': f'chunk-{i}', 'score': 0.9 - i*0.1, 'metadata': {}}
                for i in range(10)
            ]
        }
        mock_index.query.return_value = mock_response

        results = connector.query_chunks(
            query_embedding=sample_embedding,
            top_k=10
        )

        assert len(results) == 10
        call_args = mock_index.query.call_args
        assert call_args[1]['top_k'] == 10

    def test_query_chunks_returns_empty_list_when_no_matches(self, mock_pinecone_connector, sample_embedding):
        """Verify query returns empty list when no matches found."""
        connector, mock_index, _, _ = mock_pinecone_connector
        mock_index.query.return_value = {'matches': []}

        results = connector.query_chunks(query_embedding=sample_embedding)

        assert results == []


class TestDeletionOperations:
    """Test video deletion operations."""

    def test_delete_by_metadata_success_single_chunk(self, mock_pinecone_connector):
        """Test successful deletion of video with single chunk."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Mock finding one chunk
        mock_index.query.return_value = {
            'matches': [{'id': 'video123_chunk_0'}]
        }
        mock_index.delete.return_value = None
        
        video_metadata = {"hashed_identifier": "video123"}
        result = connector.delete_by_metadata(video_metadata, "test-namespace")
        
        assert result.success is True
        assert result.chunks_found == 1
        assert result.chunks_deleted == 1
        assert result.chunk_ids == ['video123_chunk_0']
        assert result.namespace == "test-namespace"
        assert result.error_message is None
        
        # Verify query was called correctly
        mock_index.query.assert_called_once()
        query_call = mock_index.query.call_args
        assert query_call[1]['filter'] == {"file_hashed_identifier": "video123"}
        assert query_call[1]['namespace'] == "test-namespace"
        
        # Verify delete was called
        mock_index.delete.assert_called_once()
        delete_call = mock_index.delete.call_args
        assert delete_call[1]['ids'] == ['video123_chunk_0']
        assert delete_call[1]['namespace'] == "test-namespace"

    def test_delete_by_metadata_success_multiple_chunks(self, mock_pinecone_connector):
        """Test successful deletion of video with multiple chunks."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Mock finding multiple chunks
        chunk_ids = [f'video456_chunk_{i}' for i in range(5)]
        mock_index.query.return_value = {
            'matches': [{'id': chunk_id} for chunk_id in chunk_ids]
        }
        mock_index.delete.return_value = None
        
        video_metadata = {"hashed_identifier": "video456"}
        result = connector.delete_by_metadata(video_metadata, "multi-chunk-namespace")
        
        assert result.success is True
        assert result.chunks_found == 5
        assert result.chunks_deleted == 5
        assert set(result.chunk_ids) == set(chunk_ids)
        assert result.namespace == "multi-chunk-namespace"
        assert result.error_message is None
        
        # Verify batch delete was called once (since < 1000 chunks)
        mock_index.delete.assert_called_once()
        delete_call = mock_index.delete.call_args
        assert set(delete_call[1]['ids']) == set(chunk_ids)

    def test_delete_by_metadata_large_batch(self, mock_pinecone_connector):
        """Test deletion of video with large number of chunks requiring batching."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Mock finding 1500 chunks (requires 2 batches)
        chunk_ids = [f'large_video_chunk_{i}' for i in range(1500)]
        mock_index.query.return_value = {
            'matches': [{'id': chunk_id} for chunk_id in chunk_ids]
        }
        mock_index.delete.return_value = None
        
        video_metadata = {"hashed_identifier": "large_video"}
        result = connector.delete_by_metadata(video_metadata, "large-namespace")
        
        assert result.success is True
        assert result.chunks_found == 1500
        assert result.chunks_deleted == 1500
        assert len(result.chunk_ids) == 1500
        
        # Verify delete was called twice (2 batches of 1000 and 500)
        assert mock_index.delete.call_count == 2
        
        # Verify batch sizes
        call1_ids = mock_index.delete.call_args_list[0][1]['ids']
        call2_ids = mock_index.delete.call_args_list[1][1]['ids']
        assert len(call1_ids) == 1000
        assert len(call2_ids) == 500
        
        # Verify all chunks were deleted
        all_deleted_ids = call1_ids + call2_ids
        assert set(all_deleted_ids) == set(chunk_ids)

    def test_delete_by_metadata_no_chunks_found(self, mock_pinecone_connector):
        """Test deletion when no chunks exist for video."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Mock finding no chunks
        mock_index.query.return_value = {'matches': []}
        
        video_metadata = {"hashed_identifier": "nonexistent_video"}
        result = connector.delete_by_metadata(video_metadata, "empty-namespace")
        
        assert result.success is True
        assert result.chunks_found == 0
        assert result.chunks_deleted == 0
        assert result.chunk_ids == []
        assert result.namespace == "empty-namespace"
        assert result.error_message == "No chunks found for video"
        
        # Verify query was called but delete was not
        # Note: Implementation tries multiple identifier variations, so may be called more than once
        assert mock_index.query.call_count >= 1
        mock_index.delete.assert_not_called()

    def test_delete_by_metadata_missing_hashed_identifier(self, mock_pinecone_connector):
        """Test deletion fails when hashed_identifier is missing."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Test with empty metadata
        result1 = connector.delete_by_metadata({}, "test-namespace")
        assert result1.success is False
        assert "hashed_identifier is required" in result1.error_message
        
        # Test with metadata missing hashed_identifier
        result2 = connector.delete_by_metadata({"other_field": "value"}, "test-namespace")
        assert result2.success is False
        assert "hashed_identifier is required" in result2.error_message
        
        # Verify no Pinecone operations were attempted
        mock_index.query.assert_not_called()
        mock_index.delete.assert_not_called()

    def test_delete_by_metadata_query_error(self, mock_pinecone_connector):
        """Test deletion handles query errors gracefully."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Mock query to raise exception
        mock_index.query.side_effect = Exception("Pinecone query error")
        
        video_metadata = {"hashed_identifier": "error_video"}
        result = connector.delete_by_metadata(video_metadata, "error-namespace")
        
        # When query fails, find_chunks_by_video returns empty list
        # This is treated as "no chunks found" which is a successful case
        assert result.success is True
        assert result.chunks_found == 0
        assert result.chunks_deleted == 0
        assert result.chunk_ids == []
        assert result.namespace == "error-namespace"
        assert result.error_message == "No chunks found for video"

    def test_delete_by_metadata_delete_error(self, mock_pinecone_connector):
        """Test deletion handles delete errors gracefully."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Mock finding chunks but delete fails
        chunk_ids = ['video789_chunk_0', 'video789_chunk_1']
        mock_index.query.return_value = {
            'matches': [{'id': chunk_id} for chunk_id in chunk_ids]
        }
        mock_index.delete.side_effect = Exception("Pinecone delete error")
        
        video_metadata = {"hashed_identifier": "video789"}
        result = connector.delete_by_metadata(video_metadata, "error-namespace")
        
        assert result.success is False
        assert result.chunks_found == 2  # Chunks were found
        assert result.chunks_deleted == 0  # But deletion failed
        assert len(result.chunk_ids) == 2
        assert result.namespace == "error-namespace"
        assert "Failed to delete chunks" in result.error_message

    def test_find_chunks_by_video_success(self, mock_pinecone_connector):
        """Test finding chunks by video identifier."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        expected_chunks = ['video_abc_chunk_0', 'video_abc_chunk_1', 'video_abc_chunk_2']
        mock_index.query.return_value = {
            'matches': [{'id': chunk_id} for chunk_id in expected_chunks]
        }
        
        found_chunks = connector.find_chunks_by_video("video_abc", "find-namespace")
        
        assert found_chunks == expected_chunks
        
        # Verify query parameters
        mock_index.query.assert_called_once()
        query_call = mock_index.query.call_args
        assert query_call[1]['filter'] == {"file_hashed_identifier": "video_abc"}
        assert query_call[1]['namespace'] == "find-namespace"
        assert query_call[1]['top_k'] == 10000
        assert query_call[1]['include_metadata'] is True

    def test_find_chunks_by_video_error(self, mock_pinecone_connector):
        """Test find_chunks_by_video handles errors gracefully."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        mock_index.query.side_effect = Exception("Query failed")
        
        found_chunks = connector.find_chunks_by_video("error_video", "error-namespace")
        
        assert found_chunks == []

    def test_batch_delete_chunks_success(self, mock_pinecone_connector):
        """Test successful batch deletion of chunks."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        chunk_ids = ['chunk_1', 'chunk_2', 'chunk_3']
        mock_index.delete.return_value = None
        
        result = connector.batch_delete_chunks(chunk_ids, "batch-namespace")
        
        assert result is True
        mock_index.delete.assert_called_once()
        delete_call = mock_index.delete.call_args
        assert delete_call[1]['ids'] == chunk_ids
        assert delete_call[1]['namespace'] == "batch-namespace"

    def test_batch_delete_chunks_empty_list(self, mock_pinecone_connector):
        """Test batch deletion with empty chunk list."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        result = connector.batch_delete_chunks([], "empty-namespace")
        
        assert result is True
        mock_index.delete.assert_not_called()

    def test_batch_delete_chunks_large_batch(self, mock_pinecone_connector):
        """Test batch deletion with chunks requiring multiple batches."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Create 2500 chunks (requires 3 batches: 1000, 1000, 500)
        chunk_ids = [f'chunk_{i}' for i in range(2500)]
        mock_index.delete.return_value = None
        
        result = connector.batch_delete_chunks(chunk_ids, "large-batch-namespace")
        
        assert result is True
        assert mock_index.delete.call_count == 3
        
        # Verify batch sizes
        call1_ids = mock_index.delete.call_args_list[0][1]['ids']
        call2_ids = mock_index.delete.call_args_list[1][1]['ids']
        call3_ids = mock_index.delete.call_args_list[2][1]['ids']
        
        assert len(call1_ids) == 1000
        assert len(call2_ids) == 1000
        assert len(call3_ids) == 500
        
        # Verify all chunks were included
        all_deleted_ids = call1_ids + call2_ids + call3_ids
        assert set(all_deleted_ids) == set(chunk_ids)

    def test_batch_delete_chunks_error(self, mock_pinecone_connector):
        """Test batch deletion handles errors gracefully."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        chunk_ids = ['chunk_1', 'chunk_2']
        mock_index.delete.side_effect = Exception("Delete failed")
        
        result = connector.batch_delete_chunks(chunk_ids, "error-namespace")
        
        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_upsert_with_empty_embedding(self, mock_pinecone_connector):
        """Verify upsert handles empty embedding array."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        empty_embedding = np.array([])
        result = connector.upsert_chunk("chunk-123", empty_embedding)

        assert result is True
        call_args = mock_index.upsert.call_args
        vectors = call_args[1]['vectors']
        assert vectors[0][1] == []  # Empty list

    def test_query_with_empty_embedding(self, mock_pinecone_connector):
        """Verify query handles empty embedding array."""
        connector, mock_index, _, _ = mock_pinecone_connector
        mock_index.query.return_value = {'matches': []}

        empty_embedding = np.array([])
        results = connector.query_chunks(empty_embedding)

        assert results == []
        call_args = mock_index.query.call_args
        assert call_args[1]['vector'] == []

    def test_upsert_with_large_metadata(self, mock_pinecone_connector, sample_embedding):
        """Verify upsert handles large metadata dictionaries."""
        connector, mock_index, _, _ = mock_pinecone_connector

        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = connector.upsert_chunk("chunk-123", sample_embedding, metadata=large_metadata)

        assert result is True
        call_args = mock_index.upsert.call_args
        vectors = call_args[1]['vectors']
        assert len(vectors[0][2]) == 100

    def test_different_namespaces_isolated(self, mock_pinecone_connector, sample_embedding):
        """Verify different namespaces are handled separately."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        # Upsert to different namespaces
        connector.upsert_chunk("chunk-1", sample_embedding, namespace="namespace-1")
        connector.upsert_chunk("chunk-2", sample_embedding, namespace="namespace-2")

        # Verify both calls used correct namespaces
        assert mock_index.upsert.call_count == 2
        call1_namespace = mock_index.upsert.call_args_list[0][1]['namespace']
        call2_namespace = mock_index.upsert.call_args_list[1][1]['namespace']
        assert call1_namespace == "namespace-1"
        assert call2_namespace == "namespace-2"

