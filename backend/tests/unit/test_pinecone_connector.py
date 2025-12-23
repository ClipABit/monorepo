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
        mock_client.Index.assert_called_once_with("test-index")
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


class TestDeleteChunks:
    """Test chunk deletion operations."""

    def test_delete_chunks_success(self, mock_pinecone_connector):
        """Verify successful chunk deletion."""
        connector, mock_index, mock_client, _ = mock_pinecone_connector
        
        chunk_ids = ["chunk-1", "chunk-2"]
        result = connector.delete_chunks(
            chunk_ids=chunk_ids,
            namespace="test-namespace"
        )

        assert result is True
        mock_index.delete.assert_called_once_with(ids=chunk_ids, namespace="test-namespace")

    def test_delete_chunks_empty_list(self, mock_pinecone_connector):
        """Verify deletion returns True immediately for empty list."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        result = connector.delete_chunks(chunk_ids=[])

        assert result is True
        mock_index.delete.assert_not_called()

    def test_delete_chunks_exception(self, mock_pinecone_connector):
        """Verify deletion handles exceptions gracefully."""
        connector, mock_index, _, _ = mock_pinecone_connector
        
        mock_index.delete.side_effect = Exception("Pinecone error")
        
        result = connector.delete_chunks(
            chunk_ids=["chunk-1"],
            namespace="test-namespace"
        )

        assert result is False
        mock_index.delete.assert_called_once()
