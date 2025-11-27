"""
Semantic Searcher using Pinecone vector database.

Coordinates text embedding and vector search to find semantically
similar content.
"""

import logging
from typing import List, Dict, Any

from database.pinecone_connector import PineconeConnector
from search.embedder import TextEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Searcher:
    """
    High-level semantic search coordinator.
    
    Combines text embedding and Pinecone vector search to provide
    an easy-to-use interface for semantic similarity search.
    
    Usage:
        searcher = Searcher(api_key="...", index_name="chunks-index")
        results = searcher.search("woman on a train", top_k=3)
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "__default__"
    ):
        """
        Initialize searcher with Pinecone connection.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of Pinecone index to search
            namespace: Optional namespace for partitioning data
        """
        self.embedder = TextEmbedder()
        self.connector = PineconeConnector(api_key=api_key, index_name=index_name)
        self.namespace = namespace
        
        logger.info(
            f"Searcher initialized (index={index_name}, namespace='{namespace}')"
        )
    
    @property
    def device(self) -> str:
        """Get the device being used for embeddings (cpu/cuda)."""
        return self.embedder.device
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar content.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default: 5)

        Returns:
            List of matches with scores and metadata, sorted by similarity

        Example:
            results = searcher.search("cooking in kitchen", top_k=3)
            for result in results:
                print(f"Score: {result['score']}")
                print(f"Metadata: {result['metadata']}")
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search Pinecone with optional filters
        matches = self.connector.query_chunks(
            query_embedding=query_embedding,
            namespace=self.namespace,
            top_k=top_k
        )
        
        # Format results
        results = []
        for match in matches:
            result = {
                'id': match.get('id'),
                'score': match.get('score', 0.0),
                'metadata': match.get('metadata', {})
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} results")
        return results
