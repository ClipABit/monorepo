import logging
import numpy as np

from pinecone import Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeConnector:
    """
    Pinecone Connector Class for managing Pinecone interactions. Manages a single index.
    """
    def __init__(self, api_key: str, index_name: str):
        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = self.client.Index(index_name)

    def upsert_chunk(self, chunk_id: str, chunk_embedding: np.ndarray, namespace: str = "__default__", metadata: dict = None) -> bool:
        """
        Upsert a chunk into the Pinecone index.

        Args:
            chunk_id: Unique identifier for the chunk
            chunk_embedding: The chunk embedding
            namespace: The namespace to upsert the chunk into (default is "__default__")
            metadata: Optional metadata dictionary to store with the chunk

        Returns:
            bool: True if upsert was successful, False otherwise
        """
        
        
        if metadata is None:
            metadata = {}

        try:
            chunk_embedding = chunk_embedding.tolist()
            self.index.upsert(vectors=[(chunk_id, chunk_embedding, metadata)], namespace=namespace)

            logger.info(f"Upserted chunk {chunk_id} into index {self.index_name} with namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error upserting chunk {chunk_id} into index {self.index_name}: {e}")
            return False

    def delete_chunks(self, chunk_ids: list[str], namespace: str = "__default__") -> bool:
        """
        Delete chunks from the Pinecone index.

        Args:
            chunk_ids: List of chunk IDs to delete
            namespace: The namespace to delete chunks from (default is "__default__")

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        
        try:
            if not chunk_ids:
                return True
                
            self.index.delete(ids=chunk_ids, namespace=namespace)
            logger.info(f"Deleted {len(chunk_ids)} chunks from index {self.index_name} with namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks from index {self.index_name}: {e}")
            return False

    def query_chunks(self, query_embedding: np.ndarray, namespace: str = "__default__", top_k: int = 5) -> list:
        """
        Query the Pinecone index for similar chunks.

        Args:
            query_embedding: The query embedding
            namespace: The namespace to query from (default is "__default__")
            top_k: Number of top similar chunks to retrieve
        
        Returns:
            list: List of top_k similar chunks with their metadata
        """
        
        try:
            query_embedding = query_embedding.tolist()
            response = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)

            logger.info(f"Queried top {top_k} chunks from index {self.index_name} with namespace {namespace}")
            return response['matches']
        except Exception as e:
            logger.error(f"Error querying chunks from index {self.index_name} with namespace {namespace}: {e}")
            return []