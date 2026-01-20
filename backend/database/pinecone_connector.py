import logging
from typing import Optional
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
        self._log_index_type()

    def _log_index_type(self) -> None:
        """Best-effort logging of Pinecone index type (serverless vs pod)."""
        try:
            desc = self.client.describe_index(self.index_name)
            index_type = "unknown"
            spec_details = None

            if isinstance(desc, dict):
                spec = desc.get("spec") or {}
                spec_details = spec
                if "serverless" in spec:
                    index_type = "serverless"
                elif "pod" in spec:
                    index_type = "pod"
                elif "type" in desc:
                    index_type = desc.get("type") or "unknown"
            else:
                spec = getattr(desc, "spec", None)
                spec_details = spec
                if spec is not None:
                    if getattr(spec, "serverless", None):
                        index_type = "serverless"
                    elif getattr(spec, "pod", None):
                        index_type = "pod"
                index_type = getattr(desc, "type", index_type)

            logger.info(
                f"Pinecone index '{self.index_name}' type: {index_type} | spec: {spec_details}"
            )
        except Exception as e:
            logger.warning(
                f"Unable to describe Pinecone index '{self.index_name}': {e}"
            )

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
        if not chunk_ids:
            return True
        try:
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
        
    def delete_by_identifier(self, hashed_identifier: str, namespace: str = "__default__") -> bool:
        """
        Delete chunks from the Pinecone index based on metadata filter.

        Args:
            hashed_identifier: The hashed identifier of the video
            namespace: The namespace to delete chunks from (default is "__default__")

        Returns:
            bool: True if deletion was successful, False otherwise
        """

        logger.info(f"Deleting chunks by metadata filter from index {self.index_name} with namespace {namespace}")

        if not hashed_identifier:
            return False

        metadata_filter = {"file_hashed_identifier": {"$eq": hashed_identifier}}

        try:
            self.index.delete(filter=metadata_filter, namespace=namespace)
            logger.info(f"Deleted chunks by metadata filter from index {self.index_name} with namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks by metadata from index {self.index_name}: {e}")
            return False

    def count_by_identifier(self, hashed_identifier: str, namespace: str = "__default__") -> Optional[int]:
        """
        Count chunks from the Pinecone index based on metadata filter.

        Args:
            hashed_identifier: The hashed identifier of the video
            namespace: The namespace to check (default is "__default__")

        Returns:
            Optional[int]: Vector count if available, otherwise None on error
        """
        if not hashed_identifier:
            return 0

        metadata_filter = {"file_hashed_identifier": {"$eq": hashed_identifier}}

        # Try describe_index_stats with filter (preferred if supported)
        try:
            try:
                stats = self.index.describe_index_stats(filter=metadata_filter, namespace=namespace)
            except TypeError:
                stats = self.index.describe_index_stats(filter=metadata_filter)

            if isinstance(stats, dict):
                total = stats.get("total_vector_count")
                if total is not None:
                    return int(total)
                namespaces = stats.get("namespaces", {})
                if namespace in namespaces and isinstance(namespaces[namespace], dict):
                    count = namespaces[namespace].get("vector_count")
                    if count is not None:
                        return int(count)
        except Exception as e:
            logger.warning(f"describe_index_stats filter failed: {e}")

        # Fallback: query with a dummy vector and filter to estimate count
        try:
            base_stats = self.index.describe_index_stats()
            dimension = base_stats.get("dimension")
            if not dimension:
                return None

            dummy_vector = [1.0] * int(dimension)
            top_k = 10000
            response = self.index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=False,
                namespace=namespace,
                filter=metadata_filter
            )
            matches = response.get("matches", [])
            count = len(matches)
            if count >= top_k:
                logger.warning(
                    f"Count for {hashed_identifier[:8]} hit top_k={top_k}; returning lower-bound estimate."
                )
            return count
        except Exception as e:
            logger.error(f"Error counting chunks by metadata in index {self.index_name}: {e}")
            return None
