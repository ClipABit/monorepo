"""
Index File Connector for managing chunk and cluster IDs.
Uses modal.Dict for persistent storage with automatic ID generation.
"""

import modal
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class IndexFileConnector:
    """
    Connector class for managing chunk and cluster ID assignment.
    
    If the storage backend changes in the future, only this class needs to be updated.
    """
    
    # taking dict_name input for future testing with mock db
    def __init__(self, dict_name: str = "clipabit-index-dict"): 
        """
        Initialize the IndexFileConnector.
        
        Args:
            dict_name: Name of the Modal Dict to use for storage
        """
        self.dict_name = dict_name
        self._dict = None
    
    def _get_dict(self):
        """Lazy load the Modal Dict."""
        if self._dict is None:
            self._dict = modal.Dict.from_name(self.dict_name, create_if_missing=True)
        return self._dict
    
    def get_next_chunk_id(self) -> str:
        """
        Get the next available chunk ID.
        
        Returns:
            str: Next chunk ID in format "chunk_XXXXXX"
        """
        d = self._get_dict()
        
        # Get current counter or initialize to 0
        current_counter = d.get("chunk_counter", 0)
        next_counter = current_counter + 1
        
        # Update counter
        d["chunk_counter"] = next_counter
        
        chunk_id = f"chunk_{next_counter:06d}"
        logger.info(f"Generated chunk ID: {chunk_id}")
        
        return chunk_id
    
    def get_next_cluster_id(self) -> str:
        """
        Get the next available cluster ID.
        
        Returns:
            str: Next cluster ID in format "cluster_XXXXXX"
        """
        d = self._get_dict()
        
        # Get current counter or initialize to 0
        current_counter = d.get("cluster_counter", 0)
        next_counter = current_counter + 1
        
        # Update counter
        d["cluster_counter"] = next_counter

        cluster_id = f"cluster_{next_counter:06d}"
        logger.info(f"Generated cluster ID: {cluster_id}")
        
        return cluster_id
    
    # Future extension for face embeddings

    def link_chunk_to_cluster(self, chunk_id: str, cluster_id: str) -> None:
        """
        Link a chunk to a cluster.
        
        Args:
            chunk_id: The chunk ID
            cluster_id: The cluster ID to link to
        """
        d = self._get_dict()
        
        d[chunk_id] = cluster_id

        # Get existing array of chunks linked to cluster or initialize to empty list
        cluster_chunks = d.get(cluster_id, [])
        
        if chunk_id not in cluster_chunks:
            cluster_chunks.append(chunk_id)
        
        d[cluster_id] = cluster_chunks
    
    def get_cluster_for_chunk(self, chunk_id: str) -> Optional[str]:
        """
        Get the cluster ID associated with a chunk.
        
        Args:
            chunk_id: The chunk ID to look up
            
        Returns:
            str: The cluster ID if found, None otherwise
        """
        d = self._get_dict()
        cluster_id = d.get(chunk_id)
        
        if cluster_id and cluster_id.startswith("cluster_"):
            return cluster_id
        return None
    
    def get_chunks_for_cluster(self, cluster_id: str) -> Optional[list[str]]:
        """
        Get all chunk IDs associated with a cluster.
        
        Args:
            cluster_id: The cluster ID to look up
            
        Returns:
            list[str]: List of chunk IDs if found, None if cluster doesn't exist
        """
        d = self._get_dict()
        chunks = d.get(cluster_id)
        
        if chunks and isinstance(chunks, list):
            return chunks
        return None

    def get_stats(self) -> Dict[str, int]:
        """
        Get current chunk counter and cluster counter.
        
        Returns:
            Dictionary with current counters
        """
        d = self._get_dict()
        return {
            "chunk_counter": d.get("chunk_counter", 0),
            "cluster_counter": d.get("cluster_counter", 0)
        }
    
    def reset_counters(self) -> None:
        """
        Reset all counters to zero.
        """
        d = self._get_dict()
        d["chunk_counter"] = 0
        d["cluster_counter"] = 0
        logger.warning("Reset all counters to 0")
    
    def clear_data(self) -> None:
        """
        Clearing all entries.
        """
        d = self._get_dict()
        d.clear()
