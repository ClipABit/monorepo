import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pinecone import Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PineconeDeletionResult:
    """Result of Pinecone deletion operation."""
    success: bool
    chunks_found: int
    chunks_deleted: int
    chunk_ids: List[str]
    namespace: str
    error_message: Optional[str] = None

class PineconeConnector:
    """
    Pinecone Connector Class for managing Pinecone interactions. Manages a single index.
    """
    def __init__(self, api_key: str, index_name: str):
        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name

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
        index = self.client.Index(self.index_name)
        
        if metadata is None:
            metadata = {}

        try:
            chunk_embedding = chunk_embedding.tolist()
            index.upsert(vectors=[(chunk_id, chunk_embedding, metadata)], namespace=namespace)

            logger.info(f"Upserted chunk {chunk_id} into index {self.index_name} with namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error upserting chunk {chunk_id} into index {self.index_name}: {e}")
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
        index = self.client.Index(self.index_name)
        
        try:
            query_embedding = query_embedding.tolist()
            response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)

            logger.info(f"Queried top {top_k} chunks from index {self.index_name} with namespace {namespace}")
            return response['matches']
        except Exception as e:
            logger.error(f"Error querying chunks from index {self.index_name} with namespace {namespace}: {e}")
            return []

    def find_chunks_by_video(self, hashed_identifier: str, namespace: str = "__default__") -> List[str]:
        """
        Find all chunk IDs associated with a video by its hashed identifier.

        Args:
            hashed_identifier: The hashed identifier of the video
            namespace: The namespace to search in (default is "__default__")

        Returns:
            List[str]: List of chunk IDs associated with video
        """
        index = self.client.Index(self.index_name)
        
        try:
            # Query with metadata filter to find chunks for this video
            # We'll use a dummy vector since we only care about metadata filtering
            dummy_vector = [0.0] * 512  # Assuming 512-dimensional embeddings
            
            # Try both with and without base64 padding to handle encoding differences
            identifiers_to_try = [hashed_identifier]
            
            # Add version without padding if current has padding
            if hashed_identifier.endswith('='):
                identifiers_to_try.append(hashed_identifier.rstrip('='))
            # Add version with padding if current doesn't have padding
            else:
                # Add appropriate padding
                padding_needed = 4 - (len(hashed_identifier) % 4)
                if padding_needed != 4:
                    identifiers_to_try.append(hashed_identifier + '=' * padding_needed)
            
            logger.info(f"Searching for video chunks with {len(identifiers_to_try)} identifier variations")
            
            all_chunk_ids = []
            
            for identifier in identifiers_to_try:
                response = index.query(
                    vector=dummy_vector,
                    top_k=10000,  # Large number to get all chunks
                    include_metadata=True,
                    namespace=namespace,
                    filter={"file_hashed_identifier": identifier}
                )
                
                chunk_ids = [match['id'] for match in response['matches']]
                all_chunk_ids.extend(chunk_ids)
                
                if chunk_ids:
                    logger.info(f"Found {len(chunk_ids)} chunks for video {identifier} in namespace {namespace}")
                    break  # Found chunks, no need to try other variations
            
            # Remove duplicates while preserving order
            unique_chunk_ids = list(dict.fromkeys(all_chunk_ids))
            
            if not unique_chunk_ids:
                logger.info(f"No chunks found for video {hashed_identifier} (tried {len(identifiers_to_try)} variations) in namespace {namespace}")
            
            return unique_chunk_ids
        except Exception as e:
            logger.error(f"Error finding chunks for video {hashed_identifier} in namespace {namespace}: {e}")
            return []

    def batch_delete_chunks(self, chunk_ids: List[str], namespace: str = "__default__") -> bool:
        """
        Delete multiple chunks by their IDs in batch.

        Args:
            chunk_ids: List of chunk IDs to delete
            namespace: The namespace to delete from (default is "__default__")

        Returns:
            bool: True if batch deletion was successful, False otherwise
        """
        if not chunk_ids:
            logger.info("No chunk IDs provided for deletion")
            return True

        index = self.client.Index(self.index_name)
        
        try:
            # Pinecone supports batch deletion up to 1000 IDs at a time
            batch_size = 1000
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i:i + batch_size]
                index.delete(ids=batch, namespace=namespace)
                logger.info(f"Deleted batch of {len(batch)} chunks from namespace {namespace}")

            logger.info(f"Successfully deleted {len(chunk_ids)} chunks from index {self.index_name} with namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error batch deleting chunks from index {self.index_name} with namespace {namespace}: {e}")
            return False

    def delete_by_metadata(self, video_metadata: Dict[str, Any], namespace: str = "__default__") -> PineconeDeletionResult:
        """
        Delete all chunks matching video metadata.

        Args:
            video_metadata: Dictionary containing video metadata to match (must include hashed_identifier)
            namespace: The namespace to delete from (default is "__default__")

        Returns:
            PineconeDeletionResult: Result of the deletion operation
        """
        if "hashed_identifier" not in video_metadata:
            return PineconeDeletionResult(
                success=False,
                chunks_found=0,
                chunks_deleted=0,
                chunk_ids=[],
                namespace=namespace,
                error_message="hashed_identifier is required in video_metadata"
            )

        hashed_identifier = video_metadata["hashed_identifier"]
        
        try:
            # Find all chunks for this video
            chunk_ids = self.find_chunks_by_video(hashed_identifier, namespace)
            
            if not chunk_ids:
                logger.info(f"No chunks found for video {hashed_identifier} in namespace {namespace}")
                return PineconeDeletionResult(
                    success=True,
                    chunks_found=0,
                    chunks_deleted=0,
                    chunk_ids=[],
                    namespace=namespace,
                    error_message="No chunks found for video"
                )

            # Delete the chunks
            deletion_success = self.batch_delete_chunks(chunk_ids, namespace)
            
            if deletion_success:
                logger.info(f"Successfully deleted {len(chunk_ids)} chunks for video {hashed_identifier}")
                return PineconeDeletionResult(
                    success=True,
                    chunks_found=len(chunk_ids),
                    chunks_deleted=len(chunk_ids),
                    chunk_ids=chunk_ids,
                    namespace=namespace
                )
            else:
                return PineconeDeletionResult(
                    success=False,
                    chunks_found=len(chunk_ids),
                    chunks_deleted=0,
                    chunk_ids=chunk_ids,
                    namespace=namespace,
                    error_message="Failed to delete chunks"
                )

        except Exception as e:
            logger.error(f"Error deleting chunks for video {hashed_identifier}: {e}")
            return PineconeDeletionResult(
                success=False,
                chunks_found=0,
                chunks_deleted=0,
                chunk_ids=[],
                namespace=namespace,
                error_message=f"Error deleting chunks: {str(e)}"
            )