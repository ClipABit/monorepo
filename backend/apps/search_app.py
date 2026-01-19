"""
Search Modal App

Handles semantic search with CLIP text encoder.
Medium-weight dependencies (~8-10s cold start) - lighter than full video processing.

Uses CLIPTextModelWithProjection (~150MB) instead of full CLIPModel (~350MB).
"""

import logging
import modal

# Import shared config (also configures logging to stdout)
from shared.config import (
    get_environment,
    get_secrets,
    get_pinecone_index,
    get_env_var,
)
from shared.images import get_search_image

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()
logger.info(f"Starting Search App in '{env}' environment")

# Create Modal app with search-specific image
app = modal.App(
    name=f"search-{env}",
    image=get_search_image(),
    secrets=[get_secrets()]
)


@app.cls(cpu=2.0, memory=2048, timeout=60, scaledown_window=120)
class SearchWorker:
    """
    Search worker class.
    
    Loads CLIP text encoder on startup and handles semantic search queries.
    Uses scaledown_window=120 to keep containers warm for 2 minutes after use.
    """

    @modal.enter()
    def startup(self):
        """
        Load CLIP text encoder and initialize connectors.
        
        The text encoder is loaded eagerly to avoid latency on first request.
        """
        from database.pinecone_connector import PineconeConnector
        from database.r2_connector import R2Connector
        from search.embedder import TextEmbedder

        logger.info(f"Search worker starting up! Environment = {env}")

        # Get environment variables
        PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
        R2_ACCOUNT_ID = get_env_var("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = get_env_var("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = get_env_var("R2_SECRET_ACCESS_KEY")
        ENVIRONMENT = get_environment()

        pinecone_index = get_pinecone_index()
        logger.info(f"Using Pinecone index: {pinecone_index}")

        # Initialize text embedder (loads CLIP text encoder)
        self.embedder = TextEmbedder()
        # Force eager loading of the model
        self.embedder._load_model()
        logger.info(f"CLIP text encoder loaded on device: {self.embedder.device}")

        # Initialize connectors
        self.pinecone_connector = PineconeConnector(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index
        )
        self.r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=ENVIRONMENT
        )

        logger.info("Search worker initialized and ready!")

    @modal.method()
    def search(self, query: str, namespace: str = "", top_k: int = 10) -> list:
        """
        Perform semantic search using CLIP text embeddings.
        
        Args:
            query: Natural language search query
            namespace: Pinecone namespace to search in
            top_k: Number of results to return
            
        Returns:
            List of search results with scores, metadata, and presigned URLs
        """
        logger.info(f"[Search] Query: '{query}' | namespace='{namespace}' | top_k={top_k}")

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Search Pinecone
        matches = self.pinecone_connector.query_chunks(
            query_embedding=query_embedding,
            namespace=namespace,
            top_k=top_k
        )

        # Format results with presigned URLs
        results = []
        for match in matches:
            metadata = match.get('metadata', {})

            # Generate presigned URL if identifier exists
            if 'file_hashed_identifier' not in metadata:
                logger.warning(
                    "Skipping result %s: missing file_hashed_identifier",
                    match.get('id'),
                )
                continue

            presigned_url = None
            if self.r2_connector:
                presigned_url = self.r2_connector.generate_presigned_url(
                    identifier=metadata['file_hashed_identifier'],
                    validate_exists=True,
                )

            if not presigned_url:
                logger.warning(
                    "Skipping result %s: unable to generate presigned URL",
                    match.get('id'),
                )
                continue

            metadata['presigned_url'] = presigned_url

            result = {
                'id': match.get('id'),
                'score': match.get('score', 0.0),
                'metadata': metadata
            }
            results.append(result)

        logger.info(f"[Search] Found {len(results)} results")
        return results
