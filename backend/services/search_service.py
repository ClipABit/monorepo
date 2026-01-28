"""
SearchService class - base class shared between search_app.py and dev_combined.py

Exposes its own ASGI app for direct HTTP access (no server gateway hop).
"""

import logging
import modal

from shared.config import get_environment, get_env_var, get_pinecone_index

logger = logging.getLogger(__name__)


class SearchService:
    """
    Search service with direct HTTP endpoint.

    Loads CLIP text encoder on startup and handles semantic search queries.
    Exposes its own ASGI app for lower latency (bypasses server gateway).
    """

    @modal.enter()
    def startup(self):
        """Load CLIP text encoder and initialize connectors."""
        from database.pinecone_connector import PineconeConnector
        from database.r2_connector import R2Connector
        from search.text_embedder import TextEmbedder

        env = get_environment()
        logger.info(f"[{self.__class__.__name__}] Starting up in '{env}' environment")

        # Get environment variables
        PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
        R2_ACCOUNT_ID = get_env_var("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = get_env_var("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = get_env_var("R2_SECRET_ACCESS_KEY")

        pinecone_index = get_pinecone_index()
        logger.info(f"[{self.__class__.__name__}] Using Pinecone index: {pinecone_index}")

        # Initialize text embedder (loads CLIP text encoder)
        self.embedder = TextEmbedder()
        self.embedder._load_model()
        logger.info(f"[{self.__class__.__name__}] CLIP text encoder (ONNX) loaded on CPU")

        # Initialize connectors
        self.pinecone_connector = PineconeConnector(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index
        )
        self.r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=env
        )

        # Create FastAPI app for direct HTTP access
        self.fastapi_app = self._create_fastapi_app()

        logger.info(f"[{self.__class__.__name__}] Initialized and ready!")

    def _create_fastapi_app(self):
        """Create FastAPI app with search routes."""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from api.search_fastapi_router import SearchFastAPIRouter

        app = FastAPI(title="ClipABit Search API")

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add search routes
        router = SearchFastAPIRouter(search_service_instance=self)
        app.include_router(router.router)

        return app

    @modal.asgi_app()
    def asgi_app(self):
        """Expose FastAPI app as ASGI endpoint."""
        return self.fastapi_app

    def _search_internal(self, query: str, namespace: str = "", top_k: int = 10) -> list:
        """
        Internal search implementation.

        Called directly by the FastAPI router (no RPC overhead).
        """
        logger.info(f"[{self.__class__.__name__}] Query: '{query}' | namespace='{namespace}' | top_k={top_k}")

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

        logger.info(f"[{self.__class__.__name__}] Found {len(results)} results")
        return results
