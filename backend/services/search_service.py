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
        from database.firebase.user_store_connector import UserStoreConnector
        from search.text_embedder import TextEmbedder
        from auth.auth_connector import AuthConnector
        from face_recognition import FaceAppearanceRepository

        env = get_environment()
        logger.info(f"[{self.__class__.__name__}] Starting up in '{env}' environment")

        # Initialize Firebase Admin SDK (required for Firestore)
        import firebase_admin
        import json
        from firebase_admin import credentials, firestore
        firebase_credentials = json.loads(get_env_var("FIREBASE_ADMIN_KEY"))
        cred = credentials.Certificate(firebase_credentials)
        try:
            firebase_admin.initialize_app(cred)
        except ValueError:
            pass  # Already initialized
        firestore_client = firestore.client()

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

        self.user_store = UserStoreConnector(firestore_client=firestore_client)
        self.face_appearance_repository = FaceAppearanceRepository(firestore_client)
        self.auth_connector = AuthConnector(
            domain=get_env_var("AUTH0_DOMAIN"),
            audience=get_env_var("AUTH0_AUDIENCE"),
            user_store=self.user_store,
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
        router = SearchFastAPIRouter(search_service_instance=self, auth_connector=self.auth_connector)
        app.include_router(router.router)

        return app

    @modal.asgi_app()
    def asgi_app(self):
        """Expose FastAPI app as ASGI endpoint."""
        return self.fastapi_app

    def _search_internal(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 10,
        include_faces: list[str] | None = None,
        include_faces_mode: str = "all",
    ) -> list:
        """
        Internal search implementation.

        Called directly by the FastAPI router (no RPC overhead).
        """
        logger.info(
            "[%s] Query: '%s' | namespace='%s' | top_k=%d | include_faces=%s | include_faces_mode=%s",
            self.__class__.__name__,
            query,
            namespace,
            top_k,
            include_faces,
            include_faces_mode,
        )

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

        # Optional face-based filtering: restrict to videos that contain given face ids
        if include_faces:
            mode = (include_faces_mode or "all").lower()
            if mode not in ("all", "any"):
                mode = "all"

            logger.info(
                "[%s][FaceFilter] include_faces=%s mode=%s",
                self.__class__.__name__,
                include_faces,
                mode,
            )

            # Build sets of chunk/video ids for each face
            face_chunk_sets: list[set[str]] = []
            for face_id in include_faces:
                appearances = self.face_appearance_repository.get_chunks_for_face(namespace, face_id)
                logger.info(
                    "[%s][FaceFilter] face_id=%s appearances_count=%s",
                    self.__class__.__name__,
                    face_id,
                    len(appearances) if appearances else 0,
                )
                chunk_ids: set[str] = set()
                if appearances:
                    for doc in appearances.values():
                        cid = doc.get("video_chunk_id")
                        if cid is not None:
                            chunk_ids.add(str(cid))
                face_chunk_sets.append(chunk_ids)

            if mode == "all":
                if not face_chunk_sets:
                    eligible_chunks: set[str] = set()
                else:
                    eligible_chunks = set.intersection(*face_chunk_sets) if face_chunk_sets else set()
            else:
                eligible_chunks = set().union(*face_chunk_sets) if face_chunk_sets else set()

            logger.info(
                "[%s][FaceFilter] eligible_chunks_count=%d eligible_chunks=%s",
                self.__class__.__name__,
                len(eligible_chunks),
                eligible_chunks,
            )

            if not eligible_chunks:
                results = []
            else:
                filtered: list[dict] = []
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    metadata = r.get("metadata", {}) if isinstance(r, dict) else {}

                    match_id = None
                    if metadata.get("video_id") is not None:
                        match_id = str(metadata.get("video_id"))

                    if match_id and (match_id in eligible_chunks):
                        filtered.append(r)

                logger.info(
                    "[%s][FaceFilter] results_before=%d results_after=%d",
                    self.__class__.__name__,
                    len(results),
                    len(filtered),
                )
                results = filtered

        logger.info(f"[{self.__class__.__name__}] Found {len(results)} results")
        return results
