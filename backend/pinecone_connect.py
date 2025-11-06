import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_CHUNKS = os.getenv("PINECONE_INDEX_CHUNKS", "clip-chunks-v1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def connect_to_pinecone():
    """
    Initialize and connect to Pinecone.
    
    Returns:
        Pinecone: Initialized Pinecone client
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


def ensure_index(pc: Pinecone, name: str = PINECONE_INDEX_CHUNKS, dimension: int = 512, metric: str = "cosine") -> None:
    """
    Ensure the target Pinecone index exists; create it if missing.

    Args:
        pc: Pinecone client
        name: index name
        dimension: vector dimension (CLIP ViT-B/32 = 512)
        metric: similarity metric (cosine recommended for CLIP)
    """
    existing = [i["name"] for i in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def get_index(pc: Pinecone, name: str = PINECONE_INDEX_CHUNKS):
    """
    Return an Index handle, creating the index first if needed.
    """
    ensure_index(pc, name=name)
    return pc.Index(name)


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / (n + 1e-12)


def upsert_batch(
    pc: Pinecone,
    items: List[Tuple[str, np.ndarray, Dict[str, Any]]],
    index_name: str = PINECONE_INDEX_CHUNKS,
) -> None:
    """
    Upsert a batch of vectors into the unified chunks index.

    Args:
        pc: Pinecone client
        items: list of (id, vector(np.ndarray), metadata)
        index_name: target index name
    """
    index = get_index(pc, index_name)
    payload: List[Tuple[str, List[float], Dict[str, Any]]] = [
        (item_id, _normalize(vec).tolist(), metadata or {}) for (item_id, vec, metadata) in items
    ]
    index.upsert(vectors=payload)


def upsert_visual_chunk(
    pc: Pinecone,
    video_id: str,
    chunk_id: str,
    start_ts: float,
    end_ts: float,
    vector_512: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    index_name: str = PINECONE_INDEX_CHUNKS,
) -> None:
    index = get_index(pc, index_name)
    vid = f"chunk:{video_id}:{chunk_id}:visual"
    meta = {
        "video_id": video_id,
        "chunk_id": chunk_id,
        "modality": "visual",
        "start_ts": start_ts,
        "end_ts": end_ts,
    }
    if metadata:
        meta.update(metadata)
    index.upsert(vectors=[(vid, _normalize(vector_512).tolist(), meta)])


def upsert_audio_chunk(
    pc: Pinecone,
    video_id: str,
    chunk_id: str,
    start_ts: float,
    end_ts: float,
    text_vec_512: np.ndarray,
    transcript_snippet: Optional[str] = None,
    language: Optional[str] = None,
    index_name: str = PINECONE_INDEX_CHUNKS,
) -> None:
    index = get_index(pc, index_name)
    vid = f"chunk:{video_id}:{chunk_id}:audio"
    meta: Dict[str, Any] = {
        "video_id": video_id,
        "chunk_id": chunk_id,
        "modality": "audio",
        "start_ts": start_ts,
        "end_ts": end_ts,
    }
    if transcript_snippet:
        meta["transcript_snippet"] = transcript_snippet[:256]
    if language:
        meta["language"] = language
    index.upsert(vectors=[(vid, _normalize(text_vec_512).tolist(), meta)])


def upsert_face_cluster(
    pc: Pinecone,
    video_id: str,
    cluster_id: str,
    face_vec_512: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    index_name: str = PINECONE_INDEX_CHUNKS,
) -> None:
    """
    Store face cluster reps in the same unified index (modality="face").
    Assumes 512-d normalized vectors to match CLIP index.
    """
    index = get_index(pc, index_name)
    vid = f"face:{video_id}:{cluster_id}"
    meta: Dict[str, Any] = {"video_id": video_id, "cluster_id": cluster_id, "modality": "face"}
    if metadata:
        meta.update(metadata)
    index.upsert(vectors=[(vid, _normalize(face_vec_512).tolist(), meta)])


def query_vectors(
    pc: Pinecone,
    query_vec_512: np.ndarray,
    top_k: int = 10,
    metadata_filter: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True,
    include_values: bool = False,
    index_name: str = PINECONE_INDEX_CHUNKS,
):
    """
    Query the unified chunks index using a dense query vector.
    """
    index = get_index(pc, index_name)
    return index.query(
        vector=_normalize(query_vec_512).tolist(),
        top_k=top_k,
        filter=metadata_filter,
        include_metadata=include_metadata,
        include_values=include_values,
    )


def add_face_embedding_to_index(pc, face_id: str, embedding: list, metadata: dict = None, index_name: str = "face-index"):
    """
    Add a face embedding to the Pinecone face index.
    
    Args:
        pc: Pinecone client instance
        face_id: Unique identifier for the face
        embedding: Face embedding vector (as list or numpy array)
        metadata: Optional metadata dictionary to store with the embedding
        index_name: Name of the face index (default: "face-index")
    
    Returns:
        None
    """
    index = pc.Index(index_name)
    
    # Convert numpy array to list if needed
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    # Upsert the face embedding
    index.upsert(vectors=[(face_id, embedding, metadata)])
    print(f"Added face embedding {face_id} to {index_name}")


def add_cluster_embedding_to_index(pc, cluster_id: str, embedding: list, metadata: dict = None, index_name: str = "chunks-index"):
    """
    Add a cluster embedding to the Pinecone cluster/chunks index.
    
    Args:
        pc: Pinecone client instance
        cluster_id: Unique identifier for the cluster
        embedding: Cluster embedding vector (as list or numpy array)
        metadata: Optional metadata dictionary to store with the embedding
        index_name: Name of the cluster index (default: "chunks-index")
    
    Returns:
        None
    """
    index = pc.Index(index_name)
    
    # Convert numpy array to list if needed
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    # Upsert the cluster embedding
    index.upsert(vectors=[(cluster_id, embedding, metadata)])
    print(f"Added cluster embedding {cluster_id} to {index_name}")


def main():
    """
    Test function demonstrating how to use the Pinecone connection and embedding functions.
    """
    # Connect to Pinecone
    print("Connecting to Pinecone...")
    pc = connect_to_pinecone()
    # Ensure the unified index exists
    ensure_index(pc, name=PINECONE_INDEX_CHUNKS, dimension=512, metric="cosine")


if __name__ == "__main__":
    main()
