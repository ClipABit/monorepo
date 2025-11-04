from typing import List, Dict, Any, Optional
import backend.embedding_service as embedding_service
import backend.pinecone_connect as pc


def embed_and_search(
    query: str,
    top_k: int = 3,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Embed a text query with CLIP and query Pinecone, returning mapped results.
    """
    if not query or not query.strip():
        return []
    qv = embedding_service.embed_text(query)
    client = pc.connect_to_pinecone()
    res = pc.query_vectors(
        client,
        qv,
        top_k=top_k,
        metadata_filter=filters,
        include_metadata=True,
        include_values=False,
    )
    out: List[Dict[str, Any]] = []
    for m in getattr(res, "matches", []) or []:
        md = m.metadata or {}
        out.append({
            "id": m.id,
            "score": m.score,
            "video_id": md.get("video_id"),
            "chunk_id": md.get("chunk_id"),
            "modality": md.get("modality"),
            "start_ts": md.get("start_ts"),
            "end_ts": md.get("end_ts"),
            "transcript_snippet": md.get("transcript_snippet"),
            "preview_frame_uri": md.get("preview_frame_uri"),
        })
    return out


