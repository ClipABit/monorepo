import os
import numpy as np
from pinecone import Pinecone
from deepface import DeepFace
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def connect_to_pinecone():
    """
    Initialize and connect to Pinecone.
    
    Returns:
        Pinecone: Initialized Pinecone client
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


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


if __name__ == "__main__":
    main()
s