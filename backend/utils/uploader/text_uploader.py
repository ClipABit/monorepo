"""
Text Uploader Utility

This script loads text from a file, generates CLIP embeddings for each line,
and uploads them to a Pinecone vector database for semantic search.

Features:
- Extracts text lines from a sample file
- Generates 512-dimensional CLIP embeddings using openai/clip-vit-base-patch32
- Stores embeddings in Pinecone with metadata (text content, source, chunk index)
- Supports querying for semantically similar text chunks
"""

import os
import sys
import torch

from dotenv import load_dotenv
from functools import cache
from transformers import CLIPProcessor, CLIPModel

from database.pinecone_connector import PineconeConnector

# Setup paths to backend directory
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, BACKEND_DIR)

# Setup environment variables
load_dotenv(os.path.join(BACKEND_DIR, ".env"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print(f"PINECONE_API_KEY: {'set' if PINECONE_API_KEY else 'not set'}")


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FILE_NAME = "sample.txt"
INDEX = "chunks-index"
NAMESPACE = "" # NOTE: specify namespace as the branch you are working on


@cache
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"Loaded CLIP model on {_DEVICE}")
    return model, processor


def extract_text_lines(file_path: str) -> list[str]:
    texts = []
    with open(file_path, "r") as f:
        sample_text = f.read()
        lines = sample_text.splitlines()
        
        for line in lines:
            if line.strip():  # avoid empty lines
                texts.append(line.strip())
    return texts


def generate_clip_embeddings(texts: list[str], model: CLIPModel, processor: CLIPProcessor) -> torch.Tensor:
    """
    Generate CLIP embeddings for each text line. Creates 512 dimensional embeddings.
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(_DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu() # dimensions: (num_texts, embedding_dim=512)


def main():
    model, processor = load_clip_model()

    # Load and preprocess get text lines from sample file
    sample_text_path = os.path.join(os.path.dirname(__file__), FILE_NAME)
    texts = extract_text_lines(sample_text_path)

    # Generate CLIP embeddings for the text lines
    embeddings = generate_clip_embeddings(texts, model, processor)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Upsert embeddings into Pinecone
    connector = PineconeConnector(api_key=PINECONE_API_KEY, index_name=INDEX)
    for i, chunk in enumerate(texts):
        chunk_embedding = embeddings[i]
        chunk_id = f"chunk-{i}"

        metadata = {
            "text": chunk,
            "source": FILE_NAME,
            "chunk_index": i
        }

        connector.upsert_chunk(chunk_id=chunk_id, chunk_embedding=chunk_embedding, namespace=NAMESPACE, metadata=metadata)

    # Example query
    # test_query = "woman on a train"

    # query_embedding = generate_clip_embeddings([test_query], model, processor)
    # results = connector.query_chunks(query_embedding=query_embedding, namespace=NAMESPACE)
    # print(f"Query results: {results}")


if __name__ == "__main__":
    main()
