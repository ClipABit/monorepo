# Conceptual Python Example (using Pinecone SDK)
from pinecone import Pinecone 
# from concurrent.futures import ThreadPoolExecutor # For parallel migration

# Initialize Pinecone & get indexes
pc = Pinecone(api_key="")
source_index = pc.Index("chunks-index")
target_index = pc.Index("dev-chunks")

# 1. List IDs (example for one namespace)
vector_ids = []
# index.list returns a generator yielding lists of IDs
for ids in source_index.list(namespace="__default__"):
    vector_ids.extend(ids)

# 2. Fetch & 3. Upsert (in batches)
batch_size = 100
for i in range(0, len(vector_ids), batch_size):
    batch_ids = vector_ids[i:i+batch_size]
    # Fetch data for the batch
    fetch_response = source_index.fetch(ids=batch_ids, namespace="__default__")
    vectors_to_upsert = []
    for vec_id, vec_data in fetch_response.vectors.items():
        vectors_to_upsert.append((vec_id, vec_data['values'], vec_data.get('metadata'))) # Include metadata

    # Upsert into target index
    target_index.upsert(vectors=vectors_to_upsert, namespace="__default__")
