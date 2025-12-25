## Index File System

The backend uses `modal.Dict` for persistent ID management:

- **`database/index_file.py`**: `IndexFileConnector` class for chunk/cluster ID generation

### Using the IndexFileConnector

```python
from database.index_file import index_connector

# Get next available IDs
chunk_id = index_connector.get_next_chunk_id()      # Returns: "chunk_000001"
cluster_id = index_connector.get_next_cluster_id()  # Returns: "cluster_000001"

# Link chunks to clusters
index_connector.link_chunk_to_cluster(chunk_id, cluster_id)

# For face embeddings:
# Get associated cluster id with a chunk id
chunk_id = index_connector.get_cluster_for_chunk()      # Returns: "cluster_000001"
# Get associated chunk ids to cluster id
cluster_id = index_connector.get_next_cluster_id()  # Returns: "[chunk_000001, chunk_000002, chunk_000003"

# Get statistics
stats = index_connector.get_stats()
```
