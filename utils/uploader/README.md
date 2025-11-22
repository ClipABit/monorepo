# Text Uploader Utility
A utility script for uploading text chunks to the Pinecone vector database with CLIP embeddings to test the search pipeline.

## Setup

### 1. Install Dependencies and Setup venv

From the `/utils` directory, run the following:

```bash
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the `/backend` directory and add your Pinecone API key to the `.env` file:

```env
PINECONE_API_KEY=your-pinecone-api-key-here
```

To obtain your Pinecone API key:
1. Log in to [Pinecone Console](https://app.pinecone.io/)
2. Navigate to "API Keys" in the sidebar
3. Copy your API key

### 3. Configure Index and Namespace

Before running the script, edit `text_uploader.py` to specify your Pinecone index name and namespace:

```python
INDEX = "chunks-index"    # Set to your Pinecone index name
NAMESPACE = ""      # Set to your desired namespace (will use "__default__" if left blank)
```

Namespaces are created automatically on first upsert. All upsert and query operations will use the specified namespace.

### 4. Prepare Text Data

Add text content to `sample.txt` in the same directory as the script: Each line will be treated as a separate chunk. Empty lines are automatically skipped.

## Usage

### Running the Script

From the `/utils` directory, run:

```bash
uv run python3 ./uploader/text_uploader.py
```

You should logs of the embeddings being generated and the records being upserted to the database.


## How It Works

1. **Text Extraction**: Reads `sample.txt` and splits content into non-empty lines
2. **Embedding Generation**: Each line is processed through CLIP to create a 512-dimensional vector
3. **Normalization**: Embeddings are L2-normalized for cosine similarity
4. **Upload**: Each embedding is stored in Pinecone with metadata:
   - `text`: Original text content
   - `source`: Source filename
   - `chunk_index`: Line position in file
5. **Query** (optional): Search for similar chunks using natural language queries
