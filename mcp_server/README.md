# NFCorpus MCP Server

FastMCP-based server that exposes Qdrant vector database search functionality for NFCorpus documents.

## Overview

This MCP server provides a `search_nfcorpus` tool that purple agents can use to retrieve relevant biomedical documents from the NFCorpus collection.

## Tools

### search_nfcorpus

Search NFCorpus biomedical documents using vector similarity.

**Parameters:**
- `query` (string): Search query text
- `top_k` (integer, default: 5): Number of results to return

**Returns:**
```json
{
  "results": [
    {
      "doc_id": "MED-123",
      "score": 0.856,
      "title": "Document title",
      "text": "Document text preview..."
    }
  ]
}
```

### health_check

Check if the MCP server and Qdrant are healthy.

**Returns:**
```json
{
  "status": "healthy",
  "collection": "nfcorpus",
  "documents": "3633",
  "qdrant_url": "http://qdrant:6333"
}
```

## Configuration

Environment variables:
- `QDRANT_URL`: Qdrant server URL (default: `http://localhost:6333`)
- `COLLECTION_NAME`: Collection name in Qdrant (default: `nfcorpus`)

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export QDRANT_URL=http://localhost:6333
export COLLECTION_NAME=nfcorpus

# Run server
python server.py
```

## Running with Docker

```bash
# Build image
docker build -t nfcorpus-mcp .

# Run container
docker run -p 8000:8000 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e COLLECTION_NAME=nfcorpus \
  nfcorpus-mcp
```

## Running with Docker Compose

The MCP server is included in the main `docker-compose.yml`:

```bash
docker-compose up mcp-server
```

## Testing

```bash
# Test search
curl -X POST http://localhost:8000/search_nfcorpus \
  -H "Content-Type: application/json" \
  -d '{"query": "calcium and bone health", "top_k": 5}'

# Test health check
curl http://localhost:8000/health_check
```
