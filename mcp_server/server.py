#!/usr/bin/env python3
"""
FastMCP server for NFCorpus Qdrant vector database.
Exposes search functionality via MCP protocol for purple agents.
"""

import os
from typing import List, Dict, Any
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nfcorpus")

mcp = FastMCP("NFCorpus Search")

model = None
client = None
qdrant_calls_total = 0


def get_model():
    """Lazy load the embedding model."""
    global model
    if model is None:
        print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model


def get_client():
    """Lazy load the Qdrant client."""
    global client
    if client is None:
        print(f"Connecting to Qdrant at {QDRANT_URL}")
        client = QdrantClient(url=QDRANT_URL)
    return client


def _search_nfcorpus_impl(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Implementation of search functionality."""
    try:
        embedding_model = get_model()
        qdrant_client = get_client()
        global qdrant_calls_total
        
        query_embedding = embedding_model.encode(query)
        
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k
        )
        qdrant_calls_total += 1
        
        results = []
        for result in search_results.points:
            text = result.payload.get("text", "")
            # Provide first 2000 chars for LLM context (or full text if shorter)
            text_preview = text[:2000] + ("..." if len(text) > 2000 else "")
            results.append({
                "doc_id": result.payload["doc_id"],
                "score": float(result.score),
                "title": result.payload.get("title", ""),
                "text": text_preview
            })
        
        return {"results": results, "qdrant_calls": 1}
    
    except Exception as e:
        return {
            "error": str(e),
            "results": []
        }


def _health_check_impl() -> Dict[str, Any]:
    """Implementation of health check functionality."""
    try:
        qdrant_client = get_client()
        global qdrant_calls_total
        
        # Try to get collection info
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        
        # Get list of all collections
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        return {
            "status": "healthy",
            "qdrant_url": QDRANT_URL,
            "collection": COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "available_collections": ", ".join(collection_names),
            "qdrant_calls_total": qdrant_calls_total
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def search_nfcorpus(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search NFCorpus biomedical documents using vector similarity.
    
    Args:
        query: Search query text
        top_k: Number of results to return (default: 5)
    
    Returns:
        Dictionary with 'results' key containing list of documents with doc_id and score
    """
    return _search_nfcorpus_impl(query, top_k)


@mcp.tool()
def health_check() -> Dict[str, Any]:
    """
    Check the health status of the MCP server and Qdrant connection.
    
    Returns:
        Dictionary with status information
    """
    return _health_check_impl()


if __name__ == "__main__":
    print(f"Starting NFCorpus MCP Server")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    
    # FastMCP runs with stdio transport by default
    # For HTTP access, we need to create a simple wrapper
    from fastapi import FastAPI, Body
    from pydantic import BaseModel
    import uvicorn
    
    app = FastAPI(title="NFCorpus MCP Server")
    
    class SearchRequest(BaseModel):
        query: str
        top_k: int = 5
    
    # Expose tools as HTTP endpoints
    @app.post("/search_nfcorpus")
    async def search_endpoint(request: SearchRequest):
        return _search_nfcorpus_impl(request.query, request.top_k)
    
    @app.post("/health_check")
    async def health_endpoint():
        return _health_check_impl()
    
    @app.get("/")
    async def root():
        return {
            "name": "NFCorpus MCP Server",
            "tools": ["search_nfcorpus", "health_check"],
            "qdrant_url": QDRANT_URL,
            "collection": COLLECTION_NAME,
            "qdrant_calls_total": qdrant_calls_total
        }
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
