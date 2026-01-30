import asyncio
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from openai import OpenAI
import os


qdrant_client = None
openai_client = None
COLLECTION_NAME = "nfcorpus"
search_call_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, openai_client
    
    qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_data")
    qdrant_client = QdrantClient(path=qdrant_path)
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    openai_client = OpenAI(api_key=openai_api_key)
    
    yield
    

app = FastAPI(title="NFCorpus MCP Server", version="1.0.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")


class SearchResult(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


@app.post("/search_nfcorpus", response_model=SearchResponse)
async def search_nfcorpus(request: SearchRequest) -> SearchResponse:
    """Search the NFCorpus vector database using semantic similarity.
    
    This tool allows purple agents to retrieve relevant biomedical documents
    from the NFCorpus corpus based on a query.
    """
    global search_call_count
    search_call_count += 1
    
    response = openai_client.embeddings.create(
        input=request.query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=request.top_k
    ).points
    
    results = []
    for hit in search_results:
        results.append(SearchResult(
            doc_id=hit.payload["doc_id"],
            title=hit.payload["title"],
            text=hit.payload["text"],
            score=hit.score
        ))
    
    return SearchResponse(results=results)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/stats")
async def get_stats():
    """Get MCP server statistics including search call count."""
    return {"search_call_count": search_call_count}


@app.post("/stats/reset")
async def reset_stats():
    """Reset the search call counter."""
    global search_call_count
    search_call_count = 0
    return {"search_call_count": search_call_count}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
