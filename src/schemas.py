from pydantic import BaseModel, Field, HttpUrl 
from typing import Any


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class RetrievalQuery(BaseModel):
    """A biomedical query to retrieve relevant documents."""
    query: str = Field(..., description="Biomedical query text")
    top_k: int = Field(..., ge=1, le=100, description="Number of documents to retrieve")


class RetrievalResponse(BaseModel):
    """Response from purple agent with ranked document IDs."""
    doc_ids: list[str] = Field(..., description="List of document IDs ranked by relevance")
