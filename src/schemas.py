"""
Pydantic schemas for NFCorpus retrieval evaluation.
Defines all data models for requests, responses, and evaluation results.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any


class EvalConfig(BaseModel):
    """Configuration for evaluation run."""
    num_queries: int = Field(default=100, ge=1, le=323, description="Number of queries to evaluate")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of documents to retrieve per query")
    random_seed: int = Field(default=42, description="Random seed for reproducible query sampling")
    mcp_server_url: Optional[str] = Field(default=None, description="Optional MCP server base URL (e.g. http://localhost:8000) used to read authoritative qdrant_calls_total via /health_check")
    verbose: bool = Field(default=False, description="If true, print detailed debugging logs")


class EvalRequest(BaseModel):
    """Request format sent by AgentBeats platform to green agents."""
    participants: Dict[str, HttpUrl]
    config: Dict[str, Any]


class QueryRequest(BaseModel):
    """Message sent to purple agent for document retrieval."""
    query: str = Field(description="Search query text")
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class RetrievalResponse(BaseModel):
    """Expected response from purple agent."""
    doc_ids: List[str] = Field(description="List of retrieved document IDs in ranked order")


class Query(BaseModel):
    """NFCorpus query."""
    id: str = Field(alias='_id')
    text: str
    
    class Config:
        populate_by_name = True


class Document(BaseModel):
    """NFCorpus document."""
    id: str = Field(alias='_id')
    title: str
    text: str
    
    class Config:
        populate_by_name = True


class QrelEntry(BaseModel):
    """Relevance judgment entry."""
    query_id: str = Field(alias='query-id')
    corpus_id: str = Field(alias='corpus-id')
    score: int = Field(description="Relevance score (0=not relevant, 1=partially relevant, 2=highly relevant)")
    
    class Config:
        populate_by_name = True


class QueryResult(BaseModel):
    """Evaluation result for a single query."""
    query_id: str
    query_text: str
    retrieved_docs: List[str]
    ndcg_at_5: float
    mrr_at_5: float
    precision_at_5: float
    recall_at_5: float
    relevant_docs_retrieved: int
    total_relevant_docs: int


class EvaluationSummary(BaseModel):
    """Aggregate evaluation metrics."""
    total_queries: int
    successful_queries: int = Field(description="Number of queries that completed successfully")
    failed_queries: int = Field(description="Number of queries that failed")
    qdrant_calls_total: int = Field(description="Total number of Qdrant calls made across all evaluated queries")
    mean_ndcg_at_5: float
    mean_mrr_at_5: float
    mean_precision_at_5: float
    mean_recall_at_5: float
    median_ndcg_at_5: float
    std_ndcg_at_5: float
    min_ndcg_at_5: float
    max_ndcg_at_5: float
    success_rate: float = Field(description="Fraction of queries with NDCG > 0")
    total_time_seconds: float = Field(description="Total evaluation time in seconds")
    queries_per_second: float = Field(description="Average queries processed per second")


class EvaluationReport(BaseModel):
    """Complete evaluation report."""
    summary: EvaluationSummary
    per_query_results: List[QueryResult]
    config: EvalConfig
    random_seed_used: int
