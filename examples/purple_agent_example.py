"""
Example Purple Agent for NFCorpus Retrieval Evaluation

This example demonstrates how a purple agent should interact with the green agent
and use the MCP server to retrieve relevant documents.
"""

import httpx
from pydantic import BaseModel
from typing import List


class RetrievalQuery(BaseModel):
    query: str
    top_k: int


class RetrievalResponse(BaseModel):
    doc_ids: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int


class SearchResult(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


async def search_nfcorpus(query: str, top_k: int, mcp_server_url: str = "http://localhost:8000") -> SearchResponse:
    """Call the MCP server to search the NFCorpus database."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{mcp_server_url}/search_nfcorpus",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return SearchResponse(**response.json())


async def handle_retrieval_query(query_json: str, mcp_server_url: str = "http://localhost:8000") -> str:
    """
    Handle a retrieval query from the green agent.
    
    Args:
        query_json: JSON string with query and top_k
        mcp_server_url: URL of the MCP server
        
    Returns:
        JSON string with doc_ids
    """
    query_obj = RetrievalQuery.model_validate_json(query_json)
    
    search_results = await search_nfcorpus(query_obj.query, query_obj.top_k, mcp_server_url)
    
    doc_ids = [result.doc_id for result in search_results.results]
    
    response = RetrievalResponse(doc_ids=doc_ids)
    return response.model_dump_json()


if __name__ == "__main__":
    import asyncio
    
    example_query = '{"query": "calcium and bone health", "top_k": 10}'
    
    result = asyncio.run(handle_retrieval_query(example_query))
    print(result)
