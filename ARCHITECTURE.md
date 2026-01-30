# NFCorpus Retrieval Green Agent Architecture

## Overview

This green agent evaluates purple agents (retrieval systems) on the NFCorpus biomedical information retrieval benchmark using the A2A protocol and AgentBeats platform.

## System Components

### 1. A2A Green Agent Server (Port 9009)
- **File**: `src/server.py`, `src/executor.py`, `src/agent.py`
- **Purpose**: Receives evaluation requests from AgentBeats, orchestrates the evaluation process
- **Protocol**: A2A (Agent-to-Agent)
- **Responsibilities**:
  - Load NFCorpus dataset (corpus, queries, qrels)
  - Sample queries based on configuration
  - Send queries to purple agents
  - Collect and validate responses
  - Compute IR metrics (NDCG@k, MRR@k, Precision@k, Recall@k)
  - Report results back to AgentBeats

### 2. MCP Server (Port 8000)
- **File**: `src/mcp_server.py`
- **Purpose**: Provides search API for purple agents to query the Qdrant vector database
- **Protocol**: HTTP REST API
- **Endpoints**:
  - `POST /search_nfcorpus`: Semantic search over NFCorpus corpus
  - `GET /health`: Health check endpoint
- **Responsibilities**:
  - Embed queries using OpenAI API
  - Query Qdrant vector database
  - Return ranked search results with document metadata

### 3. Data Preparation Pipeline
- **File**: `src/prepare_data.py`, `src/data_loader.py`
- **Purpose**: Download, embed, and index NFCorpus corpus
- **Execution**: Runs during Docker build
- **Steps**:
  1. Download NFCorpus from HuggingFace (BEIR dataset)
  2. Embed all corpus documents using OpenAI `text-embedding-3-small`
  3. Create Qdrant collection and insert embeddings
  4. Store Qdrant data locally for runtime access

## Data Flow

```
AgentBeats Platform
    |
    | (A2A Request)
    v
Green Agent (Port 9009)
    |
    | (Retrieval Query)
    v
Purple Agent (External)
    |
    | (Search Request)
    v
MCP Server (Port 8000)
    |
    | (Vector Search)
    v
Qdrant Database
    |
    | (Search Results)
    v
MCP Server
    |
    | (Document IDs)
    v
Purple Agent
    |
    | (Ranked Doc IDs)
    v
Green Agent
    |
    | (IR Metrics)
    v
AgentBeats Platform
```

## Evaluation Metrics

The green agent computes the following standard information retrieval metrics:

- **NDCG@k** (Normalized Discounted Cumulative Gain): Measures ranking quality with graded relevance
- **MRR@k** (Mean Reciprocal Rank): Measures rank of first relevant document
- **Precision@k**: Fraction of retrieved documents that are relevant
- **Recall@k**: Fraction of relevant documents that are retrieved

Metrics are computed at k ∈ {1, 3, 5, 10, 100} depending on the configuration.

## NFCorpus Dataset

- **Source**: BEIR benchmark (BeIR/nfcorpus on HuggingFace)
- **Domain**: Biomedical (PubMed abstracts)
- **Size**: 
  - 3,633 documents
  - 323 queries
  - Graded relevance judgments (0-2)
- **Task**: Ad-hoc retrieval of relevant biomedical documents

## Docker Build Process

1. Install Python dependencies via `uv`
2. Download NFCorpus dataset from HuggingFace
3. Embed corpus documents using OpenAI API (requires `OPENAI_API_KEY`)
4. Setup Qdrant vector database with embeddings
5. Copy startup script
6. Expose ports 9009 (A2A) and 8000 (MCP)

## Runtime Process

1. Start MCP server in background
2. Start A2A green agent server
3. Both services run concurrently
4. Purple agents can query MCP server while green agent evaluates them

## Purple Agent Requirements

Purple agents must:
1. Accept A2A messages with retrieval queries
2. Have network access to MCP server (port 8000)
3. Call `/search_nfcorpus` endpoint to retrieve documents
4. Process and rank results (optional re-ranking)
5. Return document IDs in relevance order via A2A response

## Security Considerations

- OpenAI API key required for embedding (stored as environment variable)
- MCP server should only be accessible within Docker network in production
- No authentication on MCP server (assumes trusted network)
- Qdrant runs in embedded mode (file-based storage)

## Scalability Notes

- Qdrant embedded mode suitable for NFCorpus size (~3.6K documents)
- For larger corpora, consider Qdrant server mode
- OpenAI API rate limits may affect data preparation time
- Batch embedding reduces API calls (100 docs per batch)
