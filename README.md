# NFCorpus Retrieval Evaluator

A **green agent** (evaluator) for the [AgentBeats](https://agentbeats.dev) competition that evaluates **purple agents** (retrieval systems) on the NFCorpus biomedical information retrieval benchmark.

## Overview

This green agent evaluates how well purple agents can retrieve relevant biomedical documents from the NFCorpus dataset (part of the BEIR benchmark). It measures performance using **NDCG@5** (Normalized Discounted Cumulative Gain at rank 5).

### Key Features

- 🔬 **NFCorpus Dataset**: ~3,633 biomedical documents, 323 test queries
- 📊 **NDCG@5 Metric**: Industry-standard ranking evaluation
- 🎲 **Reproducible Sampling**: Random seed for consistent query selection
- 🐳 **Docker Compose**: Complete infrastructure (Qdrant + MCP + Green Agent)
- 🔌 **MCP Protocol**: Purple agents access vector database via Model Context Protocol
- ✅ **Type-Safe**: Pydantic schemas for all data structures

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Green Agent (Evaluator)                   │
│  - Samples queries with random seed                         │
│  - Sends queries to purple agent via A2A protocol           │
│  - Receives top-5 document IDs from purple agent            │
│  - Calculates NDCG@5 score                                  │
│  - Reports evaluation results                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ A2A Protocol
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Purple Agent (Retrieval)                   │
│  - Receives query from green agent                          │
│  - Searches Qdrant vector database via MCP                  │
│  - Returns list of 5 most relevant document IDs             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ MCP Protocol
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Qdrant Vector Database                    │
│  - Stores NFCorpus documents as embeddings                  │
│  - sentence-transformers/all-MiniLM-L6-v2 (384-dim)        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prepare Data

Download NFCorpus dataset from BEIR:

```bash
# Install dependencies
uv sync

# Download and prepare data
uv run python scripts/prepare_data.py
```

This creates:
- `data/corpus.jsonl` - All NFCorpus documents
- `data/test_queries.jsonl` - Test queries
- `data/test_qrels.jsonl` - Ground truth relevance judgments

### 2. Start Infrastructure

Start Qdrant and MCP server:

```bash
docker-compose up -d qdrant mcp-server
```

### 3. Index Documents

Generate embeddings and populate Qdrant:

```bash
uv run python scripts/index_documents.py
```

This indexes ~3,633 documents with 384-dimensional embeddings.

### 4. Start Green Agent

```bash
# Local development
uv run src/server.py

# Or with Docker
docker-compose up green-agent
```

### 5. Run Evaluation

Send an evaluation request:

```bash
curl -X POST http://localhost:9009/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "participants": {
      "retrieval_agent": "http://purple-agent:9010"
    },
    "config": {
      "num_queries": 100,
      "top_k": 5,
      "random_seed": 42
    }
  }'
```

## Configuration

### Evaluation Parameters

- **num_queries** (1-323): Number of queries to evaluate (default: 100)
- **top_k** (1-100): Number of documents to retrieve per query (default: 5)
- **random_seed** (integer): Random seed for reproducible query sampling (default: 42)

### Example Request

```json
{
  "participants": {
    "retrieval_agent": "http://purple-agent:9010"
  },
  "config": {
    "num_queries": 100,
    "top_k": 5,
    "random_seed": 42
  }
}
```

## Building a Purple Agent

Purple agents must:
1. Implement A2A protocol
2. Accept `QueryRequest` format: `{"query": "...", "top_k": 5}`
3. Return `RetrievalResponse` format: `{"doc_ids": ["MED-123", ...]}`
4. Access MCP server at `http://mcp-server:8000` for vector search

**Full specification**: See [`docs/purple_agent_spec.md`](docs/purple_agent_spec.md)

### MCP Search Tool

Purple agents can use the `search_nfcorpus` tool:

```python
# Call MCP tool
response = await mcp_client.call_tool(
    "search_nfcorpus",
    {"query": "calcium and bone health", "top_k": 5}
)

# Extract document IDs
doc_ids = [result["doc_id"] for result in response["results"]]
```

## Evaluation Metrics

### Primary Metric: NDCG@5

**NDCG** (Normalized Discounted Cumulative Gain) measures ranking quality:
- **Range**: 0.0 to 1.0
- **1.0** = Perfect ranking
- **0.0** = No relevant documents retrieved
- **Considers**: Both relevance scores and ranking position

NFCorpus uses 3-level relevance:
- **2** = Highly relevant
- **1** = Partially relevant
- **0** = Not relevant

### Output Metrics

The evaluation report includes:
- **Mean NDCG@5**: Average across all queries
- **Median NDCG@5**: Median performance
- **Std NDCG@5**: Standard deviation
- **Min/Max NDCG@5**: Range of performance
- **Success Rate**: Fraction of queries with NDCG > 0

### Example Output

```
NFCorpus Retrieval Evaluation Results
============================================================
Total Queries: 100
Mean NDCG@5: 0.4560
Median NDCG@5: 0.4230
Std NDCG@5: 0.1230
Min NDCG@5: 0.0000
Max NDCG@5: 1.0000
Success Rate: 87.00%
============================================================
Configuration:
  - Queries: 100
  - Top-K: 5
  - Random Seed: 42
```

## Project Structure

```
nfcorpus-retrieval-evaluator/
├── data/                           # NFCorpus dataset
│   ├── corpus.jsonl
│   ├── test_queries.jsonl
│   └── test_qrels.jsonl
├── src/                            # Green agent implementation
│   ├── agent.py                    # Main evaluation logic
│   ├── server.py                   # A2A server & agent card
│   ├── schemas.py                  # Pydantic data models
│   ├── metrics.py                  # NDCG calculation
│   ├── executor.py                 # Request handler
│   └── messenger.py                # A2A messaging
├── mcp_server/                     # FastMCP server for Qdrant
│   ├── server.py
│   ├── Dockerfile
│   └── README.md
├── docs/                           # Documentation
│   ├── purple_agent_spec.md        # Purple agent interface spec
│   └── examples/
├── scripts/                        # Data preparation scripts
│   ├── prepare_data.py
│   └── index_documents.py
├── tests/                          # Test suite
│   ├── test_schemas.py
│   ├── test_metrics.py
│   └── test_agent.py
├── docker-compose.yml              # Multi-service orchestration
├── Dockerfile                      # Green agent Docker config
├── pyproject.toml                  # Python dependencies
└── TASK.md                         # Implementation roadmap
```

## Testing

### Run Unit Tests

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_metrics.py

# Run with coverage
uv run pytest --cov=src
```

### Test Reproducibility

Same random seed should produce identical results:

```bash
# Run evaluation twice with same seed
# Results should be identical
```

## Docker Compose Services

The `docker-compose.yml` defines three services:

### 1. Qdrant (Vector Database)
- **Port**: 6333
- **Volume**: Persistent storage for vectors
- **Image**: `qdrant/qdrant:latest`

### 2. MCP Server (FastMCP)
- **Port**: 8000
- **Depends on**: Qdrant
- **Tools**: `search_nfcorpus`, `health_check`

### 3. Green Agent (Evaluator)
- **Port**: 9009
- **Depends on**: Qdrant, MCP Server
- **Volumes**: `./data` mounted read-only

### Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose build
docker-compose up -d
```

## Development

### Local Development Setup

```bash
# Install dependencies
uv sync

# Start Qdrant and MCP server
docker-compose up -d qdrant mcp-server

# Run green agent locally
uv run src/server.py
```

### Adding New Metrics

1. Add metric function to `src/metrics.py`
2. Update `QueryResult` schema in `src/schemas.py`
3. Calculate metric in `src/agent.py`
4. Add tests in `tests/test_metrics.py`

## Dataset Information

**NFCorpus** (Nutrition Facts Corpus)
- **Domain**: Biomedical/nutrition
- **Documents**: 3,633 PubMed articles
- **Queries**: 323 NutritionFacts.org queries
- **Relevance**: 3-level judgments (0, 1, 2)
- **Source**: [BEIR Benchmark](https://huggingface.co/datasets/BeIR/nfcorpus)

## Troubleshooting

### Qdrant Connection Issues

```bash
# Check Qdrant health
curl http://localhost:6333/health

# View Qdrant logs
docker-compose logs qdrant
```

### MCP Server Issues

```bash
# Test MCP server
curl http://localhost:8000/health_check

# View MCP logs
docker-compose logs mcp-server
```

### Data Not Found

Ensure data files exist:
```bash
ls data/
# Should show: corpus.jsonl, test_queries.jsonl, test_qrels.jsonl
```

## Contributing

This is a competition submission for AgentBeats. For questions:
- Review [`TASK.md`](TASK.md) for implementation details
- Check [`docs/purple_agent_spec.md`](docs/purple_agent_spec.md) for purple agent interface

## License

MIT License - See LICENSE file for details

## References

- **AgentBeats**: https://agentbeats.dev
- **A2A Protocol**: https://a2a-protocol.org/latest/
- **BEIR Benchmark**: https://github.com/beir-cellar/beir
- **NFCorpus Dataset**: https://huggingface.co/datasets/BeIR/nfcorpus
- **Qdrant**: https://qdrant.tech/
- **FastMCP**: https://github.com/jlowin/fastmcp
