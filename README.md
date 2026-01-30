# NFCorpus Retrieval Green Agent

A **green agent** (evaluator) for the [AgentBeats](https://agentbeats.dev) competition that evaluates **purple agents** (retrieval systems) on the NFCorpus biomedical information retrieval benchmark. Built using the [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol.

## Overview

This green agent tests a purple agent's ability to retrieve relevant biomedical documents from the NFCorpus corpus. The agent:
- Downloads and embeds the NFCorpus corpus using OpenAI embeddings
- Stores embeddings in a local Qdrant vector database
- Sends biomedical queries to purple agents via MCP server
- Collects ranked document IDs from purple agents
- Evaluates retrieval performance using standard IR metrics

## Dataset

The agent uses the [NFCorpus dataset](https://huggingface.co/datasets/BeIR/nfcorpus) from the BEIR benchmark, which contains:
- **Corpus**: 3,633 biomedical documents (PubMed abstracts)
- **Queries**: 323 natural language queries about medical topics
- **Qrels**: Ground truth relevance judgments (graded relevance: 0-2)

The dataset is downloaded during Docker build and embeddings are pre-computed using OpenAI's `text-embedding-3-small` model.

## Project Structure

```
src/
├─ server.py          # Server setup and agent card configuration
├─ executor.py        # A2A request handling
├─ agent.py           # Core evaluation logic with IR metrics
├─ messenger.py       # A2A messaging utilities
├─ data_loader.py     # NFCorpus dataset loading and Qdrant setup
├─ schemas.py         # Pydantic models for requests/responses
└─ mcp_server.py      # MCP server for purple agents to query Qdrant
tests/
└─ test_agent.py      # Agent tests
Dockerfile            # Docker configuration with data download
pyproject.toml        # Python dependencies
```

## Assessment Request Format

The green agent expects requests in the following format:

```json
{
  "participants": {
    "retrieval_agent": "http://purple-agent:9010"
  },
  "config": {
    "num_queries": 50,
    "top_k": 10,
    "random_seed": 777
  }
}
```

### Configuration Parameters

- **num_queries** (required): Number of queries to evaluate (1-323)
- **top_k** (required): Number of documents to retrieve per query (1-100)
- **random_seed** (optional): Integer seed for reproducible query sampling

## Purple Agent Requirements

Purple agents being evaluated must:

1. **Accept biomedical queries as JSON**:
```json
{
  "query": "calcium and bone health",
  "top_k": 10
}
```

2. **Have access to MCP server** with `search_nfcorpus` tool:
   - The green agent provides a Qdrant vector database with embedded NFCorpus documents
   - Purple agents use the MCP server to search the vector database
   - The MCP server runs on port 8000 within the Docker network

3. **Respond with ranked document IDs**:
```json
{
  "doc_ids": ["MED-123", "MED-456", "MED-789", "MED-234", "MED-567"]
}
```

The `doc_ids` list should be ordered by relevance (most relevant first) and contain up to `top_k` document IDs.

## Results Format

The green agent returns comprehensive information retrieval metrics:

```json
{
  "assessment_type": "nfcorpus_retrieval",
  "num_queries": 50,
  "evaluated_queries": 50,
  "top_k": 10,
  "metrics": {
    "ndcg@10": 0.3421,
    "mrr@10": 0.4821,
    "precision@10": 0.2840,
    "recall@10": 0.4521,
  },
  "execution_time_seconds": 62.5,
  "successful_queries": 50,
  "num_of_vector_sea": 0
}
```

## Environment Variables

The green agent requires the following environment variable:

- **OPENAI_API_KEY** (required): OpenAI API key for embedding corpus documents and queries

## Running Locally

```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Install dependencies
uv sync

# Prepare data (download, embed, setup Qdrant)
uv run src/prepare_data.py

# Start MCP server (in one terminal)
uv run src/mcp_server.py

# Start A2A server (in another terminal)
uv run src/server.py
```

## Running with Docker

The Docker build process automatically downloads the NFCorpus dataset, embeds documents using OpenAI, and sets up the Qdrant vector database.

```bash
# Build the image with OpenAI API key
docker build --build-arg OPENAI_API_KEY=your_api_key_here -t nfcorpus-evaluator .

# Or using Docker secrets (recommended for production)
echo "your_api_key_here" > openai_key.txt
docker build --secret id=openai_key,src=openai_key.txt -t nfcorpus-evaluator .
rm openai_key.txt

# Run the container
docker run -p 9009:9009 -p 8000:8000 -e OPENAI_API_KEY=your_api_key_here nfcorpus-evaluator
```

The container exposes two ports:
- **9009**: A2A green agent server
- **8000**: MCP server for purple agents to query Qdrant

## MCP Server API

The green agent provides an MCP server that purple agents can use to search the NFCorpus vector database. The MCP server runs on port 8000 and provides the following endpoint:

### POST /search_nfcorpus

Search the NFCorpus corpus using semantic similarity.

**Request Body:**
```json
{
  "query": "calcium and bone health",
  "top_k": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "doc_id": "MED-123",
      "title": "Calcium supplementation and bone density",
      "text": "Full document text...",
      "score": 0.8542
    }
  ]
}
```

Purple agents should:
1. Receive a query from the green agent via A2A protocol
2. Call the MCP server's `/search_nfcorpus` endpoint to retrieve relevant documents
3. Process the results (e.g., re-rank, filter, analyze)
4. Return a list of document IDs ordered by relevance

## Quick Start with Docker Compose

```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Build and start the services
docker-compose up --build

# The green agent will be available at:
# - A2A Server: http://localhost:9009
# - MCP Server: http://localhost:8000
```

## Example Usage

See `examples/purple_agent_example.py` for a complete example of how a purple agent should interact with the green agent.

**Basic workflow:**

1. Green agent sends a retrieval query to purple agent:
```json
{"query": "calcium and bone health", "top_k": 10}
```

2. Purple agent queries the MCP server:
```python
import httpx

async def search(query: str, top_k: int):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/search_nfcorpus",
            json={"query": query, "top_k": top_k}
        )
        return response.json()
```

3. Purple agent returns ranked document IDs:
```json
{"doc_ids": ["MED-123", "MED-456", "MED-789"]}
```

4. Green agent evaluates the results using IR metrics

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
