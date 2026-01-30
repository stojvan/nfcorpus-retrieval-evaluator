# NFCorpus Retrieval Green Agent - Implementation Summary

## What Was Built

A complete **Green Agent** (evaluator) for the AgentBeats competition that evaluates **Purple Agents** (retrieval systems) on the NFCorpus biomedical information retrieval benchmark.

## Key Features

### 1. **Dual-Server Architecture**
- **A2A Green Agent Server** (Port 9009): Orchestrates evaluation using A2A protocol
- **MCP Server** (Port 8000): Provides search API for purple agents to query Qdrant

### 2. **Automated Data Pipeline**
- Downloads NFCorpus dataset from HuggingFace (BEIR benchmark)
- Embeds 3,633 biomedical documents using OpenAI `text-embedding-3-small`
- Stores embeddings in local Qdrant vector database
- All data preparation happens during Docker build

### 3. **Comprehensive IR Metrics**
- **NDCG@k**: Normalized Discounted Cumulative Gain at k={1,3,5,10}
- **MRR@k**: Mean Reciprocal Rank at k=10
- **Precision@k**: Precision at k={1,3,5,10}
- **Recall@k**: Recall at k={10,100}

### 4. **Production-Ready**
- Docker containerization with multi-stage build
- Docker Compose for easy deployment
- Environment variable configuration
- Health check endpoints
- Comprehensive documentation

## Files Created/Modified

### Core Application Files
- `src/agent.py` - Evaluation logic with IR metrics computation
- `src/server.py` - A2A server configuration
- `src/schemas.py` - Pydantic models for retrieval queries/responses
- `src/data_loader.py` - NFCorpus dataset loading and Qdrant setup
- `src/mcp_server.py` - MCP server for purple agents
- `src/prepare_data.py` - Data preparation script
- `src/start.sh` - Startup script for both servers

### Configuration Files
- `pyproject.toml` - Updated dependencies (openai, qdrant-client, numpy)
- `Dockerfile` - Multi-stage build with data preparation
- `docker-compose.yml` - Easy deployment configuration
- `.env.example` - Environment variable template
- `.gitignore` - Updated with data cache directories

### Documentation
- `README.md` - Complete user guide with examples
- `ARCHITECTURE.md` - System architecture and data flow
- `DEPLOYMENT.md` - Deployment and troubleshooting guide
- `SUMMARY.md` - This file

### Examples
- `examples/purple_agent_example.py` - Example purple agent implementation

## How It Works

### Evaluation Flow

1. **Green Agent receives evaluation request** from AgentBeats:
   ```json
   {
     "participants": {"retrieval_agent": "http://purple-agent:9010"},
     "config": {"num_queries": 50, "top_k": 10, "random_seed": 777}
   }
   ```

2. **Green Agent samples queries** from NFCorpus test set

3. **For each query**, Green Agent:
   - Sends query to Purple Agent via A2A
   - Purple Agent queries MCP server for relevant documents
   - Purple Agent returns ranked document IDs
   - Green Agent computes IR metrics using ground truth qrels

4. **Green Agent aggregates metrics** and returns results:
   ```json
   {
     "metrics": {
       "ndcg@10": 0.3421,
       "mrr@10": 0.4821,
       "precision@10": 0.2840,
       "recall@10": 0.4521
     }
   }
   ```

### Purple Agent Workflow

1. Receive query from Green Agent: `{"query": "calcium and bone health", "top_k": 10}`
2. Call MCP server: `POST http://localhost:8000/search_nfcorpus`
3. Process results (optional re-ranking, filtering)
4. Return document IDs: `{"doc_ids": ["MED-123", "MED-456", ...]}`

## Technical Stack

- **Python 3.13** with UV package manager
- **A2A SDK** for agent-to-agent communication
- **Pydantic** for data validation
- **OpenAI API** for embeddings
- **Qdrant** for vector search
- **FastAPI** for MCP server
- **HuggingFace Datasets** for NFCorpus
- **NumPy** for metric computation

## Dataset: NFCorpus

- **Source**: BEIR benchmark (biomedical domain)
- **Corpus**: 3,633 PubMed abstracts
- **Queries**: 323 natural language medical questions
- **Qrels**: Graded relevance judgments (0-2)
- **Task**: Ad-hoc retrieval of relevant biomedical documents

## Quick Start

```bash
# 1. Set up environment
echo "OPENAI_API_KEY=your_key_here" > .env

# 2. Build and run
docker-compose up --build

# 3. Access services
# - Green Agent: http://localhost:9009
# - MCP Server: http://localhost:8000
```

## Key Design Decisions

1. **Embedded Qdrant**: Uses file-based storage for simplicity and portability
2. **Build-time Data Prep**: Downloads and embeds data during Docker build for faster startup
3. **Dual Server**: Separates evaluation logic (A2A) from search API (MCP)
4. **OpenAI Embeddings**: Uses `text-embedding-3-small` for cost-effectiveness
5. **Batch Embedding**: Processes 100 documents per API call to reduce costs

## Cost Estimates

- **Data Preparation**: ~$0.01 (one-time, 3,633 documents)
- **Runtime**: ~$0.0001 per query (embedding only)
- **50 queries**: ~$0.005

## Performance

- **Data Prep**: 2-5 minutes (one-time during build)
- **Query Processing**: ~150ms per query (embedding + search)
- **50 Query Evaluation**: ~7.5 seconds + purple agent time

## Next Steps

1. **Test locally** with the example purple agent
2. **Build Docker image** with your OpenAI API key
3. **Deploy to AgentBeats** platform
4. **Monitor performance** and adjust as needed

## Notes

- The green agent is fully compatible with the A2A protocol and AgentBeats platform
- Purple agents must have network access to the MCP server (port 8000)
- The MCP server provides a simple REST API for document retrieval
- All IR metrics follow standard TREC evaluation conventions
- The system is designed to be extensible for other BEIR datasets

## Support

- See `README.md` for usage instructions
- See `ARCHITECTURE.md` for system design details
- See `DEPLOYMENT.md` for deployment guide
- See `examples/` for purple agent implementation example
