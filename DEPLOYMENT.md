# Deployment Guide

## Prerequisites

- Docker with BuildKit support
- OpenAI API key
- At least 2GB free disk space for data and embeddings

## Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-...
```

## Local Development

### Option 1: Using UV (Recommended for Development)

```bash
# Install dependencies
uv sync

# Prepare data (one-time setup)
export OPENAI_API_KEY=your_key_here
uv run src/prepare_data.py

# Terminal 1: Start MCP server
uv run src/mcp_server.py

# Terminal 2: Start A2A server
uv run src/server.py
```

### Option 2: Using Docker Compose (Recommended for Testing)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Production Deployment

### Building the Docker Image

**Method 1: Using build args**
```bash
docker build \
  --build-arg OPENAI_API_KEY=your_key_here \
  -t ghcr.io/your-username/nfcorpus-evaluator:latest \
  .
```

**Method 2: Using Docker secrets (recommended)**
```bash
echo "your_key_here" > openai_key.txt
docker build \
  --secret id=openai_key,src=openai_key.txt \
  -t ghcr.io/your-username/nfcorpus-evaluator:latest \
  .
rm openai_key.txt
```

### Running the Container

```bash
docker run -d \
  --name nfcorpus-evaluator \
  -p 9009:9009 \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  ghcr.io/your-username/nfcorpus-evaluator:latest
```

### Publishing to GitHub Container Registry

The repository includes a GitHub Actions workflow that automatically:
1. Builds the Docker image
2. Runs tests
3. Publishes to GHCR

**Setup:**
1. Add `OPENAI_API_KEY` to GitHub Secrets (Settings → Secrets → Actions)
2. Push to `main` branch or create a version tag

**Tags:**
- Push to `main` → `ghcr.io/username/repo:latest`
- Tag `v1.0.0` → `ghcr.io/username/repo:1.0.0` and `ghcr.io/username/repo:1`

## AgentBeats Platform Deployment

1. Build and publish your Docker image to GHCR
2. Register your agent on AgentBeats platform
3. Provide the image URL: `ghcr.io/your-username/nfcorpus-evaluator:latest`
4. Configure environment variables in AgentBeats:
   - `OPENAI_API_KEY`: Your OpenAI API key

## Health Checks

### A2A Server
```bash
curl http://localhost:9009/
```

### MCP Server
```bash
curl http://localhost:8000/health
```

### Test MCP Search
```bash
curl -X POST http://localhost:8000/search_nfcorpus \
  -H "Content-Type: application/json" \
  -d '{"query": "calcium and bone health", "top_k": 5}'
```

## Troubleshooting

### Data preparation fails during build
- Verify OpenAI API key is correct
- Check OpenAI API rate limits
- Ensure sufficient disk space

### MCP server not responding
- Check if port 8000 is exposed
- Verify Qdrant data exists in `/home/agent/qdrant_data`
- Check MCP server logs

### A2A server not responding
- Check if port 9009 is exposed
- Verify dataset was loaded correctly
- Check A2A server logs

### Purple agent can't reach MCP server
- Ensure both containers are on same Docker network
- Use container name instead of localhost
- Check firewall rules

## Performance Considerations

### Data Preparation Time
- ~3,633 documents to embed
- ~37 API calls (100 docs per batch)
- Estimated time: 2-5 minutes
- Cost: ~$0.01 with text-embedding-3-small

### Runtime Performance
- Query embedding: ~100ms per query
- Vector search: <50ms per query
- Total per query: ~150ms
- 50 queries: ~7.5 seconds (search only)

### Resource Requirements
- Memory: ~1GB (including Qdrant)
- Disk: ~500MB (dataset + embeddings)
- CPU: Minimal (mostly I/O bound)

## Monitoring

### Key Metrics to Monitor
- A2A request success rate
- MCP search latency
- OpenAI API errors
- Qdrant query performance

### Logs
```bash
# Docker Compose
docker-compose logs -f green-agent

# Docker
docker logs -f nfcorpus-evaluator
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use Docker secrets** for production builds
3. **Restrict MCP server access** to trusted networks only
4. **Rotate API keys** regularly
5. **Monitor API usage** to detect anomalies

## Backup and Recovery

### Backup Qdrant Data
```bash
docker cp nfcorpus-evaluator:/home/agent/qdrant_data ./backup/
```

### Restore Qdrant Data
```bash
docker cp ./backup/qdrant_data nfcorpus-evaluator:/home/agent/
```

### Rebuild from Scratch
If data is corrupted, simply rebuild the Docker image with a valid OpenAI API key.
