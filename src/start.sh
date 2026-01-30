#!/bin/bash

echo "Starting NFCorpus Retrieval Evaluator Green Agent..."

echo "Starting MCP server on port 8000..."
uv run src/mcp_server.py &
MCP_PID=$!

sleep 2

echo "Starting A2A server on port 9009..."
CARD_URL="${CARD_URL:-http://green-agent:9009/}"
uv run src/server.py --host 0.0.0.0 --port 9009 --card-url "$CARD_URL" &
A2A_PID=$!

wait -n

exit $?
