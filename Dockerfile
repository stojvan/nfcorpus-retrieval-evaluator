FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
WORKDIR /home/agent

COPY --chown=agent:agent pyproject.toml uv.lock README.md ./
COPY --chown=agent:agent src src

USER agent

ENV HF_HOME=/home/agent/.cache/huggingface
ENV QDRANT_PATH=/home/agent/qdrant_data

RUN mkdir -p /home/agent/.cache/huggingface /home/agent/qdrant_data /home/agent/data_cache

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    --mount=type=secret,id=openai_key,target=/run/secrets/openai_key \
    if [ -f /run/secrets/openai_key ]; then \
        export OPENAI_API_KEY=$(cat /run/secrets/openai_key); \
    fi && \
    if [ -n "$OPENAI_API_KEY" ]; then \
        uv run src/prepare_data.py; \
    else \
        echo "WARNING: OPENAI_API_KEY not provided, skipping data preparation"; \
    fi

COPY --chown=agent:agent src/start.sh ./
RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]
EXPOSE 9009 8000