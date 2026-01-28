FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
WORKDIR /home/agent

COPY --chown=agent:agent pyproject.toml uv.lock README.md ./
COPY --chown=agent:agent src src
COPY --chown=agent:agent scripts scripts

USER agent

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run"]
CMD ["src/server.py", "--host", "0.0.0.0"]
EXPOSE 9009