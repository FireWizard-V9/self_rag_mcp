FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps, use CPU torch index)
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Pre-download FlashRank ONNX model at build time so container starts fast
RUN uv run python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/tmp/flashrank')"

EXPOSE 8765

CMD ["uv", "run", "python", "src/self_rag/mcp/server.py"]
