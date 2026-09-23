# Self-RAG MCP Server Setup Guide

**Version:** 1.0  
**Focus:** MCP Server Deployment  
**Status:** Production-Ready

---

## Quick Start (5 Minutes)

```bash
# 1. Clone and install
git clone <repo-url> && cd self-rag-retrieval
uv sync

# 2. Create minimal .env (see below)
cat > .env << 'EOF'
VECTORDB_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
RETRIEVAL_COLLECTION=documents
PARENT_EXPANSION_COLLECTION=null
EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY
EOF

# 3. Start vector DB
docker compose up -d

# 4. Start MCP server
uv run python src/self_rag/mcp/server.py

# Done! Server running on http://127.0.0.1:8000/sse
```

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Essential Configuration](#essential-configuration)
6. [Starting the Server](#starting-the-server)
7. [Verification](#verification)
8. [Common Setups](#common-setups)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Self-RAG MCP Server is a **production-grade retrieval engine** exposed as an MCP (Model Context Protocol) server.

### Core Purpose

Provide advanced document retrieval capabilities to any LLM or RAG system:

```
Your Application/LLM
        ↓
    MCP Protocol
        ↓
Self-RAG Retrieval Engine
    ├─ Hybrid Search (Dense + Sparse)
    ├─ MMR Diversity Filtering
    ├─ Cross-Encoder Reranking
    └─ Optional Parent Context
        ↓
Vector Database (Qdrant, Pinecone, etc.)
```

### What You Get

- ✅ **Advanced Retrieval** — 4-stage pipeline optimized for quality
- ✅ **Optional Grading** — LLM-based quality checks (optional)
- ✅ **Database Agnostic** — Works with any vector DB
- ✅ **Provider Agnostic** — Works with any embedding/LLM provider
- ✅ **Production Ready** — Enterprise error handling, configurable

---

## Architecture

### Data Flow

```
Client/LLM
    ↓
MCP Server (http://localhost:8000/sse)
    ├─ retrieve_documents (core retrieval)
    ├─ rag_answer (optional: with grading)
    └─ server_health (status check)
    ↓
Retrieval Pipeline
    ├─ Stage 1: Hybrid Search (0.5s)
    ├─ Stage 2: MMR Reranking (0.1s)
    ├─ Stage 3: Cross-Encoder (0.1s)
    └─ Stage 4: Parent Expansion (0.05s, optional)
    ↓
Vector Database
    ├─ search_dense() - Semantic search
    ├─ search_sparse() - Keyword search (if supported)
    └─ retrieve_by_ids() - Fetch full context
```

### Two Operating Modes

#### **Mode 1: Retrieval-Only (Core)**
```
retrieve_documents(query) 
  → Top documents ranked by relevance
  → Send to your LLM for generation
  
Perfect for: Speed, external control, custom RAG logic
```

#### **Mode 2: Full Self-RAG (Optional)**
```
rag_answer(question)
  → retrieve documents
  → verify relevance
  → generate answer
  → check if grounded
  → grade usefulness
  → retry if quality issues
  
Perfect for: Autonomous quality control, less manual intervention
```

---

## Prerequisites

### System

- **Python:** 3.12+
- **RAM:** 4GB minimum (8GB+ recommended)
- **Disk:** 10GB for models/data
- **OS:** Linux, macOS, Windows (WSL2)

### Required: Package Manager

```bash
# Install uv (one-liner)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Required: Vector Database (Choose One)

```bash
# Option 1: Qdrant (Recommended, Docker)
docker run -p 6333:6333 qdrant/qdrant:latest

# Option 2: Pinecone (Cloud, requires API key)
# Sign up: https://pinecone.io

# Option 3: Weaviate (Docker)
docker run -p 8080:8080 semitechnologies/weaviate:latest

# Option 4: Chroma (Embedded, no setup)
# Just use it - data stored in ./chroma_data
```

### Optional: Embedding Provider

For embeddings (required but can be any provider):

```bash
# Option A: OpenAI (via OpenRouter)
# Sign up: https://openrouter.ai
# Get API key: https://openrouter.ai/keys

# Option B: HuggingFace
# Sign up: https://huggingface.co
# Get token: https://huggingface.co/settings/tokens

# Option C: Cohere
# Sign up: https://cohere.com
# Get API key

# Option D: Local embeddings
# Use any sentence-transformers model locally
```

### Optional: LLM Provider (For Grading Only)

If you want optional Self-RAG grading:

```bash
# Option A: Any OpenAI-compatible API
# - OpenRouter, Azure OpenAI, Ollama, vLLM, etc.

# Option B: Custom LLM
# - Anthropic, Cohere, HuggingFace, etc.
```

---

## Installation

### 1. Clone & Install

```bash
git clone <repository-url> self-rag-retrieval
cd self-rag-retrieval

# Install dependencies
uv sync

# Verify
uv run python -c "from self_rag.core.config import get_settings; print('✓ Ready')"
```

### 2. Start Vector Database

```bash
# If using Qdrant
docker compose up -d

# If using Pinecone, Weaviate, or Chroma
# (Follow their respective setup instructions)
```

---

## Essential Configuration

### Minimal Setup (.env)

**All you absolutely need:**

```env
# ============================================================
# RETRIEVAL ENGINE (Required)
# ============================================================

# Vector Database
VECTORDB_PROVIDER=qdrant              # qdrant, pinecone, weaviate, or chroma
QDRANT_URL=http://localhost:6333      # Only if using Qdrant
RETRIEVAL_COLLECTION=documents        # Collection name in your vector DB
PARENT_EXPANSION_COLLECTION=null      # null = no parent context (faster)

# Embeddings (Required)
# Any provider with OpenAI-compatible API:
EMBEDDING_MODEL=openai/text-embedding-3-small

# Choose your embedding provider:
# Option 1: OpenRouter (easiest)
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY

# Option 2: OpenAI direct
# OPENAI_API_KEY=sk-YOUR_KEY
# EMBEDDING_MODEL=text-embedding-3-small

# Option 3: HuggingFace
# HUGGINGFACE_API_KEY=hf_YOUR_KEY
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ============================================================
# OPTIONAL: Self-RAG Grading (for rag_answer tool)
# ============================================================

# LLM for grading (leave blank to disable grading)
CHAT_MODEL=openai/gpt-4.1-mini  # Any OpenAI-compatible LLM

# Use same API key as embeddings if possible
# Or set different key for different LLM provider
LLM_TEMPERATURE=0.0             # DO NOT CHANGE (must be 0 for grading)
MAX_RETRIES=3                   # How many times to retry if quality issues
```

**That's it!** Paste above into `.env` and replace the API key.

---

## Provider Configuration

### Option 1: OpenRouter (Simplest - Get Everything from One Place)

```env
# Embeddings + LLM from same provider
EMBEDDING_MODEL=openai/text-embedding-3-small
CHAT_MODEL=openai/gpt-4.1-mini
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY

# Setup:
# 1. Sign up: https://openrouter.ai
# 2. Get key: https://openrouter.ai/keys
# 3. Add credits to account
# 4. Paste key above
```

### Option 2: OpenAI Direct

```env
# Use your own OpenAI keys
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4.1-mini
OPENAI_API_KEY=sk-YOUR_KEY

# Setup:
# 1. Sign up: https://openai.com
# 2. Get key: https://platform.openai.com/api-keys
# 3. Paste key above
```

### Option 3: HuggingFace

```env
# Free embeddings from HuggingFace
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_API_KEY=hf_YOUR_TOKEN

# Setup:
# 1. Sign up: https://huggingface.co
# 2. Get token: https://huggingface.co/settings/tokens
# 3. Paste token above
# 4. LLM grading: use OpenRouter or another provider
```

### Option 4: Ollama (Local, Free)

```env
# Local LLM and embeddings (no API keys needed!)
LLM_PROVIDER=openrouter  # Route to local Ollama
OPENROUTER_BASE_URL=http://localhost:11434/v1
CHAT_MODEL=llama2  # Whatever model you have running

# Embeddings: can also be local if available
EMBEDDING_MODEL=local  # or use OpenRouter for embeddings only
```

---

## Tuning Configuration (Optional)

### If You Want to Adjust Retrieval Quality

```env
# Retrieval pipeline tuning (optional)
VECTORDB_HYBRID_STRATEGY=rrf        # rrf, weighted, semantic, two_pass
RETRIEVAL_K_INITIAL=20              # Candidate pool
RETRIEVAL_K_MMR=15                  # After diversity filter
RETRIEVAL_K_RERANK=4                # Final results
RETRIEVAL_LAMBDA=0.5                # Relevance vs diversity (0-1)

# Document chunking (only if re-ingesting)
CHUNK_SIZE=600                      # Child chunk size
CHUNK_OVERLAP=150
PARENT_CHUNK_SIZE=1200              # Full context chunk
```

---

## Starting the Server

### Step 1: Verify Configuration

```bash
uv run python -c "
from self_rag.core.config import get_settings
settings = get_settings()
print(f'✓ Vector DB: {settings.vectordb_provider}')
print(f'✓ Collection: {settings.retrieval_collection}')
print(f'✓ Embedding: {settings.embedding_model}')
print(f'✓ Chat Model: {settings.chat_model}')
"
```

### Step 2: Start MCP Server

```bash
uv run python src/self_rag/mcp/server.py

# Expected output:
# INFO  Warming up retriever...
# INFO  Warming up models...
# INFO  Warmup complete — server ready.
# INFO  Uvicorn running on http://127.0.0.1:8000
```

Server is now running and accepting MCP connections.

---

## Verification

### Test 1: Health Check

```bash
curl -X POST http://127.0.0.1:8000/sse \
  -H "Content-Type: application/json" \
  -d '{"tool": "server_health", "params": {}}'

# Should return: {"status": "healthy", ...}
```

### Test 2: Retrieval (Core Feature)

```bash
curl -X POST http://127.0.0.1:8000/sse \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "retrieve_documents",
    "params": {
      "query": "your question here",
      "top_k": 3,
      "expand_parents": false
    }
  }'

# Should return: {"status": "success", "documents": [...]}
```

### Test 3: Full Self-RAG (if LLM configured)

```bash
curl -X POST http://127.0.0.1:8000/sse \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "rag_answer",
    "params": {
      "question": "your question",
      "max_retries": 2
    }
  }'

# Should return: {"status": "success", "result": {"answer": "..."}}
```

---

## Common Setups

### Setup 1: Minimal Retrieval-Only

```env
# Just retrieval, no LLM
VECTORDB_PROVIDER=chroma
RETRIEVAL_COLLECTION=documents
PARENT_EXPANSION_COLLECTION=null

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_API_KEY=hf_YOUR_TOKEN

# No CHAT_MODEL - grading disabled
```

**Cost:** Free (HuggingFace + Chroma)  
**Speed:** Fast  
**Best for:** Lightweight retrieval only

### Setup 2: Cloud Vector DB + Local Embeddings

```env
# Pinecone for vectors
VECTORDB_PROVIDER=pinecone
PINECONE_API_KEY=pk-YOUR_KEY
PINECONE_INDEX_NAME=my-index
RETRIEVAL_COLLECTION=my-index

# Free embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_API_KEY=hf_YOUR_TOKEN
```

**Cost:** Pinecone's serverless pricing  
**Speed:** Fast  
**Best for:** Production with zero ops

### Setup 3: Full Production with Grading

```env
# Qdrant for vectors
VECTORDB_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
RETRIEVAL_COLLECTION=documents
PARENT_EXPANSION_COLLECTION=documents_full

# Both from OpenRouter
EMBEDDING_MODEL=openai/text-embedding-3-small
CHAT_MODEL=openai/gpt-4.1-mini
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY

# Tuning
RETRIEVAL_K_RERANK=6
MAX_RETRIES=3
```

**Cost:** Pay-per-use  
**Speed:** Medium  
**Best for:** Full Self-RAG pipeline with quality control

---

## Troubleshooting

### "Connection refused to vector DB"

```bash
# Verify DB is running
curl http://localhost:6333/health  # For Qdrant

# Or restart
docker compose down && docker compose up -d
```

### "Invalid API key"

```bash
# Verify key is set
echo $OPENROUTER_API_KEY

# Check key format (should start with sk-or-v1-)
# Visit https://openrouter.ai to verify account has credits
```

### "Collection not found"

```bash
# You need to add documents to your vector DB first
# Option 1: Ingest from data directory
uv run python scripts/ingest.py --reset

# Option 2: Use external documents
# Upload to your Pinecone/Weaviate/Qdrant instance
```

### "No embeddings provider configured"

```bash
# At minimum, you need EMBEDDING_MODEL and API key
# Examples:
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_API_KEY=hf_YOUR_TOKEN

# Or
EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY
```

### "High latency (>2 seconds)"

```env
# Reduce retrieval stages
RETRIEVAL_K_INITIAL=10
RETRIEVAL_K_RERANK=2

# Or use faster search
VECTORDB_HYBRID_STRATEGY=semantic
```

---

## MCP Integration Examples

### Using with Claude Desktop

```json
{
  "mcp": {
    "self-rag": {
      "command": "uv",
      "args": ["run", "python", "/path/to/src/self_rag/mcp/server.py"]
    }
  }
}
```

### Using with Custom Application

```python
import requests

class SelfRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Core retrieval function"""
        response = requests.post(
            f"{self.base_url}/sse",
            json={
                "tool": "retrieve_documents",
                "params": {"query": query, "top_k": top_k}
            }
        )
        return response.json()["documents"]
    
    def answer(self, question: str) -> str:
        """Optional: Full Self-RAG with grading"""
        response = requests.post(
            f"{self.base_url}/sse",
            json={
                "tool": "rag_answer",
                "params": {"question": question}
            }
        )
        return response.json()["result"]["answer"]

# Usage
client = SelfRAGClient()
docs = client.retrieve("What is your policy?")
```

---

## Next Steps

1. ✅ **Install** — `uv sync`
2. ✅ **Configure** — Create `.env` with minimal settings
3. ✅ **Setup Vector DB** — Start Qdrant/Pinecone/etc
4. ✅ **Start Server** — `uv run python src/self_rag/mcp/server.py`
5. ✅ **Test** — Call `retrieve_documents` endpoint
6. ✅ **Integrate** — Use in your application or Claude Desktop

---

## Key Takeaways

| What | What You Need |
|------|---------------|
| **Retrieval Engine** | Vector DB + Embedding Model (Required) |
| **Optional Grading** | LLM Provider (Optional, for full Self-RAG) |
| **Vector DB** | Any: Qdrant, Pinecone, Weaviate, Chroma |
| **Embeddings** | Any: OpenAI, HuggingFace, Cohere, Ollama |
| **LLM** | Any: OpenRouter, OpenAI, Anthropic, Ollama |
| **Configuration** | Just 4-5 lines in `.env` |
| **Setup Time** | ~5 minutes |

---

**Ready to go!** Start with the minimal setup above, then customize as needed. 🚀
