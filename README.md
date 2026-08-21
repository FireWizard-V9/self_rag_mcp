# Self-RAG Retrieval Engine

 **Self-Reflective Retrieval-Augmented Generation** system built with LangGraph, Qdrant, and exposed as an MCP (Model Context Protocol) server over SSE transport.

Unlike standard RAG pipelines that blindly retrieve and generate, Self-RAG makes the LLM an active participant in its own quality control — deciding whether to retrieve, grading what it retrieved, verifying what it generated, and retrying when the answer isn't good enough.

---

## Table of Contents

- [What is Self-RAG?](#what-is-self-rag)
- [Architecture Overview](#architecture-overview)
- [Self-RAG Graph Flow](#self-rag-graph-flow)
  - [Node Reference](#node-reference)
  - [Routing Logic](#routing-logic)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Ingestion Pipeline](#ingestion-pipeline)
- [MCP Server & Client](#mcp-server--client)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Running Tests](#running-tests)
- [Tech Stack](#tech-stack)

---

## What is Self-RAG?

Standard RAG has a fundamental problem: it always retrieves (even when unnecessary), never checks if retrieved documents are relevant, and never verifies if the generated answer is actually grounded in those documents.

**Self-RAG** (introduced in the paper [Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection](https://arxiv.org/abs/2310.11511)) solves this by inserting reflection steps at every stage:

| Stage | Standard RAG | Self-RAG |
|---|---|---|
| Retrieval decision | Always retrieves | LLM decides if retrieval is needed |
| Document filtering | Uses all retrieved docs | LLM grades each doc for relevance |
| Generation | Generate once | Generate, then verify grounding |
| Answer quality | No check | LLM grades usefulness, retries if needed |

This implementation uses **LangGraph** to model the Self-RAG flow as a stateful directed graph with conditional edges, enabling dynamic routing, retry loops, and full state traceability.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Client (SSE)                         │
│                    rich interactive terminal                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ SSE  http://127.0.0.1:8000/sse
┌──────────────────────────▼──────────────────────────────────────┐
│                      MCP Server (SSE)                           │
│              MCPServer  ·  3 tools exposed                      │
│         rag_answer  ·  retrieve  ·  server_health               │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
┌──────────▼──────────┐       ┌───────────▼──────────────────────┐
│   Self-RAG Graph    │       │       Hybrid Retriever           │
│   (LangGraph)       │       │                                  │
│                     │       │  1. Qdrant Hybrid Search         │
│  retrieval_decision │       │     Dense (OpenAI embeddings)    │
│  retrieve           │       │     Sparse (BM25 / FastEmbed)    │
│  relevance_grader   │       │     Fusion: RRF                  │
│  context_builder    │       │                                  │
│  generator          │       │  2. MMR Diversity Reranking      │
│  support_grader     │       │                                  │
│  usefulness_grader  │       │  3. FlashRank Cross-Encoder      │
│                     │       │     (ms-marco-MiniLM-L-12-v2)    │
└──────────┬──────────┘       │                                  │
           │                  │  4. Parent Document Expansion    │
           │                  └───────────────┬──────────────────┘
           │                                  │
┌──────────▼──────────────────▼──────────────────────────────────┐
│                         Qdrant                                  │
│                                                                 │
│   self_rag_documents  (child chunks  · dense + sparse)         │
│   self_rag_parents    (parent chunks · dense only)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Self-RAG Graph Flow

```mermaid
flowchart TD
    START([START]) --> RD[retrieval_decision]

    RD -->|should_retrieve = true| RET[retrieve]
    RD -->|should_retrieve = false| GEN[generator]

    RET --> REL[relevance_grader]
    REL --> CTX[context_builder]
    CTX --> GEN

    GEN --> SUP[support_grader]

    SUP -->|fully_supported\npartially_supported| USE[usefulness_grader]
    SUP -->|not_supported\n& retry_count < max_retries| INC1[increment_retry]
    SUP -->|not_supported\n& retry_count >= max_retries| USE

    INC1 --> GEN

    USE -->|useful| END([END])
    USE -->|not_useful\n& retry_count >= max_retries| END
    USE -->|not_useful\n& retry_count < max_retries| INC2[increment_retry_for_retrieval]

    INC2 --> RET

    style START fill:#2d6a4f,color:#fff
    style END fill:#2d6a4f,color:#fff
    style RD fill:#1d3557,color:#fff
    style RET fill:#457b9d,color:#fff
    style REL fill:#457b9d,color:#fff
    style CTX fill:#457b9d,color:#fff
    style GEN fill:#e63946,color:#fff
    style SUP fill:#f4a261,color:#000
    style USE fill:#f4a261,color:#000
    style INC1 fill:#6d6875,color:#fff
    style INC2 fill:#6d6875,color:#fff
```

### Node Reference

#### `retrieval_decision`
The entry point of the graph. The LLM analyzes the user's question and decides whether external knowledge retrieval is actually needed.

- Conversational queries (`"Hello"`, `"What is 2+2"`) → skip retrieval, go directly to `generator`
- Factual / domain queries → proceed to `retrieve`

Uses structured output: `RetrievalDecision { thought: str, answer: "YES" | "NO" }`

---

#### `retrieve`
Runs the full **Hybrid Retrieval Pipeline** against Qdrant:

1. **Hybrid Search** — combines dense (OpenAI `text-embedding-3-small`) and sparse (BM25 via FastEmbed) vectors, fused server-side with Reciprocal Rank Fusion (RRF)
2. **MMR** — Maximal Marginal Relevance reranking for diversity (avoids returning near-duplicate chunks)
3. **FlashRank** — lightweight ONNX cross-encoder reranker (`ms-marco-MiniLM-L-12-v2`) for final relevance scoring
4. **Parent Expansion** — child chunks are retrieved for precision, but the full parent chunk is returned to the LLM for richer context

---

#### `relevance_grader`
Filters retrieved documents. Each document is individually graded by the LLM against the question.

- Documents graded `YES` → kept as `relevant_documents`
- Documents graded `NO` → discarded

Uses structured output: `RelevanceGrade { thought: str, answer: "YES" | "NO" }`

---

#### `context_builder`
Formats the relevant documents into a structured XML context block optimized for LLM attention:

```xml
<context>
  <document index="1">
    <metadata>Source: hr.pdf | Relevance Score: 0.9821</metadata>
    <content>
      Human Resource Management (HRM) refers to...
    </content>
  </document>
</context>
```

---

#### `generator`
The LLM generates an answer using **only** the facts in the context block. The prompt explicitly instructs the model not to use outside knowledge and to cite document indices (`[Doc 1]`).

---

#### `support_grader`
Verifies that the generated answer is grounded in the context. Performs a claim-by-claim audit.

Returns one of:
- `fully_supported` — every claim is backed by the context
- `partially_supported` — some claims are grounded, others are not
- `not_supported` — answer contains hallucinations or contradicts the context

Uses structured output: `SupportGrade { thought: str, label: "fully_supported" | "partially_supported" | "not_supported" }`

---

#### `usefulness_grader`
Evaluates whether the answer actually resolves the user's question — even if it's grounded, it might be evasive or incomplete.

Returns one of:
- `useful` — answer directly satisfies the query
- `not_useful` — answer is off-topic, incomplete, or evasive

Uses structured output: `UsefulnessGrade { thought: str, label: "useful" | "not_useful" }`

---

#### `increment_retry` / `increment_retry_for_retrieval`
Bookkeeping nodes that increment `retry_count` in the graph state before looping back to `generator` or `retrieve` respectively.

---

### Routing Logic

| Router | Condition | Next Node |
|---|---|---|
| `route_after_retrieval_decision` | `should_retrieve = True` | `retrieve` |
| | `should_retrieve = False` | `generator` |
| `route_after_support` | `fully_supported` or `partially_supported` | `usefulness_grader` |
| | `not_supported` and `retry_count < max_retries` | `increment_retry` → `generator` |
| | `not_supported` and `retry_count >= max_retries` | `usefulness_grader` |
| `route_after_usefulness` | `useful` | `END` |
| | `not_useful` and `retry_count < max_retries` | `increment_retry_for_retrieval` → `retrieve` |
| | `not_useful` and `retry_count >= max_retries` | `END` |

---

## Retrieval Pipeline

```
Query
  │
  ▼
Qdrant Hybrid Search (Dense + BM25 + RRF)   k=20 candidates
  │
  ▼
MMR Diversity Reranking                      k=15 diverse docs
  │
  ▼
FlashRank Cross-Encoder                      top_k=4 final docs
  │
  ▼
Parent Document Expansion                    fetch full parent chunks
  │
  ▼
List[Document] → relevance_grader
```

**Why this multi-stage funnel?**

- Hybrid search (dense + sparse) gives better recall than either alone — dense catches semantic matches, BM25 catches exact keyword matches
- MMR prevents the LLM from seeing 4 near-identical chunks — forces diversity
- FlashRank (ONNX int8 quantized) gives cross-encoder quality at ~0.1s vs ~19s for a full PyTorch CrossEncoder
- Parent expansion means retrieval precision comes from small child chunks, but the LLM gets the full surrounding context

---

## Ingestion Pipeline

Documents are split into a **parent-child chunk hierarchy**:

```
PDF Document
  │
  ├── Parent Chunk 1  (1200 chars, overlap=0)  → stored in self_rag_parents
  │     ├── Child Chunk 1a  (600 chars, overlap=150)  → stored in self_rag_documents
  │     ├── Child Chunk 1b
  │     └── Child Chunk 1c
  │
  ├── Parent Chunk 2
  │     ├── Child Chunk 2a
  │     └── Child Chunk 2b
  ...
```

- **Child chunks** are indexed with both dense + sparse vectors for hybrid search precision
- **Parent chunks** are stored with dense vectors only, used for context expansion after retrieval
- UUIDs are deterministic (UUID5) so re-ingestion is idempotent

---

## MCP Server & Client

The system is exposed as an **MCP server** over SSE transport, making it compatible with any MCP client (Claude Desktop, custom clients, etc.).

### Tools

| Tool | Description |
|---|---|
| `rag_answer` | Runs the full Self-RAG graph — retrieval decision → retrieve → grade → generate → verify → retry |
| `retrieve` | Raw hybrid retrieval only, no generation or grading |
| `server_health` | Returns operational status of retriever and reranker components |

### Interactive Client

A rich terminal client is included with a menu-driven interface:

```
╭─────────────────────────────────╮
│ Self-RAG MCP Interactive Client │
│ Connected via SSE Transport     │
╰─────────────────────────────────╯

[1] 💬 Ask Question     (rag_answer)
[2] 🔍 Raw Search       (retrieve)
[3] 🏥 System Health    (server_health)
[4] 📋 List Tools
[0] 🚪 Exit
```

---

## Project Structure

```
self_rag_retrieval/
├── src/self_rag/
│   ├── clients/
│   │   ├── llm.py              # LiteLLM chat model + OpenAI embeddings (cached)
│   │   └── qdrant.py           # Qdrant client singleton
│   ├── core/
│   │   └── config.py           # Pydantic settings from .env
│   ├── graph/
│   │   ├── engine.py           # Compiled graph singleton (lru_cache)
│   │   ├── routes.py           # Conditional edge routing functions
│   │   └── workflow.py         # LangGraph StateGraph definition
│   ├── ingestion/
│   │   ├── chunker.py          # Parent-child chunk splitting
│   │   ├── indexer.py          # Qdrant collection management
│   │   ├── loaders.py          # PDF loader
│   │   └── pipeline.py         # Ingestion orchestration
│   ├── mcp/
│   │   ├── server.py           # MCPServer with 3 tools + startup warmup
│   │   ├── mcp_client.py       # Rich interactive terminal client
│   │   └── tools.py            # Tool implementations (answer, retrieve, health)
│   ├── models/
│   │   ├── graph_state.py      # LangGraph TypedDict state
│   │   └── schemas.py          # Pydantic structured output schemas
│   ├── nodes/
│   │   ├── context_builder.py  # XML context formatter
│   │   ├── generator.py        # LLM answer generation
│   │   ├── relevance_grader.py # Per-document relevance grading
│   │   ├── retrieval_decision.py # Retrieval necessity classifier
│   │   ├── retrieve.py         # Retrieval node
│   │   ├── support_grader.py   # Hallucination / grounding checker
│   │   └── usefulness_grader.py # Answer quality checker
│   ├── prompts/
│   │   ├── generation.py
│   │   ├── relevance.py
│   │   ├── retrieval.py
│   │   ├── support.py
│   │   └── usefulness.py
│   ├── retrieval/
│   │   ├── mmr.py              # Maximal Marginal Relevance
│   │   ├── reranker.py         # FlashRank ONNX cross-encoder
│   │   ├── retriever.py        # HybridRetriever orchestrator (cached)
│   │   └── vector_store.py     # Qdrant vector store (dense + sparse, cached)
│   └── services/
│       └── rag_service.py      # Business layer wrapping the graph
├── scripts/
│   └── ingest.py               # CLI ingestion script
├── tests/
│   ├── test_mcp_server.py
│   ├── test_mcp_tools.py
│   └── test_routes.py
├── docker-compose.yaml
├── pyproject.toml
└── .env
```

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for Qdrant)
- OpenRouter API key

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd self_rag_retrieval
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-...

CHAT_MODEL=openrouter/openai/gpt-4.1-mini
EMBEDDING_MODEL=openai/text-embedding-3-small

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=self_rag_documents
QDRANT_PARENT_COLLECTION=self_rag_parents

DATA_DIR=src/self_rag/data
```

### 3. Start Qdrant

```bash
docker compose up -d
```

### 4. Add your documents

Place PDF files in `src/self_rag/data/`.

### 5. Ingest documents

```bash
# First time
uv run python scripts/ingest.py

# Full rebuild (wipes existing collections)
uv run python scripts/ingest.py --reset
```

---

## Configuration

All settings are in `.env` and validated by Pydantic. Key parameters:

| Variable | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `openrouter/openai/gpt-4.1-mini` | LLM for all grading and generation nodes |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Dense embedding model |
| `CHUNK_SIZE` | `600` | Child chunk size (chars) |
| `CHUNK_OVERLAP` | `150` | Child chunk overlap |
| `PARENT_CHUNK_SIZE` | `1200` | Parent chunk size (chars) |
| `RETRIEVAL_K_INITIAL` | `20` | Hybrid search candidate pool |
| `RETRIEVAL_K_MMR` | `15` | Docs after MMR diversity filter |
| `RETRIEVAL_K_RERANK` | `4` | Final docs after FlashRank |
| `MAX_RETRIES` | `3` | Max Self-RAG retry loops |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature (0 = deterministic) |

---

## Running the System

### Terminal 1 — Start the MCP server

```bash
uv run python src/self_rag/mcp/server.py
```

The server warms up all models before accepting connections:

```
INFO  Warming up retriever...
INFO  Warming up reranker...
INFO  Warming up graph...
INFO  Warmup complete — server ready.
INFO  Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 — Start the interactive client

```bash
uv run python src/self_rag/mcp/mcp_client.py
```

### Sample questions (HR domain)

```
What is Human Resource Management and what are its main objectives?
What are the nine broad areas of HRM activities identified by ASTD?
What is the difference between training and organizational development?
How does compensation and benefits management work in HRM?
What is the role of HRM in the new millennium?
What is the significance of HR planning in an organization?
Explain the scope of HRM and what it covers in an employee's working life.
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

```
tests/test_routes.py::test_retrieval_decision_retrieve          PASSED
tests/test_routes.py::test_retrieval_decision_skip              PASSED
tests/test_routes.py::test_support_fully_supported_...          PASSED
tests/test_routes.py::test_support_not_supported_retries...     PASSED
tests/test_routes.py::test_usefulness_useful_ends               PASSED
...
24 passed
```

Test coverage:
- **`test_routes.py`** — all routing branches (retrieval decision, support grading, usefulness grading)
- **`test_mcp_tools.py`** — tool functions with mocked Qdrant/LLM (empty input, clamping, exceptions, health)
- **`test_mcp_server.py`** — server type, tool registration, tool descriptions

---

## Tech Stack

| Component | Technology |
|---|---|
| Graph orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM routing | [LiteLLM](https://github.com/BerriAI/litellm) via OpenRouter |
| LLM | OpenAI GPT-4.1-mini (via OpenRouter) |
| Embeddings | OpenAI `text-embedding-3-small` (via OpenRouter) |
| Vector database | [Qdrant](https://qdrant.tech/) |
| Sparse embeddings | [FastEmbed](https://github.com/qdrant/fastembed) BM25 |
| Reranker | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) `ms-marco-MiniLM-L-12-v2` (ONNX int8) |
| MCP framework | [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) v2 |
| Settings | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| Terminal UI | [Rich](https://github.com/Textualize/rich) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Runtime | Python 3.12 |
