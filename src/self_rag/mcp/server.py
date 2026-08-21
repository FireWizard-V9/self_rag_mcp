"""Self-RAG MCP server."""

import logging

from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.server import MCPServer
from pydantic import Field

from self_rag.mcp.tools import (
    answer_question,
    health,
    retrieve_documents,
)

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "Self-RAG Engine",
    description="A self-reflective Retrieval-Augmented Generation server for answering queries with verified factual grounding.",
)


def _warmup() -> None:
    """Load all models into memory before uvicorn starts accepting connections."""
    from self_rag.core.config import get_settings
    from self_rag.graph.engine import get_graph
    from self_rag.retrieval.reranker import get_reranker
    from self_rag.retrieval.retriever import get_retriever

    logger.info("Warming up retriever...")
    get_retriever()
    logger.info("Warming up reranker...")
    get_reranker()
    logger.info("Warming up graph...")
    get_graph()
    logger.info("Warmup complete — server ready.")


@mcp.tool()
async def rag_answer(
    ctx: Context,
    question: str = Field(
        ..., description="The user query or question to be answered by Self-RAG."
    ),
    max_retries: int = Field(
        2,
        description="Maximum internal state graph retries if hallucination is detected (1-5).",
    ),
) -> dict:
    """Run the complete Self-RAG state graph workflow (Decision -> Retrieval -> Verification -> Reflection)."""

    await ctx.info(f"Initiating Self-RAG workflow for question: '{question[:50]}...'")

    result = answer_question(question=question, max_retries=max_retries)

    if result.get("status") == "error":
        await ctx.error(f"Self-RAG failed: {result.get('message')}")
    else:
        await ctx.info("Self-RAG successfully generated and verified answer.")

    return result


@mcp.tool()
async def retrieve(
    ctx: Context,
    query: str = Field(..., description="Search query to retrieve documents for."),
    top_k: int = Field(
        10, description="Number of top document chunks to retrieve (1-50)."
    ),
) -> dict:
    """Retrieve raw relevant document chunks with metadata without running generation or self-correction."""
    await ctx.info(f"Retrieving top {top_k} documents for query: '{query}'")

    result = retrieve_documents(query=query, top_k=top_k)

    if result.get("status") == "error":
        await ctx.error(f"Retrieval error: {result.get('message')}")
    else:
        await ctx.info(f"Retrieved {result.get('count', 0)} chunks successfully.")

    return result


@mcp.tool()
def server_health() -> dict:
    """Check health and component status of the Self-RAG pipeline."""
    return health()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _warmup()
    mcp.run(transport="sse", host="127.0.0.1", port=8000)
