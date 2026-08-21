from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openrouter_api_key: str

    # LLM & Embedding Models
    chat_model: str = "openai/gpt-4.1-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    llm_temperature: float = 0.0

    # Vector Database & Data Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "self_rag_documents"
    qdrant_parent_collection: str = "self_rag_parents"
    data_dir: str = "data"

    # Document Chunking
    chunk_size: int = Field(default=600, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    parent_chunk_size: int = Field(default=1200, gt=0)
    # Parents are the final context returned to the LLM, so keep them disjoint.
    # Child overlap preserves retrieval recall without repeating parent context.
    parent_chunk_overlap: int = Field(default=0, ge=0)

    # Progressive Multi-Stage Retrieval Funnel
    retrieval_k_initial: int = Field(
        default=20, gt=0  # Step 1: Hybrid candidate pool (Qdrant BM25 + Vector)
    )
    retrieval_k_rerank: int = Field(default=4, gt=0)
    retrieval_k_mmr: int = Field(default=15, gt=0)

    # MMR Diversity Tuning
    retrieval_lambda: float = Field(default=0.5, ge=0.0, le=1.0)

    # HuggingFace Reranker Model
    rerank_model: str = "BAAI/bge-reranker-base"

    # Self-RAG Flow Controls
    max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    """Returns a cached singleton instance of system configuration settings."""
    return Settings()
