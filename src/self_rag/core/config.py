from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Backend Selection
    use_vllm: bool = True  # Use vLLM if True, fall back to OpenRouter if False
    llm_provider: str = "vllm"  # "vllm" or "openrouter"

    # vLLM Settings (local GPU inference)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "qwen2.5-7b-instruct-awq"
    vllm_api_key: str = ""  # Local vLLM doesn't require authentication

    # OpenRouter Settings (cloud fallback + embeddings API)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4.1-mini"

    # Unified LLM API (auto-selected based on use_vllm)
    llm_api_key: str = ""
    llm_base_url: str = ""
    chat_model: str = ""

    # Embeddings API (always uses OpenAI via OpenRouter)
    embeddings_api_key: str = ""
    embeddings_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    llm_temperature: float = 0.0

    # Vector Database Provider Selection
    vectordb_provider: str = Field(
        default="qdrant",
        description="Vector DB provider: qdrant, pinecone, weaviate, chroma"
    )
    vectordb_hybrid_strategy: str = Field(
        default="rrf",
        description="Hybrid search strategy: rrf, weighted, semantic, two_pass"
    )

    # Vector Database & Data Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Retrieval Collections (configurable for plug-and-play use)
    retrieval_collection: str = "self_rag_documents"
    parent_expansion_collection: str | None = "self_rag_parents"  # None to disable

    # Legacy aliases (for backward compatibility)
    qdrant_collection: str = ""  # Auto-populated from retrieval_collection
    qdrant_parent_collection: str = ""  # Auto-populated from parent_expansion_collection

    # Pinecone settings (optional)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "self-rag"
    pinecone_namespace: str = ""

    # Weaviate settings (optional)
    weaviate_url: str = "http://localhost:8080"
    weaviate_class_name: str = "SelfRAGDocument"

    # Chroma settings (optional)
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "self_rag"

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

    @model_validator(mode="after")
    def resolve_llm_provider(self) -> "Settings":
        """Auto-configure LLM endpoint based on provider selection."""
        if self.use_vllm:
            self.llm_provider = "vllm"
            self.llm_base_url = self.vllm_base_url
            self.chat_model = self.vllm_model
            self.llm_api_key = self.vllm_api_key
        else:
            self.llm_provider = "openrouter"
            self.llm_base_url = self.openrouter_base_url
            self.chat_model = self.openrouter_model
            self.llm_api_key = self.openrouter_api_key

        # Embeddings always use OpenRouter (independent of LLM provider)
        self.embeddings_api_key = self.openrouter_api_key
        self.embeddings_base_url = self.openrouter_base_url

        # Backward compatibility: populate legacy aliases from new config names
        if not self.qdrant_collection:
            self.qdrant_collection = self.retrieval_collection
        if not self.qdrant_parent_collection and self.parent_expansion_collection:
            self.qdrant_parent_collection = self.parent_expansion_collection

        return self


@lru_cache
def get_settings() -> Settings:
    """Returns a cached singleton instance of system configuration settings."""
    return Settings()
