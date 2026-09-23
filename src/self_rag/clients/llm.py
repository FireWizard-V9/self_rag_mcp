from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLM
from langchain_openai import OpenAIEmbeddings

from self_rag.core.config import get_settings


@lru_cache
def get_chat_model() -> BaseChatModel:
    """
    Return the shared chat model used throughout the application.
    Caches the model singleton since instantiation is expensive.
    Provider is auto-selected: vLLM (default) or OpenRouter (fallback).
    """
    settings = get_settings()
    return ChatLiteLLM(
        model=settings.chat_model,
        api_base=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
    )


@lru_cache
def get_embedding_model() -> Embeddings:
    """
    Return the shared embedding model used throughout the application.
    Embeddings always use OpenAI API via OpenRouter (independent of LLM provider).
    """
    settings = get_settings()

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embeddings_api_key,
        base_url=settings.embeddings_base_url,
    )
