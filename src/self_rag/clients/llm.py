from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLM
from langchain_openai import OpenAIEmbeddings

from self_rag.core.config import get_settings


def get_chat_model() -> BaseChatModel:
    """
    Return the shared chat model used throughout the application.

    LiteLLM routes requests to OpenRouter.
    """

    settings = get_settings()
    return ChatLiteLLM(
        model=settings.chat_model,
        api_key=settings.openrouter_api_key,
        temperature=settings.llm_temperature,
    )


@lru_cache
def get_embedding_model() -> Embeddings:
    """
    Return the shared embedding model used throughout the application.
    """

    settings = get_settings()

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
