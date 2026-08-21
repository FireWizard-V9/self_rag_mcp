import numpy as np
from langchain_core.documents import Document

from self_rag.clients.llm import get_embedding_model


def cosine_similarity(left: list[list[float]], right: list[list[float]]) -> np.ndarray:
    """Return pairwise cosine similarity without relying on deprecated LangChain helpers."""
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left_array, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right_array, axis=1, keepdims=True)
    return (left_array @ right_array.T) / np.maximum(left_norm * right_norm.T, 1e-12)


def apply_mmr(
    query: str,
    docs: list[Document],
    k: int,
    lambda_param: float = 0.5,
) -> list[Document]:

    if not docs:
        return []

    emb = get_embedding_model()

    doc_embeddings = emb.embed_documents([d.page_content for d in docs])
    query_embedding = emb.embed_query(query)

    similarities = cosine_similarity(
        [query_embedding],
        doc_embeddings,
    )[0]

    selected = []
    selected_idx = []

    while len(selected) < min(k, len(docs)):
        if not selected:
            idx = int(np.argmax(similarities))
        else:
            scores = []

            for i in range(len(docs)):
                if i in selected_idx:
                    continue

                relevance = similarities[i]

                diversity = max(
                    cosine_similarity(
                        [doc_embeddings[i]],
                        [doc_embeddings[j]],
                    )[0][0]
                    for j in selected_idx
                )

                score = lambda_param * relevance - (1 - lambda_param) * diversity

                scores.append((i, score))

            idx = max(scores, key=lambda x: x[1])[0]

        selected.append(docs[idx])
        selected_idx.append(idx)

    return selected
