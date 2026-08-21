from functools import lru_cache

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_CACHE = "/tmp/flashrank"


@lru_cache(maxsize=1)
def get_reranker() -> Ranker:
    return Ranker(model_name=FLASHRANK_MODEL, cache_dir=FLASHRANK_CACHE)


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int,
    **_kwargs,  # absorb unused model_name etc from old call sites
) -> list[Document]:
    if not docs:
        return []

    ranker = get_reranker()
    passages = [
        {"id": i, "text": doc.page_content[:1024]}
        for i, doc in enumerate(docs)
    ]
    results = ranker.rerank(RerankRequest(query=query, passages=passages))

    scored = sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]
    for r in scored:
        docs[r["id"]].metadata["rerank_score"] = float(r["score"])

    return [docs[r["id"]] for r in scored]
