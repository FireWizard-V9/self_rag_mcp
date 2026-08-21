from self_rag.models.graph_state import GraphState
from self_rag.retrieval.retriever import get_retriever


def retrieve_node(state: GraphState) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])

    return {
        "documents": docs,
        "retrieved_documents": docs,
    }
