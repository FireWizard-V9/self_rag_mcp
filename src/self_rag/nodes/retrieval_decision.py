from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.models.schemas import RetrievalDecision
from self_rag.prompts.retrieval import RETRIEVAL_SYSTEM


def retrieval_decision(state: GraphState):
    structured_model = get_chat_model().with_structured_output(RetrievalDecision)

    res: RetrievalDecision = structured_model.invoke([
        SystemMessage(content=RETRIEVAL_SYSTEM),
        HumanMessage(content=state["question"]),
    ])

    need_retrieve = res.answer == "YES"
    return {
        "need_retrieval": need_retrieve,
        "should_retrieve": need_retrieve,
    }
