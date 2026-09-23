from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.prompts.generation import GENERATION_SYSTEM


def generator(state: GraphState) -> dict:
    model = get_chat_model()

    context = state.get("context", "")
    answer = model.invoke([
        SystemMessage(content=GENERATION_SYSTEM),
        HumanMessage(content=f"{context}\n\nQuestion: {state['question']}\n\nAnswer:"),
    ]).content

    return {"answer": answer}
