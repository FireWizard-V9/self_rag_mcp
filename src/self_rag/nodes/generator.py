from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.prompts.generation import GENERATION_PROMPT


def generator(state: GraphState) -> dict:
    model = get_chat_model()

    prompt = GENERATION_PROMPT.format(
        context=state.get("context", ""),
        question=state["question"],
    )

    answer = model.invoke(prompt).content

    return {"answer": answer}
