from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.models.schemas import SupportGrade
from self_rag.prompts.support import SUPPORT_SYSTEM


def support_grader(state: GraphState) -> dict:
    structured_model = get_chat_model().with_structured_output(SupportGrade)

    context = state.get("context", "")
    res: SupportGrade = structured_model.invoke([
        SystemMessage(content=SUPPORT_SYSTEM),
        HumanMessage(content=f"{context}\n\nQuestion: {state['question']}\n\nAnswer: {state.get('answer', '')}"),
    ])

    return {"support_grade": res.label}
