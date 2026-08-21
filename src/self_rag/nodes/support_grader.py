from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.models.schemas import SupportGrade
from self_rag.prompts.support import SUPPORT_PROMPT


def support_grader(state: GraphState) -> dict:
    structured_model = get_chat_model().with_structured_output(SupportGrade)

    prompt = SUPPORT_PROMPT.format(
        context=state.get("context", ""),
        answer=state.get("answer", ""),
    )

    res: SupportGrade = structured_model.invoke(prompt)

    return {"support_grade": res.label}
