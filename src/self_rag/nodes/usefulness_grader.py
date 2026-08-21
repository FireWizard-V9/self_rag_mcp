from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.models.schemas import UsefulnessGrade
from self_rag.prompts.usefulness import USEFULNESS_PROMPT


def usefulness_grader(state: GraphState) -> dict:

    structured_model = get_chat_model().with_structured_output(UsefulnessGrade)

    prompt = USEFULNESS_PROMPT.format(
        question=state["question"],
        answer=state.get("answer", ""),
    )

    res: UsefulnessGrade = structured_model.invoke(prompt)

    return {"usefulness_grade": res.label}
