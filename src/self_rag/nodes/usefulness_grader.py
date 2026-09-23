from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.models.schemas import UsefulnessGrade
from self_rag.prompts.usefulness import USEFULNESS_SYSTEM


def usefulness_grader(state: GraphState) -> dict:
    structured_model = get_chat_model().with_structured_output(UsefulnessGrade)

    res: UsefulnessGrade = structured_model.invoke([
        SystemMessage(content=USEFULNESS_SYSTEM),
        HumanMessage(content=f"Question: {state['question']}\n\nAnswer: {state.get('answer', '')}"),
    ])

    return {"usefulness_grade": res.label}
