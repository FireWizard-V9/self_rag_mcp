from langchain_core.messages import HumanMessage, SystemMessage

from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.models.schemas import RelevanceGrade
from self_rag.prompts.relevance import RELEVANCE_SYSTEM


def relevance_grader(state: GraphState) -> dict:
    structured_model = get_chat_model().with_structured_output(RelevanceGrade)

    filtered_docs = []
    docs_to_grade = state.get("retrieved_documents") or state.get("documents", [])

    for doc in docs_to_grade:
        res: RelevanceGrade = structured_model.invoke([
            SystemMessage(content=RELEVANCE_SYSTEM),
            HumanMessage(content=f"Query: {state['question']}\n\nDocument:\n{doc.page_content[:1000]}"),
        ])

        if res.answer == "YES":
            filtered_docs.append(doc)

    return {"relevant_documents": filtered_docs}
