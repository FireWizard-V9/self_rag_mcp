from self_rag.clients.llm import get_chat_model
from self_rag.models.graph_state import GraphState
from self_rag.models.schemas import RelevanceGrade
from self_rag.prompts.relevance import RELEVANCE_PROMPT


def relevance_grader(state: GraphState) -> dict:
    structured_model = get_chat_model().with_structured_output(RelevanceGrade)

    filtered_docs = []
    # Use retrieved_documents if present, otherwise fall back to documents
    docs_to_grade = state.get("retrieved_documents") or state.get("documents", [])

    for doc in docs_to_grade:
        prompt = RELEVANCE_PROMPT.format(
            question=state["question"],
            document=doc.page_content[:1000],
        )

        res: RelevanceGrade = structured_model.invoke(prompt)

        if res.answer == "YES":
            filtered_docs.append(doc)

    return {"relevant_documents": filtered_docs}
