from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from self_rag.core.config import get_settings


def load_documents() -> list[Document]:
    settings = get_settings()

    data_dir = Path(settings.data_dir)
    documents: list[Document] = []

    for pdf_path in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))

        loaded_documents = loader.load()
        documents.extend(loaded_documents)
    return documents
