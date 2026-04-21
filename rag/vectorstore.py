import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "internal_docs"
_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)


def initialize_vectorstore(documents: list[Document]) -> Chroma:
    """
    Create or load a ChromaDB vector store.
    If the collection already exists and is populated, reuse it.
    If documents are provided and the collection is empty, add them.
    """
    embeddings = _get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    existing = vectorstore._collection.count()
    if existing == 0 and documents:
        vectorstore.add_documents(documents)
    elif existing > 0:
        print(f"[VectorStore] Reusing {existing} existing document chunks.")

    return vectorstore


def reset_vectorstore(documents: list[Document]) -> Chroma:
    """Drop the existing collection and rebuild from documents."""
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.delete_collection()
    return initialize_vectorstore(documents)
