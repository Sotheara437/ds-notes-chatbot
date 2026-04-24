# src/retriever.py
# PURPOSE: Store embeddings in ChromaDB and retrieve relevant chunks for a query.
# ChromaDB is like a smart filing cabinet — you store text+vectors,
# then ask "what's most similar to this question?" and it finds the best matches.

import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.embeddings import get_embedding_model


# Path where ChromaDB saves its files on disk
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "ds_notes"


def build_vector_store(chunks: list[Document]) -> Chroma:
    """
    Creates a new ChromaDB vector store from document chunks.
    This embeds all chunks and saves them to disk.

    Args:
        chunks: List of Document objects (from split_text_into_chunks).

    Returns:
        A Chroma vector store object.
    """
    embedding_model = get_embedding_model()

    print(f"Building vector store with {len(chunks)} chunks...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    print(f"Vector store saved to: {CHROMA_DB_PATH}")
    return vector_store


def load_vector_store() -> Chroma | None:
    """
    Loads an existing ChromaDB store from disk (if it exists).
    This avoids re-embedding everything on every app restart.

    Returns:
        Chroma vector store, or None if not yet created.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        print("No existing vector store found. Please index your PDFs first.")
        return None

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    print("Loaded existing vector store from disk.")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Creates a retriever that fetches the top-k most relevant chunks.

    Args:
        vector_store: Your Chroma vector store.
        k: Number of chunks to retrieve per query (4 is a good default).

    Returns:
        A LangChain retriever object.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever