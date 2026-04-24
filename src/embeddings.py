# src/embeddings.py
# PURPOSE: Split text into chunks and convert them into vector embeddings.
# WHY CHUNKS? LLMs have context limits. We split text so we only send
# the RELEVANT pieces to the LLM, not the entire document.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def split_text_into_chunks(text: str, source_name: str = "unknown") -> list[Document]:
    """
    Splits a large text string into smaller overlapping chunks.

    Args:
        text: The full document text.
        source_name: The filename — stored as metadata so we can cite sources later.

    Returns:
        A list of LangChain Document objects (chunk text + metadata).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Each chunk is ~800 characters
        chunk_overlap=100,    # 100 chars overlap so context isn't lost at edges
        length_function=len,
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source_name}]  # Tag each chunk with its filename
    )

    print(f"Split '{source_name}' into {len(chunks)} chunks.")
    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads a local embedding model (no API key needed).
    This model converts text into numerical vectors for similarity search.

    First run downloads ~90MB model. Subsequent runs are instant.
    """
    print("Loading embedding model (first run downloads ~90MB)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Fast, small, and accurate for Q&A
        model_kwargs={"device": "cpu"},   # Use CPU (works on all machines)
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Embedding model loaded.")
    return embeddings