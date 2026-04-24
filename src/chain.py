# src/chain.py
# Using Groq API (Free) instead of OpenAI

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)


def build_rag_chain(retriever):
    """
    Builds RAG pipeline using FREE Groq API.
    """

    # --- Free Groq LLM ---
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # --- Prompt ---
    prompt_template = """
You are a helpful Data Science tutor.
Use ONLY the following context from the student's notes to answer the question.
If the answer is not in the context, say: "I couldn't find that in your notes. Try rephrasing or upload more relevant documents."
Do NOT use outside knowledge.

Context from notes:
{context}

Student's Question: {question}

Answer (clear and educational):"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # --- Build Chain ---
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain


def ask_question(chain, question: str) -> dict:
    """
    Sends a question through the RAG chain.
    """
    result = chain.invoke({"query": question})

    answer = result["result"]

    sources = list(set([
        doc.metadata.get("source", "Unknown")
        for doc in result["source_documents"]
    ]))

    return {
        "answer": answer,
        "sources": sources
    }