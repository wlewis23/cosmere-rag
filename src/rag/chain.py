"""
The RAG chain: retrieves relevant Coppermind chunks then generates an answer with Claude.

Architecture (LangChain Expression Language / LCEL):

    query  →  retriever  →  format_context  →  prompt  →  Claude  →  answer

Why LCEL?
  - The pipe (|) syntax is composable and easy to read/debug
  - Each step is a Runnable — swappable without rewriting the whole chain
  - Supports streaming out of the box (used in the CLI)

Why MMR retrieval instead of plain similarity?
  - Plain top-k can return 5 near-identical chunks from the same paragraph.
  - MMR (Maximal Marginal Relevance) balances similarity to the query against
    diversity among results, so we get broader coverage per query.
  - fetch_k=20 candidates are scored; the top k=6 diverse ones are kept.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.rag.indexer import load_index

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_MODEL = "claude-sonnet-4-6"
RETRIEVAL_K = 6       # number of chunks returned per query
RETRIEVAL_FETCH_K = 20  # MMR candidate pool size (larger → better diversity)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert on Brandon Sanderson's Cosmere universe, \
with deep knowledge of every book, character, magic system, and world.

Answer the user's question using ONLY the context passages provided below. \
Each passage is labelled with its source article.

Rules:
- If the context contains the answer, give a thorough, accurate response.
- Cite your sources inline by naming the article (e.g. "According to the Kaladin article...").
- If the context does not contain enough information to answer, say so clearly — \
  do not invent details or draw on outside knowledge.
- Keep answers focused. Avoid padding or unnecessary repetition.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(docs: list[Document]) -> str:
    """
    Formats retrieved documents into a numbered context block for the prompt.
    Including the source title helps Claude cite correctly and helps us debug
    which chunks were retrieved.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------

def build_chain(stream: bool = False):
    """
    Builds and returns the full RAG chain.

    Args:
        stream: If True the chain will yield text tokens as they arrive
                (use with .stream()). If False, use .invoke() for a full string.

    Returns a LangChain Runnable that accepts {"question": str} and produces
    a string answer.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    store = load_index()

    # MMR retriever — diverse, relevant chunks
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_FETCH_K},
    )

    llm = ChatAnthropic(
        model=LLM_MODEL,
        api_key=api_key,
        streaming=stream,
    )

    # LCEL chain
    # RunnablePassthrough keeps the original question available downstream
    # after the retriever has consumed it.
    chain = (
        {
            "context": retriever | _format_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str) -> tuple[str, list[Document]]:
    """
    Convenience function: runs a query and returns (answer, source_docs).
    Source docs are fetched separately so callers can display citations.
    """
    store = load_index()
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_FETCH_K},
    )

    docs = retriever.invoke(question)
    chain = build_chain()
    answer = chain.invoke(question)

    return answer, docs
