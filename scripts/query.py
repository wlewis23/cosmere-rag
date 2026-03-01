"""
CLI interface for the Cosmere RAG system.

Two modes:
  Single query:    python scripts/query.py "Who is Kaladin?"
  Interactive REPL: python scripts/query.py

In both modes, answers stream token-by-token and source citations are printed
after the answer so you can see exactly which Coppermind articles were used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chain import build_chain, load_index, RETRIEVAL_K, RETRIEVAL_FETCH_K

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 60


def print_header():
    print()
    print("  Cosmere RAG")
    print("  Powered by Coppermind + Claude")
    print(f"{DIVIDER}")
    print("  Type your question and press Enter.")
    print("  Commands: 'quit' or 'exit' to leave, 'sources' to toggle citation display.")
    print(f"{DIVIDER}")
    print()


def print_sources(docs: list) -> None:
    """Prints the retrieved source chunks used to generate the answer."""
    print(f"\n{DIVIDER}")
    print(f"  Sources ({len(docs)} chunks retrieved)")
    print(DIVIDER)

    seen_sources = {}
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        total = doc.metadata.get("total_chunks", "?")
        if source not in seen_sources:
            seen_sources[source] = []
        seen_sources[source].append(f"chunk {chunk_idx}/{total}")

    for source, chunks in seen_sources.items():
        print(f"  • {source}  ({', '.join(chunks)})")
    print()


def run_query(question: str, chain, store, show_sources: bool = True) -> None:
    """Streams an answer for a single question, then prints sources."""
    print()

    # Retrieve docs first so we can show them after streaming
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_FETCH_K},
    )
    docs = retriever.invoke(question)

    # Stream the answer token by token
    for token in chain.stream(question):
        print(token, end="", flush=True)

    print()  # newline after streamed answer

    if show_sources:
        print_sources(docs)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def single_query_mode(question: str) -> None:
    """Run one question and exit — good for scripting."""
    from src.rag.chain import build_chain
    chain = build_chain(stream=True)
    store = load_index()
    run_query(question, chain, store, show_sources=True)


def interactive_mode() -> None:
    """REPL loop: prompt → stream answer → prompt again."""
    from src.rag.chain import build_chain

    print_header()

    print("  Loading embedding model and vector store...", end="", flush=True)
    store = load_index()
    chain = build_chain(stream=True)
    print(" ready.\n")

    show_sources = True

    while True:
        try:
            question = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("\n  Goodbye!")
            break

        if question.lower() == "sources":
            show_sources = not show_sources
            state = "on" if show_sources else "off"
            print(f"  Source citations turned {state}.\n")
            continue

        print(f"\n  Claude: ", end="", flush=True)
        run_query(question, chain, store, show_sources=show_sources)


def main():
    if len(sys.argv) > 1:
        # Join all args so the user doesn't need quotes:
        # python scripts/query.py Who is Kaladin?
        question = " ".join(sys.argv[1:])
        single_query_mode(question)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
