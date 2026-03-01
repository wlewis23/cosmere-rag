"""
Embeds chunked Documents and stores them in a persistent Chroma vector store.

Flow:
  load_and_chunk_all()  →  list[Document]
  HuggingFaceEmbeddings →  384-dim vectors (all-MiniLM-L6-v2)
  Chroma.add_documents() →  persisted to chroma_db/

Why all-MiniLM-L6-v2?
  - Runs fully local (no API key, no cost)
  - Fast: ~14,000 sentences/sec on CPU
  - 384 dimensions — small enough for Chroma to handle efficiently
  - Strong semantic similarity performance for its size

Why batch in groups of 500?
  - sentence-transformers loads all texts into RAM before encoding.
    Batching keeps peak memory predictable (~200MB per batch).
  - Chroma also benefits from batched writes vs one-at-a-time inserts.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

from src.processing.chunker import load_and_chunk_all

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "cosmere"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 500


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a LangChain-compatible embedding model backed by sentence-transformers.
    The model is downloaded once (~80MB) and cached in ~/.cache/huggingface/.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity works better normalised
    )


def get_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Returns a Chroma vector store connected to the persisted collection.
    Creates the collection if it doesn't exist yet.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def build_index(docs: list[Document] | None = None) -> Chroma:
    """
    Embeds all documents and writes them to the Chroma store.

    If docs is None, loads and chunks everything in data/raw/ automatically.
    Returns the populated Chroma store (also persisted to disk).

    The store is append-only here — run scripts/build_index.py --reset
    to wipe and rebuild from scratch.
    """
    if docs is None:
        print("Loading and chunking raw pages...")
        docs = load_and_chunk_all()

    print(f"Total chunks to index: {len(docs)}")

    print(f"Loading embedding model '{EMBED_MODEL}' (downloads on first run)...")
    embeddings = get_embeddings()

    store = get_vector_store(embeddings)

    # Check how many documents are already in the store
    existing_count = store._collection.count()
    if existing_count > 0:
        print(f"Chroma already contains {existing_count} vectors. Appending new chunks.")

    # Add in batches with a progress bar
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Indexing batches"):
        store.add_documents(batch)

    final_count = store._collection.count()
    print(f"\nIndexing complete. Chroma store contains {final_count} vectors.")
    print(f"Persisted to: {CHROMA_DIR.resolve()}/")

    return store


def load_index() -> Chroma:
    """
    Loads an existing Chroma store from disk without re-embedding anything.
    Raises FileNotFoundError if the store hasn't been built yet.
    """
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            f"No Chroma store found at '{CHROMA_DIR}'. "
            "Run 'python scripts/build_index.py' first."
        )
    embeddings = get_embeddings()
    return get_vector_store(embeddings)
