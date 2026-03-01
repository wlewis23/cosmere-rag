"""
Converts raw Coppermind wikitext into clean LangChain Documents ready for embedding.

Two stages:
  1. clean_wikitext()  — strips MediaWiki markup, leaving readable prose
  2. chunk_document()  — splits prose into overlapping chunks with metadata

Why these chunk settings?
  - chunk_size=800: ~600 tokens, fits well within embedding model context windows
    and gives enough context for a retriever to find relevant passages.
  - chunk_overlap=100: repeated text at boundaries prevents answers being cut off
    mid-sentence when a relevant passage straddles two chunks.
  - RecursiveCharacterTextSplitter: tries to split on paragraphs → sentences →
    words in order, so chunks stay semantically coherent.
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DATA_DIR = Path("data/raw")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Wikitext cleaning
# ---------------------------------------------------------------------------

def _remove_nested_braces(text: str) -> str:
    """
    Removes {{ }} template blocks (infoboxes, navboxes, citation templates, etc.).
    MediaWiki templates can be nested, so a simple regex won't work — we walk
    the string character by character to track brace depth.
    """
    result = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
        elif text[i:i+2] == "}}":
            depth = max(0, depth - 1)
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    return "".join(result)


def _remove_nested_brackets(text: str) -> str:
    """
    Removes {| |} table markup (wikitables).
    Tables rarely contain useful prose for a RAG — mostly structured data
    that doesn't embed well as plain text.
    """
    result = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i:i+2] == "{|":
            depth += 1
            i += 2
        elif text[i:i+2] == "|}":
            depth = max(0, depth - 1)
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    return "".join(result)


def clean_wikitext(wikitext: str) -> str:
    """
    Strips MediaWiki markup from raw wikitext, returning clean readable prose.

    Removal order matters — templates/tables first, then inline markup,
    then whitespace normalisation.
    """
    text = wikitext

    # Remove <ref>...</ref> citation blocks (multi-line)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    # Self-closing refs: <ref name="foo" />
    text = re.sub(r"<ref[^/]*/?>", "", text)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove template blocks {{ }} (infoboxes, navboxes, etc.)
    text = _remove_nested_braces(text)

    # Remove wikitables {| |}
    text = _remove_nested_brackets(text)

    # Remove File/Image embeds: [[File:...]] or [[Image:...]]
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # Convert piped wikilinks [[target|display]] → display text
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    # Convert plain wikilinks [[target]] → target
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Remove external links [url display] → display
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    # Remove bare external links [url]
    text = re.sub(r"\[https?://\S+\]", "", text)

    # Strip bold/italic wiki markup (''' and '')
    text = re.sub(r"'{2,3}", "", text)

    # Strip HTML tags (<br>, <div>, <span>, etc.)
    text = re.sub(r"<[^>]+>", "", text)

    # Convert section headers == Heading == → plain text (keep the words)
    text = re.sub(r"={2,6}\s*(.+?)\s*={2,6}", r"\n\n\1\n", text)

    # Remove list markers (* # :) at line start but keep the text
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.MULTILINE)

    # Normalise whitespace: collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # Split order: paragraph → sentence-ish breaks → word breaks
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
)


def chunk_document(page: dict) -> list[Document]:
    """
    Takes a raw page dict (from data/raw/*.json) and returns a list of
    LangChain Documents, one per chunk.

    Each Document carries metadata so we can cite sources in answers:
      - source: the article title
      - categories: list of Coppermind categories (e.g. ['Stormlight Archive', 'Characters'])
      - chunk_index: position of this chunk within the article
    """
    title = page["title"]
    categories = page.get("categories", [])
    cleaned = clean_wikitext(page["wikitext"])

    if not cleaned.strip():
        return []

    chunks = _splitter.split_text(cleaned)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": title,
                    "categories": ", ".join(categories),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
        )
    return documents


def load_and_chunk_all(raw_dir: Path = RAW_DATA_DIR) -> list[Document]:
    """
    Loads every JSON file from data/raw/, cleans and chunks each one.
    Returns a flat list of all Documents across all articles.
    """
    all_docs = []
    json_files = sorted(raw_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {raw_dir}. Run 'python scripts/ingest.py' first."
        )

    for path in json_files:
        with open(path, encoding="utf-8") as f:
            page = json.load(f)
        docs = chunk_document(page)
        all_docs.extend(docs)

    return all_docs
