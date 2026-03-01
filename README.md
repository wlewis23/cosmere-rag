# cosmere-rag

First attempt at building a RAG to understand tooling and process. Built with claude-code using claude-sonnet-4.6 model.

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   Get a key at [console.anthropic.com](https://console.anthropic.com).

## Running the pipeline

### 1. Fetch wiki pages
```bash
python scripts/ingest.py             # fetch all ~3800 Cosmere pages
python scripts/ingest.py --limit 50  # fetch 50 pages (quick test)
```
Pages are saved to `data/raw/` and the fetcher is resumable — re-running skips already-downloaded pages.

### 2. Build the vector index
```bash
python scripts/build_index.py         # embed and index everything in data/raw/
python scripts/build_index.py --reset # wipe and rebuild from scratch
```

### 3. Query
```bash
python scripts/query.py                  # interactive REPL
python scripts/query.py "Who is Kaladin?" # single query
```

In the REPL, type `sources` to toggle citation display, and `quit` or `exit` to leave.

## Stack

- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers (local, no API key needed)
- **Vector DB:** Chroma (persisted to `chroma_db/`)
- **LLM:** Claude Sonnet via `langchain-anthropic`
- **Orchestration:** LangChain LCEL