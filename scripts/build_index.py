"""
Embed all chunked wiki pages and store them in the Chroma vector database.

Usage:
    python scripts/build_index.py           # embed everything in data/raw/
    python scripts/build_index.py --reset   # wipe the store first, then rebuild
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.indexer import CHROMA_DIR, build_index


def main():
    parser = argparse.ArgumentParser(description="Build the Chroma vector index")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma store before rebuilding",
    )
    args = parser.parse_args()

    if args.reset and CHROMA_DIR.exists():
        print(f"Wiping existing Chroma store at {CHROMA_DIR}/ ...")
        shutil.rmtree(CHROMA_DIR)

    build_index()


if __name__ == "__main__":
    main()
