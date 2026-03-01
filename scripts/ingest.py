"""
Run the full ingestion pipeline.

Usage:
    python scripts/ingest.py           # fetch all ~3800 Cosmere pages
    python scripts/ingest.py --limit 50  # fetch 50 pages (for testing)
"""

import argparse
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.fetch_wiki import run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Ingest Coppermind wiki pages")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pages to fetch (for testing)",
    )
    args = parser.parse_args()
    run_ingestion(limit=args.limit)


if __name__ == "__main__":
    main()
