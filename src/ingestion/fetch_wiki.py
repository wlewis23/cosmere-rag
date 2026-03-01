"""
Fetches Cosmere article pages from the Coppermind wiki via the MediaWiki Action API.

The MediaWiki API lets us retrieve article text in bulk without scraping HTML.
We fetch all pages in Category:Cosmere, then pull the wikitext for each page
and save it as a JSON file per article in data/raw/.
"""

import json
import time
from pathlib import Path

# curl-cffi uses libcurl's TLS stack, which Cloudflare accepts.
# Plain `requests` (urllib3) has a known non-browser TLS fingerprint that
# Cloudflare's managed-challenge mode blocks — even with a proper User-Agent.
from curl_cffi import requests
from tqdm import tqdm

API_URL = "https://coppermind.net/w/api.php"
RAW_DATA_DIR = Path("data/raw")

# Polite crawl delay (seconds) — respect the server
CRAWL_DELAY = 0.5

# When using curl-cffi with impersonate="chrome", it sets its own browser headers.
# Overriding User-Agent would break the fingerprint match and trigger Cloudflare.
# No custom headers needed — the Chrome impersonation handles it.
HEADERS = {}


def get_all_cosmere_page_titles() -> list[str]:
    """
    Uses the MediaWiki 'categorymembers' query to list every page title
    in the Cosmere category (and its subcategories, recursively via generator).
    Returns a flat list of page titles.
    """
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Cosmere",
        "cmtype": "page",
        "cmlimit": "500",  # max allowed per request
        "format": "json",
    }

    print("Fetching page titles from Category:Cosmere...")
    while True:
        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30, impersonate="chrome")
        response.raise_for_status()
        data = response.json()

        members = data["query"]["categorymembers"]
        titles.extend(m["title"] for m in members)

        # MediaWiki paginates with a 'continue' token
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
        time.sleep(CRAWL_DELAY)

    print(f"Found {len(titles)} pages in Category:Cosmere")
    return titles


def fetch_page_wikitext(title: str) -> dict | None:
    """
    Fetches the plain wikitext content + metadata for a single page.
    Returns a dict with title, wikitext, and categories, or None on failure.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions|categories",
        "rvprop": "content",
        "rvslots": "main",
        "cllimit": "50",
        "format": "json",
        "formatversion": "2",  # cleaner response structure
    }

    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30, impersonate="chrome")
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    if not pages:
        return None

    page = pages[0]
    if "missing" in page:
        return None

    wikitext = page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
    categories = [c["title"].replace("Category:", "") for c in page.get("categories", [])]

    return {
        "title": page["title"],
        "wikitext": wikitext,
        "categories": categories,
    }


def fetch_pages_batch(titles: list[str]) -> list[dict]:
    """
    Fetches wikitext for up to 50 titles in one API call (MediaWiki limit).
    More efficient than fetching one at a time.
    """
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "revisions|categories",
        "rvprop": "content",
        "rvslots": "main",
        "cllimit": "50",
        "format": "json",
        "formatversion": "2",
    }

    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=60, impersonate="chrome")
    response.raise_for_status()
    data = response.json()

    results = []
    for page in data["query"]["pages"]:
        if "missing" in page:
            continue
        wikitext = (
            page.get("revisions", [{}])[0]
            .get("slots", {})
            .get("main", {})
            .get("content", "")
        )
        categories = [c["title"].replace("Category:", "") for c in page.get("categories", [])]
        results.append({
            "title": page["title"],
            "wikitext": wikitext,
            "categories": categories,
        })
    return results


def save_page(page: dict) -> None:
    """Saves a page dict as a JSON file. Filename is derived from the title."""
    safe_name = page["title"].replace("/", "_").replace(" ", "_")
    path = RAW_DATA_DIR / f"{safe_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(page, f, ensure_ascii=False, indent=2)


def run_ingestion(limit: int | None = None) -> None:
    """
    Main ingestion entry point.
    Fetches all Cosmere pages and saves them to data/raw/.

    Args:
        limit: Optional cap on number of pages to fetch (useful for testing).
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    titles = get_all_cosmere_page_titles()
    if limit:
        titles = titles[:limit]
        print(f"Limiting to {limit} pages for this run.")

    # Check which pages are already downloaded to allow resuming
    already_fetched = {p.stem.replace("_", " ") for p in RAW_DATA_DIR.glob("*.json")}
    titles_to_fetch = [t for t in titles if t not in already_fetched]
    print(f"{len(already_fetched)} pages already on disk. Fetching {len(titles_to_fetch)} new pages...")

    # Batch in groups of 50 (MediaWiki API limit for multi-title queries)
    batch_size = 50
    batches = [titles_to_fetch[i:i + batch_size] for i in range(0, len(titles_to_fetch), batch_size)]

    for batch in tqdm(batches, desc="Fetching pages"):
        pages = fetch_pages_batch(batch)
        for page in pages:
            save_page(page)
        time.sleep(CRAWL_DELAY)

    total = len(list(RAW_DATA_DIR.glob("*.json")))
    print(f"\nIngestion complete. {total} pages saved to {RAW_DATA_DIR}/")


if __name__ == "__main__":
    run_ingestion()
