"""
Crawler for the VC/Silicon Valley Knowledge Graph project.

Design principles:
- Identifies itself with a descriptive User-Agent
- Checks robots.txt before crawling each domain
- Respects a crawl delay between requests
- Uses trafilatura for clean main-content extraction
- Saves raw text alongside metadata for reproducibility

Seed URLs: 8 focused Wikipedia pages → targets 50–200 unique entities.
(KGE expansion to 20k–50k triples is handled via SPARQL in a later step.)
"""

import json
import logging
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Ethical crawling settings ──────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "KGResearchBot/1.0 (academic knowledge-graph project; non-commercial)"
}
CRAWL_DELAY = 2.0
REQUEST_TIMEOUT = 15

# ── Seed URLs (8 pages, tightly scoped) ───────────────────────────────────────
# Selected to cover the three main entity types in the VC KG:
#   VC Firms, Key Investors, and Portfolio Companies.
# Wikipedia is preferred: open license, clean HTML, trafilatura handles it well.
SEED_URLS = [
    # VC Firms
    "https://en.wikipedia.org/wiki/Sequoia_Capital",
    "https://en.wikipedia.org/wiki/Andreessen_Horowitz",
    "https://en.wikipedia.org/wiki/Y_Combinator",
    # Key investors
    "https://en.wikipedia.org/wiki/Marc_Andreessen",
    "https://en.wikipedia.org/wiki/Peter_Thiel",
]


class RobotChecker:
    """
    Checks robots.txt per domain and caches the result.
    Checks robots.txt per domain before making any requests.
    """

    def __init__(self):
        self._cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{domain}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp.allow_all = True
            self._cache[domain] = rp
        return self._cache[domain].can_fetch(HEADERS["User-Agent"], url)


def extract_text(html: str, url: str) -> str:
    """
    Extract main article text using trafilatura.
    Removes boilerplate (navbars, ads, footers) and returns clean prose.
    """
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    return text or ""


def crawl(urls: list[str], output_dir: Path) -> list[dict]:
    """
    Crawl a list of URLs. For each URL:
      1. Check robots.txt (ethics)
      2. Fetch the page
      3. Extract main text via trafilatura
      4. Save to JSONL

    Returns list of document dicts: {url, title, text, status}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    robot_checker = RobotChecker()
    documents = []

    for i, url in enumerate(urls):
        logger.info(f"[{i+1}/{len(urls)}] Crawling: {url}")

        if not robot_checker.is_allowed(url):
            logger.warning(f"  BLOCKED by robots.txt: {url}")
            documents.append({"url": url, "status": "blocked_by_robots", "text": ""})
            continue

        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"  HTTP error: {e}")
            documents.append({"url": url, "status": f"http_error", "text": ""})
            time.sleep(CRAWL_DELAY)
            continue
        except Exception as e:
            logger.warning(f"  Error: {e}")
            documents.append({"url": url, "status": "error", "text": ""})
            time.sleep(CRAWL_DELAY)
            continue

        text = extract_text(response.text, url)

        # Extract title from trafilatura metadata
        meta = trafilatura.extract_metadata(response.text)
        title = meta.title if meta and meta.title else ""

        doc = {
            "url": url,
            "title": title,
            "text": text,
            "char_count": len(text),
            "status": "ok",
        }
        documents.append(doc)
        logger.info(f"  OK — {len(text):,} chars")

        time.sleep(CRAWL_DELAY)

    out_file = output_dir / "raw_documents.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    ok = sum(1 for d in documents if d["status"] == "ok")
    logger.info(f"\nCrawled {ok}/{len(urls)} pages → {out_file}")
    return documents


if __name__ == "__main__":
    crawl(SEED_URLS, output_dir=Path("data/raw"))
