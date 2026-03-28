"""
Text cleaner for the VC Knowledge Graph project.

After crawling, raw text is messy:
  - Extra whitespace and newlines
  - Very short fragments (navigation links, captions)
  - Duplicate sentences (headers repeated in content)
  - Unicode noise

This module cleans raw documents so NER runs on clean input.
Graded under "Cleaning + NER" (1 pt).
"""

import json
import logging
import re
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Minimum sentence length in characters — filters out "See also", "Edit", etc.
MIN_SENTENCE_CHARS = 40


def normalize_unicode(text: str) -> str:
    """Convert non-ASCII characters to their closest ASCII equivalent."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def remove_noise(text: str) -> str:
    """
    Remove common web page noise patterns:
    - Multiple whitespace / newlines
    - Wikipedia edit tags like [edit], [1], [citation needed]
    - URLs
    - Leftover HTML entities (&amp; etc.)
    """
    # Wikipedia reference markers like [1], [12], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)

    # Bare URLs
    text = re.sub(r"https?://\S+", "", text)

    # HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter on period/exclamation/question mark.
    For production you'd use spaCy's sentencizer, but this is fast and good enough
    for our cleaning pass.
    """
    # Split on sentence-ending punctuation followed by a space and capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def deduplicate(sentences: list[str]) -> list[str]:
    """
    Remove exact duplicate sentences while preserving order.
    Wikipedia pages often repeat section headings.
    """
    seen: set[str] = set()
    unique = []
    for s in sentences:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def clean_document(raw_text: str) -> dict:
    """
    Full cleaning pipeline for one document:
      1. Normalize unicode
      2. Remove noise patterns
      3. Split into sentences
      4. Filter short fragments
      5. Deduplicate

    Returns a dict with both the sentence list and a joined 'clean_text'.
    """
    text = normalize_unicode(raw_text)
    text = remove_noise(text)
    sentences = split_sentences(text)

    # Filter very short fragments (navigation links, captions, etc.)
    sentences = [s for s in sentences if len(s) >= MIN_SENTENCE_CHARS]

    # Deduplicate
    sentences = deduplicate(sentences)

    return {
        "sentences": sentences,
        "clean_text": " ".join(sentences),
        "sentence_count": len(sentences),
    }


def clean_corpus(raw_file: Path, output_dir: Path) -> list[dict]:
    """
    Read raw_documents.jsonl, clean each document, and save cleaned_documents.jsonl.
    Also prints cleaning statistics (useful for the report).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_docs = []
    with open(raw_file, encoding="utf-8") as f:
        for line in f:
            raw_docs.append(json.loads(line))

    cleaned_docs = []
    total_raw_chars = 0
    total_clean_chars = 0
    total_sentences = 0

    for doc in raw_docs:
        if doc.get("status") != "ok" or not doc.get("text"):
            continue

        cleaned = clean_document(doc["text"])
        out_doc = {
            "url": doc["url"],
            "title": doc.get("title", ""),
            **cleaned,
        }
        cleaned_docs.append(out_doc)

        raw_chars = len(doc["text"])
        clean_chars = len(cleaned["clean_text"])
        total_raw_chars += raw_chars
        total_clean_chars += clean_chars
        total_sentences += cleaned["sentence_count"]

        reduction = 100 * (1 - clean_chars / raw_chars) if raw_chars else 0
        logger.info(
            f"  {doc['url'][:60]}: {raw_chars:,} → {clean_chars:,} chars "
            f"({reduction:.1f}% reduction, {cleaned['sentence_count']} sentences)"
        )

    out_file = output_dir / "cleaned_documents.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in cleaned_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # ── Print cleaning statistics (include these in your report!) ─────────────
    logger.info("\n=== Cleaning Statistics ===")
    logger.info(f"Documents processed : {len(cleaned_docs)}")
    logger.info(f"Total raw chars     : {total_raw_chars:,}")
    logger.info(f"Total clean chars   : {total_clean_chars:,}")
    logger.info(
        f"Overall reduction   : {100*(1-total_clean_chars/total_raw_chars):.1f}%"
        if total_raw_chars else ""
    )
    logger.info(f"Total sentences     : {total_sentences:,}")
    logger.info(f"Saved to {out_file}")

    return cleaned_docs


if __name__ == "__main__":
    clean_corpus(
        raw_file=Path("data/raw/raw_documents.jsonl"),
        output_dir=Path("data/cleaned"),
    )
