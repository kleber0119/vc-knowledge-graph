"""
Run Step 1 end-to-end: Crawl → Clean → NER

Usage:
    cd vc-knowledge-graph
    python src/crawl/run_step1.py
"""
import sys
from pathlib import Path

# Make sure imports work from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.crawl.crawler import crawl, SEED_URLS as ALL_SEED_URLS
from src.crawl.cleaner import clean_corpus
from src.ie.ner import run_ner


def main():
    print("=" * 60)
    print("Step 1: Data Acquisition & Information Extraction")
    print("=" * 60)

    # Step 1a: Crawl
    print("\n[1/3] Crawling seed URLs...")
    crawl(ALL_SEED_URLS, output_dir=Path("data/raw"))

    # Step 1b: Clean
    print("\n[2/3] Cleaning documents...")
    clean_corpus(
        raw_file=Path("data/raw/raw_documents.jsonl"),
        output_dir=Path("data/cleaned"),
    )

    # Step 1c: NER
    print("\n[3/3] Running Named Entity Recognition...")
    run_ner(
        cleaned_file=Path("data/cleaned/cleaned_documents.jsonl"),
        output_dir=Path("data/ner"),
    )

    print("\nDone! Output files:")
    print("  data/raw/raw_documents.jsonl")
    print("  data/cleaned/cleaned_documents.jsonl")
    print("  data/ner/ner_results.jsonl")
    print("  data/ner/entity_counts.json")
    print("  data/ner/ambiguity_cases.json")


if __name__ == "__main__":
    main()
