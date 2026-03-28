"""
Run Step 2: Build the RDF Knowledge Graph

Usage:
    cd vc-knowledge-graph
    python src/kg/run_step2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.kg.builder import build_graph


def main():
    print("=" * 60)
    print("Step 2: RDF Graph Construction")
    print("=" * 60)

    build_graph(
        ner_file=Path("data/ner/ner_results.jsonl"),
        cleaned_file=Path("data/cleaned/cleaned_documents.jsonl"),
        output_dir=Path("kg_artifacts"),
    )

    print("\nDone! Output files:")
    print("  kg_artifacts/initial_graph.ttl")


if __name__ == "__main__":
    main()
