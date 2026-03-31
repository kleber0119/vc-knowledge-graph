"""
Named Entity Recognition (NER) for the VC Knowledge Graph project.

This module extracts entities from cleaned text using spaCy.
We use two layers:
  1. spaCy's pretrained en_core_web_lg model (handles ORG, PERSON, MONEY, DATE, GPE)
  2. Custom EntityRuler patterns for VC-specific vocabulary

Three documented ambiguity cases illustrate NER challenges:
  1. "Apple" → fruit vs. Apple Inc. (ORG) — context-dependent
  2. "Amazon" → river/ecosystem vs. Amazon.com (ORG)
  3. "Mercury" → planet/element vs. Mercury (startup) vs. Freddie Mercury (PERSON)
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from spacy.language import Language

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── spaCy model ────────────────────────────────────────────────────────────────
# en_core_web_lg is a large English model with good NER accuracy.
# Install: python -m spacy download en_core_web_lg
MODEL_NAME = "en_core_web_lg"

# ── Entity type mapping ────────────────────────────────────────────────────────
# We keep these spaCy types (others like CARDINAL, ORDINAL are not useful for KG)
KEEP_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT"}

# ── Custom patterns for VC domain ──────────────────────────────────────────────
# Custom EntityRuler patterns for domain-specific terms fire BEFORE the ML model,
# ensuring well-known VC firms and investors are always correctly tagged.
VC_PATTERNS = [
    # VC Firms
    {"label": "VC_FIRM", "pattern": "Sequoia Capital"},
    {"label": "VC_FIRM", "pattern": "Sequoia"},
    {"label": "VC_FIRM", "pattern": "Andreessen Horowitz"},
    {"label": "VC_FIRM", "pattern": "a16z"},
    {"label": "VC_FIRM", "pattern": "Kleiner Perkins"},
    {"label": "VC_FIRM", "pattern": "KPCB"},
    {"label": "VC_FIRM", "pattern": "Y Combinator"},
    {"label": "VC_FIRM", "pattern": "YC"},
    {"label": "VC_FIRM", "pattern": "Benchmark Capital"},
    {"label": "VC_FIRM", "pattern": "Accel Partners"},
    {"label": "VC_FIRM", "pattern": "Accel"},
    {"label": "VC_FIRM", "pattern": "Tiger Global"},
    {"label": "VC_FIRM", "pattern": "General Catalyst"},
    {"label": "VC_FIRM", "pattern": "Lightspeed Venture Partners"},
    {"label": "VC_FIRM", "pattern": "Greylock Partners"},
    {"label": "VC_FIRM", "pattern": "Founders Fund"},
    {"label": "VC_FIRM", "pattern": "Index Ventures"},
    {"label": "VC_FIRM", "pattern": "NEA"},
    # Funding round types
    {"label": "FUNDING_ROUND", "pattern": "Series A"},
    {"label": "FUNDING_ROUND", "pattern": "Series B"},
    {"label": "FUNDING_ROUND", "pattern": "Series C"},
    {"label": "FUNDING_ROUND", "pattern": "Series D"},
    {"label": "FUNDING_ROUND", "pattern": "seed round"},
    {"label": "FUNDING_ROUND", "pattern": "seed funding"},
    {"label": "FUNDING_ROUND", "pattern": "angel round"},
    # Exit types
    {"label": "EXIT_TYPE", "pattern": "IPO"},
    {"label": "EXIT_TYPE", "pattern": "initial public offering"},
    {"label": "EXIT_TYPE", "pattern": "acquisition"},
    {"label": "EXIT_TYPE", "pattern": "merger"},
    # Sectors / verticals
    {"label": "SECTOR", "pattern": "SaaS"},
    {"label": "SECTOR", "pattern": "FinTech"},
    {"label": "SECTOR", "pattern": "HealthTech"},
    {"label": "SECTOR", "pattern": "EdTech"},
    {"label": "SECTOR", "pattern": "AI/ML"},
    {"label": "SECTOR", "pattern": "enterprise software"},
    {"label": "SECTOR", "pattern": "consumer internet"},
    {"label": "SECTOR", "pattern": "deep tech"},
    {"label": "SECTOR", "pattern": "clean tech"},
    {"label": "SECTOR", "pattern": "biotech"},
]

# Labels we want to keep in the final output (includes our custom labels)
ALL_KEEP_LABELS = KEEP_LABELS | {"VC_FIRM", "FUNDING_ROUND", "EXIT_TYPE", "SECTOR"}


def build_nlp() -> Language:
    """Load spaCy model and add the custom EntityRuler for VC terms."""
    logger.info(f"Loading spaCy model: {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME)

    # Add EntityRuler BEFORE the NER component so custom rules take priority
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    ruler.add_patterns(VC_PATTERNS)
    logger.info(f"Added {len(VC_PATTERNS)} custom VC patterns")
    return nlp


def normalize_entities(entities: list[dict]) -> list[dict]:
    """
    Normalize entity surface forms so that partial/possessive mentions
    collapse into one canonical entity.

    Two transformations, applied per label group:
      1. Strip possessives  — "Peter Thiel's" → "Peter Thiel"
      2. Resolve partials   — "Thiel" → "Peter Thiel" if a longer form
                              whose last word is "Thiel" exists in the same doc.

    The canonical form is always the longest surface form seen.
    """
    # Step 1: strip possessives
    for ent in entities:
        ent["text"] = ent["text"].rstrip("'s").rstrip("'s").strip()

    # Step 2: build canonical map per label (longest form wins)
    by_label: dict[str, set[str]] = defaultdict(set)
    for ent in entities:
        by_label[ent["label"]].add(ent["text"])

    canonical: dict[str, dict[str, str]] = {}  # label → {surface: canonical}
    for label, surface_forms in by_label.items():
        # Sort longest first so the first match is always the canonical form
        sorted_forms = sorted(surface_forms, key=len, reverse=True)
        mapping: dict[str, str] = {}
        for form in sorted_forms:
            if form in mapping:
                continue
            mapping[form] = form  # canonical for itself
            form_words = form.lower().split()
            # Map any shorter form that is a contiguous suffix of this form
            for other in sorted_forms:
                if other == form or len(other) >= len(form):
                    continue
                other_words = other.lower().split()
                if form_words[-len(other_words):] == other_words:
                    if other not in mapping:
                        mapping[other] = form
        canonical[label] = mapping

    # Step 3: apply mapping
    for ent in entities:
        ent["text"] = canonical.get(ent["label"], {}).get(ent["text"], ent["text"])

    return entities


def extract_entities_from_text(nlp: Language, text: str) -> list[dict]:
    """
    Run NER on text and return a list of normalized entity dicts.
    Each dict: {text, label, start_char, end_char}
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ALL_KEEP_LABELS:
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
    return normalize_entities(entities)


# ── Per-label caps (max unique entities kept, by descending frequency) ─────────
# Custom labels are kept in full; generic spaCy labels are capped.
LABEL_CAPS = {
    "PERSON":       50,
    "ORG":          50,
    "GPE":          30,
    "PRODUCT":      20,
    "VC_FIRM":      None,   # keep all
    "FUNDING_ROUND": None,
    "EXIT_TYPE":    None,
    "SECTOR":       None,
}

# Minimum number of mentions across all docs for an entity to be kept.
# Filters out entities that appear only in passing.
MIN_MENTIONS = 3


def build_global_canonical(all_entities: list[dict], global_counts: dict[str, Counter]) -> dict[str, dict[str, str]]:
    """
    Build a canonical name map across ALL documents.
    Rules:
      - For PERSON: discard entities with > 3 tokens (extraction artifacts).
        Then group by last name; if exactly one multi-word form shares a last
        name with a bare last-name mention, map the bare form to it.
        Canonical = most frequent form in that group, not longest.
      - For all other labels: map shorter forms to the most frequent form that
        contains them as a suffix. Only apply when there is exactly one candidate.
    Returns: label → {surface_form: canonical_form}
    """
    canonical: dict[str, dict[str, str]] = {}

    by_label: dict[str, set[str]] = defaultdict(set)
    for ent in all_entities:
        by_label[ent["label"]].add(ent["text"])

    for label, surface_forms in by_label.items():
        mapping: dict[str, str] = {}
        counts = global_counts.get(label, Counter())

        if label == "PERSON":
            # Drop extraction artifacts (> 3 tokens)
            clean_forms = {f for f in surface_forms if len(f.split()) <= 3}

            # Group by last word (last name)
            by_last: dict[str, list[str]] = defaultdict(list)
            for form in clean_forms:
                by_last[form.split()[-1].lower()].append(form)

            for _, forms in by_last.items():
                multi = [f for f in forms if len(f.split()) > 1]
                single = [f for f in forms if len(f.split()) == 1]

                if len(multi) == 1:
                    # Unambiguous: one full name → map bare last name to it
                    best = multi[0]
                    mapping[best] = best
                    for s in single:
                        mapping[s] = best
                elif len(multi) > 1:
                    # Ambiguous last name (e.g. two different people): keep each form as-is
                    for f in forms:
                        mapping[f] = f
                else:
                    # Only bare last name seen, keep it
                    for s in single:
                        mapping[s] = s

            # Mark artifacts as mapped to empty string so they get dropped
            for form in surface_forms:
                if form not in mapping:
                    mapping[form] = ""   # will be filtered out later

        else:
            # For ORG, GPE, etc.: map shorter forms to the most frequent longer form
            # that ends with them, only when there is exactly one candidate.
            sorted_by_freq = sorted(surface_forms, key=lambda f: counts[f], reverse=True)
            for form in sorted_by_freq:
                if form in mapping:
                    continue
                mapping[form] = form
                for other in surface_forms:
                    if other == form or len(other) >= len(form) or other in mapping:
                        continue
                    other_words = other.lower().split()
                    # Only map if this is the sole longer form ending with 'other'
                    candidates = [
                        f for f in surface_forms
                        if f != other and len(f) > len(other)
                        and f.lower().split()[-len(other_words):] == other_words
                    ]
                    if len(candidates) == 1:
                        mapping[other] = form

        canonical[label] = mapping

    return canonical


def run_ner(cleaned_file: Path, output_dir: Path) -> dict:
    """
    Run NER on all cleaned documents and save results.

    Pipeline:
      1. Extract raw entities from every document
      2. Strip possessives corpus-wide
      3. Build global canonical map (resolves partials across docs)
      4. Apply canonical map + count global frequencies
      5. Filter by MIN_MENTIONS and LABEL_CAPS (target ≤ 200 unique entities)
      6. Re-filter each document's entity list to only kept entities
      7. Save results

    Output:
      - ner_results.jsonl  — entities per document (filtered)
      - entity_counts.json — frequency counts per entity type (filtered)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp = build_nlp()

    with open(cleaned_file, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    logger.info(f"Running NER on {len(docs)} documents...")

    # ── Pass 1: extract raw entities from all docs ─────────────────────────────
    raw_results = []
    for doc in docs:
        text = doc.get("clean_text", "")
        if not text:
            continue
        entities = extract_entities_from_text(nlp, text)
        raw_results.append({"url": doc["url"], "title": doc.get("title", ""), "entities": entities})

    # ── Pass 2: global normalization ───────────────────────────────────────────
    # Strip possessives across all entities first
    all_entities = [ent for r in raw_results for ent in r["entities"]]
    for ent in all_entities:
        ent["text"] = ent["text"].rstrip("'s").rstrip("'s").strip()

    # Count raw frequencies (pre-canonical) so build_global_canonical can use them
    raw_counts: dict[str, Counter] = defaultdict(Counter)
    for ent in all_entities:
        raw_counts[ent["label"]][ent["text"]] += 1

    # Build canonical map from the full corpus
    canonical = build_global_canonical(all_entities, raw_counts)

    # Apply canonical map and count global frequencies
    global_counts: dict[str, Counter] = defaultdict(Counter)
    for ent in all_entities:
        ent["text"] = canonical.get(ent["label"], {}).get(ent["text"], ent["text"])
        global_counts[ent["label"]][ent["text"]] += 1

    # ── Pass 3: build the kept-entity whitelist ────────────────────────────────
    kept: dict[str, set[str]] = {}
    for label, counter in global_counts.items():
        cap = LABEL_CAPS.get(label, 30)
        # Filter by min mentions, then take top-N by frequency
        candidates = [(text, cnt) for text, cnt in counter.items() if cnt >= MIN_MENTIONS]
        candidates.sort(key=lambda x: x[1], reverse=True)
        if cap is not None:
            candidates = candidates[:cap]
        kept[label] = {text for text, _ in candidates}

    # ── Pass 4: filter each doc's entity list ─────────────────────────────────
    ner_results = []
    filtered_counts: dict[str, Counter] = defaultdict(Counter)
    for r in raw_results:
        filtered = [e for e in r["entities"] if e["text"] in kept.get(e["label"], set())]
        ner_results.append({
            "url": r["url"],
            "title": r["title"],
            "entity_count": len(filtered),
            "entities": filtered,
        })
        for ent in filtered:
            filtered_counts[ent["label"]][ent["text"]] += 1

    # ── Save NER results ───────────────────────────────────────────────────────
    out_file = output_dir / "ner_results.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in ner_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Save entity frequency counts ───────────────────────────────────────────
    counts_serializable = {
        label: dict(counter.most_common())
        for label, counter in filtered_counts.items()
    }
    counts_file = output_dir / "entity_counts.json"
    with open(counts_file, "w", encoding="utf-8") as f:
        json.dump(counts_serializable, f, indent=2, ensure_ascii=False)

    # ── Print summary ──────────────────────────────────────────────────────────
    logger.info("\n=== NER Summary ===")
    total_unique = sum(len(v) for v in kept.values())
    logger.info(f"Total unique entities (after filtering) : {total_unique}")
    for label, counter in sorted(filtered_counts.items()):
        logger.info(f"  {label:20s}: {sum(counter.values()):,} mentions, "
                    f"{len(counter):,} unique")

    logger.info("\n=== Top Entities per Type ===")
    for label, counter in sorted(filtered_counts.items()):
        logger.info(f"  {label}: {counter.most_common(5)}")

    # ── Print ambiguity examples ───────────────────────────────────────────────
    logger.info("\n=== Ambiguity Cases ===")
    ambiguity_cases = [
        {
            "entity": "Apple",
            "problem": "Could be Apple Inc. (ORG) or the fruit. Without context, "
                       "spaCy must rely on surrounding words.",
            "example_correct": "Apple's App Store generated $1.1B in revenue (ORG)",
            "example_ambiguous": "The company grew like an apple tree (could be either)",
            "resolution": "spaCy's ML model uses context window — 'revenue', 'store' "
                          "push toward ORG classification."
        },
        {
            "entity": "Amazon",
            "problem": "Amazon.com (ORG) vs. Amazon River/Rainforest (LOC/GPE). "
                       "Both appear in tech and environment articles.",
            "example_correct": "Amazon invested $4B in Anthropic (ORG)",
            "example_ambiguous": "Deforestation threatens the Amazon ecosystem (LOC)",
            "resolution": "Context words like 'invested', 'AWS', 'Bezos' disambiguate "
                          "toward ORG; 'rainforest', 'Brazil' push toward LOC."
        },
        {
            "entity": "Mercury",
            "problem": "Mercury (startup/company), Mercury (planet), or "
                       "Freddie Mercury (PERSON). Three distinct types.",
            "example_correct": "Mercury raised a $120M Series B for banking (ORG)",
            "example_ambiguous": "Mercury is closest to the Sun (could be planet or brand)",
            "resolution": "Without domain-specific patterns, generic NER often "
                          "mis-classifies Mercury the startup. Our EntityRuler "
                          "can be extended with specific company patterns to fix this."
        },
    ]
    ambiguity_file = output_dir / "ambiguity_cases.json"
    with open(ambiguity_file, "w", encoding="utf-8") as f:
        json.dump(ambiguity_cases, f, indent=2)
    logger.info(f"Saved 3 ambiguity cases to {ambiguity_file}")

    return counts_serializable


if __name__ == "__main__":
    run_ner(
        cleaned_file=Path("data/cleaned/cleaned_documents.jsonl"),
        output_dir=Path("data/ner"),
    )
