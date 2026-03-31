"""
RDF Graph Builder for the VC Knowledge Graph.

Reads:
  - data/ner/ner_results.jsonl     — entities per document (filtered, canonical)
  - data/cleaned/cleaned_documents.jsonl — sentences per document

Pipeline:
  1. Build a global entity registry: text → (URI, class)
  2. Apply label overrides (fix mislabelled entities)
  3. Apply alias resolution (merge prefix duplicates into one URI)
  4. For each sentence, find which entities appear in it
  5. Apply relation patterns (keyword + co-occurrence) to emit triples
  6. Serialize to kg_artifacts/initial_graph.ttl
  7. Print KB statistics
"""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import XSD

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Namespaces ─────────────────────────────────────────────────────────────────
VCKG = Namespace("http://vckg.org/ontology#")
SCHEMA = Namespace("https://schema.org/")

# ── NER label → ontology class ─────────────────────────────────────────────────
LABEL_TO_CLASS = {
    "VC_FIRM":       VCKG.VCFirm,
    "ORG":           VCKG.Company,
    "PERSON":        VCKG.Person,
    "SECTOR":        VCKG.Sector,
    "GPE":           None,          # used as literal on headquarteredIn, not a node
    "PRODUCT":       VCKG.Company,
    "FUNDING_ROUND": VCKG.FundingRound,
    "EXIT_TYPE":     VCKG.ExitEvent,
}

# ── Label overrides ─────────────────────────────────────────────────────────────
# Entities that spaCy mislabels. Maps entity text (lowercase) → correct NER label.
LABEL_OVERRIDES: dict[str, str] = {
    "fintech":              "SECTOR",
    "saas":                 "SECTOR",
    "ai":                   "SECTOR",
    "ai/ml":                "SECTOR",
    "deep tech":            "SECTOR",
    "clean tech":           "SECTOR",
    "biotech":              "SECTOR",
    "edtech":               "SECTOR",
    "healthtech":           "SECTOR",
}

# Entities to drop entirely — extraction artifacts or clearly out-of-domain.
ENTITY_BLOCKLIST: set[str] = {
    # concatenation / extraction artifacts
    "marc andreessen marc",
    "thiel capital thiel capital",
    "valar ventures valar venture",
    "anduril industries anduril",
    "vc shaun maguire",
    "pronomos capital thiel",
    "starinvestor peter thiel",
    "tap into trump",
    "thiel to epstein",
    "peter thiel political campaign contribution",
    "moira weigel palantir",
    "lowell andreessen",
    "rooney, kate",
    # out-of-domain / noise
    "fbi",
    "isbn",
    "dei",
    "the white house",
    "sri lanka guardian",
    "quantum system",
    "loizo",
    "breakout lab",
    "breakout venture",
    # duplicate aliases (handled via alias resolution → canonical)
    "yc",     # → YCombinator
    "a16z",   # → AndreessenHorowitz
    # too-generic single abbreviation
    "vc",
}

# Override class for known institutions that spaCy tags as ORG but are not companies.
CLASS_OVERRIDES: dict[str, URIRef] = {
    "stanford university": VCKG.Organization,
    "stanford":            VCKG.Organization,
    "harvard":             VCKG.Organization,
    "mit":                 VCKG.Organization,
    "yale":                VCKG.Organization,
    "columbia":            VCKG.Organization,
    "oxford":              VCKG.Organization,
    "cambridge":           VCKG.Organization,
}

# ── Controlled-vocabulary mapping for funding rounds and exit types ─────────────
FUNDING_ROUND_URI = {
    "series a":              VCKG.SeriesA,
    "series b":              VCKG.SeriesB,
    "series c":              VCKG.SeriesC,
    "series d":              VCKG.SeriesD,
    "seed round":            VCKG.SeedRound,
    "seed funding":          VCKG.SeedRound,
    "angel round":           VCKG.SeedRound,
}

EXIT_TYPE_URI = {
    "ipo":                       VCKG.IPO,
    "initial public offering":   VCKG.IPO,
    "acquisition":               VCKG.Acquisition,
    "merger":                    VCKG.Merger,
}

# ── Keywords that indicate a real HQ relationship (not just GPE co-occurrence) ──
HQ_KEYWORDS = [
    "headquartered in", "headquartered at",
    "based in", "based out of",
    "located in", "located at",
    "offices in", "office in",
    "founded in",
]

# ── Relation patterns ──────────────────────────────────────────────────────────
RELATION_PATTERNS = [
    {
        "predicate":      VCKG.foundedBy,
        "subject_labels": {"ORG", "VC_FIRM"},
        "object_labels":  {"PERSON"},
        "keywords":       ["founded", "co-founded", "cofounded", "started", "created"],
    },
    {
        "predicate":      VCKG.investedIn,
        "subject_labels": {"VC_FIRM"},
        "object_labels":  {"ORG"},
        "keywords":       ["invested", "investment", "backed", "funded", "portfolio", "lead investor"],
    },
    {
        "predicate":      VCKG.partnerAt,
        "subject_labels": {"PERSON"},
        "object_labels":  {"VC_FIRM"},
        "keywords":       ["partner", "general partner", "managing partner", "joined", "co-founder of"],
    },
    {
        "predicate":      VCKG.ceoOf,
        "subject_labels": {"PERSON"},
        "object_labels":  {"ORG", "VC_FIRM"},
        "keywords":       ["ceo", "chief executive", "leads", "runs", "head of"],
    },
    {
        "predicate":      VCKG.operatesIn,
        "subject_labels": {"ORG", "VC_FIRM"},
        "object_labels":  {"SECTOR"},
        "keywords":       [],
    },
    {
        "predicate":      VCKG.hasFundingRound,
        "subject_labels": {"ORG"},
        "object_labels":  {"FUNDING_ROUND"},
        "keywords":       [],
    },
    {
        "predicate":      VCKG.hadExit,
        "subject_labels": {"ORG"},
        "object_labels":  {"EXIT_TYPE"},
        "keywords":       [],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# URI utilities
# ═══════════════════════════════════════════════════════════════════════════════

def make_uri_slug(text: str) -> str:
    """Convert an entity name to a CamelCase URI slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    parts = re.sub(r"[^a-zA-Z0-9\s]", "", text).split()
    return "".join(p.capitalize() for p in parts) if parts else "Unknown"


def entity_uri(text: str, label: str) -> URIRef:
    """Return the URIRef for an entity."""
    if label == "FUNDING_ROUND":
        return FUNDING_ROUND_URI.get(text.lower(), VCKG[make_uri_slug(text)])
    if label == "EXIT_TYPE":
        return EXIT_TYPE_URI.get(text.lower(), VCKG[make_uri_slug(text)])
    return VCKG[make_uri_slug(text)]


# ═══════════════════════════════════════════════════════════════════════════════
# Entity registry + cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def build_entity_registry(ner_docs: list[dict]) -> dict[str, dict]:
    """
    Build a global map: entity_text → {uri, label, class_uri}.
    Applies LABEL_OVERRIDES to fix mislabelled entities.
    """
    registry: dict[str, dict] = {}
    for doc in ner_docs:
        for ent in doc.get("entities", []):
            text = ent["text"]
            tokens = text.lower().split()
            if not text or text in registry or text.lower() in ENTITY_BLOCKLIST:
                continue
            # Apply label override if present
            label = LABEL_OVERRIDES.get(text.lower(), ent["label"])
            # Drop single-token PERSON entities — they are first-name-only artifacts
            if label == "PERSON" and len(tokens) < 2:
                continue
            # Drop ORG/VC_FIRM entities longer than 5 tokens — likely sentence fragments
            if label in {"ORG", "VC_FIRM"} and len(tokens) > 5:
                continue
            # Drop entities with repeated word sequences (e.g. "Thiel Capital Thiel Capital")
            if len(tokens) >= 4:
                half = len(tokens) // 2
                if tokens[:half] == tokens[half : half * 2]:
                    continue
            # Apply class override for known institutions, then fall back to label map
            cls = CLASS_OVERRIDES.get(text.lower(), LABEL_TO_CLASS.get(label))
            registry[text] = {
                "uri":       entity_uri(text, label),
                "label":     label,
                "class_uri": cls,
            }
    return registry


def resolve_aliases(registry: dict[str, dict]) -> dict[str, dict]:
    """
    Merge entities where one name is a word-level prefix of another within the
    same label group. The longer (more specific) name becomes canonical and all
    shorter aliases point to its URI.

    Example:
      "Sequoia"          → merges into vckg:SequoiaCapital
      "Stanford"         → merges into vckg:StanfordUniversity
      "Andreessen"       → already merged at NER stage; handled here as fallback
    """
    # Group texts by label
    by_label: dict[str, list[str]] = defaultdict(list)
    for text, info in registry.items():
        by_label[info["label"]].append(text)

    # alias_map: short_text → canonical_text
    alias_map: dict[str, str] = {}

    for label, texts in by_label.items():
        # Sort longest (most specific) first
        sorted_texts = sorted(texts, key=lambda t: len(t.split()), reverse=True)

        for longer in sorted_texts:
            longer_words = longer.lower().split()
            for shorter in sorted_texts:
                if shorter == longer or len(shorter) >= len(longer):
                    continue
                shorter_words = shorter.lower().split()
                # shorter is a word-level prefix of longer
                if longer_words[: len(shorter_words)] == shorter_words:
                    # Only merge if shorter has no better (longer) canonical yet
                    if shorter not in alias_map:
                        alias_map[shorter] = longer

    # Apply the alias map: point all aliases to the canonical URI
    for alias, canonical in alias_map.items():
        if alias in registry and canonical in registry:
            canonical_uri = registry[canonical]["uri"]
            registry[alias]["uri"] = canonical_uri
            logger.info(f"  Alias merged: '{alias}' → '{canonical}'")

    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# Graph building
# ═══════════════════════════════════════════════════════════════════════════════

def add_entity_triples(g: Graph, registry: dict[str, dict]) -> None:
    """
    Declare every entity as an instance of its ontology class.
    Because aliases share a URI, duplicate rdf:type triples are silently ignored
    by the graph (sets semantics), but we add all rdfs:label variants so the
    node retains every surface form it was seen as.
    """
    for text, info in registry.items():
        uri = info["uri"]
        cls = info["class_uri"]
        if cls is None:
            continue
        g.add((uri, RDF.type, cls))
        g.add((uri, RDFS.label, Literal(text, lang="en")))


def extract_relation_triples(
    g: Graph,
    sentences: list[str],
    doc_entities: list[dict],
    registry: dict[str, dict],
) -> int:
    """
    For each sentence, find entity mentions and apply relation patterns.
    Returns the number of triples added.
    """
    added = 0
    doc_lookup = {e["text"]: e for e in doc_entities if e["text"] in registry}

    for sentence in sentences:
        s_lower = sentence.lower()

        present = [
            registry[text] | {"text": text}
            for text in doc_lookup
            if text.lower() in s_lower and registry[text]["class_uri"] is not None
        ]

        if len(present) < 2:
            continue

        for subj_info in present:
            for obj_info in present:
                if subj_info["uri"] == obj_info["uri"]:
                    continue

                for pattern in RELATION_PATTERNS:
                    if subj_info["label"] not in pattern["subject_labels"]:
                        continue
                    if obj_info["label"] not in pattern["object_labels"]:
                        continue
                    keywords = pattern["keywords"]
                    if keywords and not any(kw in s_lower for kw in keywords):
                        continue

                    triple = (subj_info["uri"], pattern["predicate"], obj_info["uri"])
                    if triple not in g:
                        g.add(triple)
                        added += 1

        # GPE → headquarteredIn: only when the sentence explicitly signals a HQ relation
        if any(kw in s_lower for kw in HQ_KEYWORDS):
            gpe_mentions = [e for e in doc_entities if e["label"] == "GPE" and e["text"].lower() in s_lower]
            org_mentions = [e for e in present if e["label"] in {"ORG", "VC_FIRM"}]
            for org in org_mentions:
                for gpe in gpe_mentions:
                    triple = (org["uri"], VCKG.headquarteredIn, Literal(gpe["text"], lang="en"))
                    if triple not in g:
                        g.add(triple)
                        added += 1

    return added


def print_stats(g: Graph) -> None:
    """Print KB statistics."""
    logger.info("\n=== KB Statistics ===")
    logger.info(f"Total triples      : {len(g):,}")

    class_counts: dict[str, int] = defaultdict(int)
    for _, _, o in g.triples((None, RDF.type, None)):
        class_counts[str(o).split("#")[-1]] += 1
    for cls, cnt in sorted(class_counts.items()):
        logger.info(f"  {cls:25s}: {cnt} instances")

    pred_counts: dict[str, int] = defaultdict(int)
    for _, p, _ in g:
        pred_counts[str(p).split("#")[-1]] += 1
    logger.info("\n  Predicate distribution:")
    for pred, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {pred:30s}: {cnt}")


def build_graph(
    ner_file: Path,
    cleaned_file: Path,
    output_dir: Path,
) -> Graph:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(ner_file, encoding="utf-8") as f:
        ner_docs = [json.loads(line) for line in f]

    with open(cleaned_file, encoding="utf-8") as f:
        cleaned_docs = {doc["url"]: doc for doc in (json.loads(line) for line in f)}

    logger.info(f"Loaded {len(ner_docs)} NER documents")

    # ── Build + clean entity registry ─────────────────────────────────────────
    registry = build_entity_registry(ner_docs)
    logger.info(f"Entity registry before alias resolution: {len(registry)} entries")
    logger.info("Resolving aliases...")
    registry = resolve_aliases(registry)

    # ── Initialise graph ───────────────────────────────────────────────────────
    g = Graph()
    g.bind("vckg",   VCKG)
    g.bind("schema", SCHEMA)
    g.bind("rdf",    RDF)
    g.bind("rdfs",   RDFS)
    g.bind("owl",    OWL)
    g.bind("xsd",    XSD)

    g.add((URIRef("http://vckg.org/graph"), RDF.type, OWL.Ontology))
    g.add((URIRef("http://vckg.org/graph"), OWL.imports, URIRef("http://vckg.org/ontology")))

    add_entity_triples(g, registry)
    logger.info(f"After entity declarations: {len(g)} triples")

    # ── Relation triples ───────────────────────────────────────────────────────
    total_relation_triples = 0
    for ner_doc in ner_docs:
        url = ner_doc["url"]
        cleaned_doc = cleaned_docs.get(url, {})
        sentences = cleaned_doc.get("sentences", [])
        doc_entities = ner_doc.get("entities", [])
        n = extract_relation_triples(g, sentences, doc_entities, registry)
        total_relation_triples += n

    logger.info(f"Relation triples added: {total_relation_triples}")
    logger.info(f"Total triples in graph: {len(g)}")

    out_path = output_dir / "initial_graph.ttl"
    g.serialize(destination=str(out_path), format="turtle")
    logger.info(f"Saved to {out_path}")

    print_stats(g)
    return g


if __name__ == "__main__":
    build_graph(
        ner_file=Path("data/ner/ner_results.jsonl"),
        cleaned_file=Path("data/cleaned/cleaned_documents.jsonl"),
        output_dir=Path("kg_artifacts"),
    )
