"""
Alignment for the VC Knowledge Graph — entities AND predicates.

Two things are aligned to Wikidata, which is the same concept applied to
different kinds of KG elements:

  Entity linking   — "vckg:SequoiaCapital is the same as wd:Q1852025"
                     Uses the Wikidata Search API per entity node.
                     → owl:sameAs (conf ≥ 0.85) or skos:closeMatch (≥ 0.70)

  Predicate alignment — "vckg:foundedBy means the same as wdt:P112"
                        Uses label-based + triple-based SPARQL on Wikidata.
                        → owl:equivalentProperty or skos:closeMatch

Both write to kg_artifacts/alignment.ttl.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS, XSD

from src.kg.builder import build_entity_registry, resolve_aliases, VCKG

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA = Namespace("https://schema.org/")
WD     = Namespace("http://www.wikidata.org/entity/")
WDT    = Namespace("http://www.wikidata.org/prop/direct/")

WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "KGResearchBot/1.0 (academic KG project; non-commercial)",
    "Accept":     "application/sparql-results+json",
}
REQUEST_DELAY = 1.0

# ── Confidence thresholds (entity linking) ─────────────────────────────────────
SAME_AS_THRESHOLD     = 0.85
CLOSE_MATCH_THRESHOLD = 0.70

# ── Local semantic definitions for entities with no Wikidata match ────────────
# Maps entity text (as it appears in NER data) → specific vckg subclass.
# These are manually reviewed decisions, making the pipeline reproducible.
LOCAL_CLASS_OVERRIDES: dict[str, URIRef] = {
    "René Girard":          VCKG.PublicFigure,   # French philosopher
    "Leo Strauss":          VCKG.PublicFigure,   # German-American philosopher
    "Jim Clark":            VCKG.Founder,        # co-founder of Netscape / Silicon Graphics
    "Owen Thomas":          VCKG.PublicFigure,   # tech journalist
    "The Thiel Foundation": VCKG.NonProfit,      # philanthropic foundation
    "the Thiel Fellowship": VCKG.NonProfit,      # fellowship programme
    "Sri Lanka Guardian":   VCKG.MediaOutlet,    # online news publication
}

# ── Forced owl:sameAs for entities the API finds correctly but scores too low ──
# The confidence scorer penalises these because of name mismatches (truncated NER
# text, accents, company renames). The QIDs are manually verified as correct.
FORCED_ALIGNMENTS: dict[str, str] = {
    "Twitter":           "Q918",    # renamed to X but same entity
    "Harvard":           "Q13371",  # "Harvard" vs "Harvard University"
    "MIT":               "Q49108",  # abbreviation vs full name
    "The New York Time": "Q9684",   # NER truncated the trailing 's'
    "Rene Girard":       "Q129228", # NER stripped the accent from René
    "Leo Strau":         "Q77144",  # NER truncated "Strauss" to "Strau"
    "Palantir":          "Q2047336", # "Palantir" vs "Palantir Technologies"
}

# ── Type hints for confidence scoring ─────────────────────────────────────────
TYPE_HINTS: dict[str, list[str]] = {
    "VC_FIRM":  ["venture capital", "investment firm", "private equity", "fund"],
    "ORG":      ["company", "corporation", "startup", "software", "technology",
                 "platform", "service", "business", "enterprise"],
    "PERSON":   ["investor", "entrepreneur", "businessman", "founder", "executive",
                 "computer scientist", "engineer", "philanthropist"],
    "SECTOR":   ["industry", "sector", "technology", "field"],
    "PRODUCT":  ["software", "application", "platform", "product", "service"],
}

# ── Predicate definitions ──────────────────────────────────────────────────────
# Each entry defines one vckg: predicate to align.
#
# keyword  → term for Wikidata property label search
# pairs    → (subj_qid, obj_qid) known aligned pairs for triple-based validation
# chosen_pid → manually chosen Wikidata property after reviewing SPARQL results
# relation → owl:equivalentProperty (identical) or skos:closeMatch (broader/narrower)
#
# Workflow: run the script → read the candidate output → encode your decision
# in chosen_pid/relation. The script will warn if your choice wasn't found by
# either search, flagging a potential misalignment.
PREDICATES = [
    {
        "local":       VCKG.investedIn,
        "keyword":     "participant of",
        "pairs":       [("Q1852025", "Q63327")],  # Sequoia → Airbnb
        # No dedicated "investedIn" exists in Wikidata; P1344 (participant of) is
        # the closest — a portfolio company "participated in" a round that Sequoia led.
        # Semantics differ slightly so marked closeMatch, not equivalentProperty.
        "chosen_pid":  "P1344",
        "relation":    SKOS.closeMatch,
    },
    {
        "local":       VCKG.foundedBy,
        "keyword":     "founded",
        "pairs":       [("Q210057", "Q62882")],   # Netscape → Marc Andreessen
        # Triple validation confirms P112 (founded by) on Netscape → Andreessen.
        "chosen_pid":  "P112",
        "relation":    OWL.equivalentProperty,
    },
    {
        "local":       VCKG.partnerAt,
        "keyword":     "employer",
        "pairs":       [("Q62882", "Q4034010")],  # Marc Andreessen → a16z
        # Triple validation finds P108 (employer). "partner" is narrower than
        # general employment so marked closeMatch, not equivalentProperty.
        "chosen_pid":  "P108",
        "relation":    SKOS.closeMatch,
    },
    {
        "local":       VCKG.ceoOf,
        "keyword":     "chief executive",
        "pairs":       [("Q36215", "Q355")],      # Mark Zuckerberg → Facebook
        # Label search surfaces P169 (chief executive officer of).
        "chosen_pid":  "P169",
        "relation":    OWL.equivalentProperty,
    },
    {
        "local":       VCKG.worksAt,
        "keyword":     "employer",
        "pairs":       [("Q317521", "Q2616400")], # Elon Musk → Y Combinator (no P108 link)
        # Label search surfaces P108 (employer); semantically equivalent.
        "chosen_pid":  "P108",
        "relation":    OWL.equivalentProperty,
    },
    {
        "local":       VCKG.operatesIn,
        "keyword":     "industry",
        "pairs":       [("Q2283", "Q16319025")],  # Microsoft → fintech
        # Label search surfaces P452 (industry); triple validation confirms it.
        "chosen_pid":  "P452",
        "relation":    OWL.equivalentProperty,
    },
    {
        "local":       VCKG.hasFundingRound,
        "keyword":     "funded",
        "pairs":       [],
        # No direct Wikidata equivalent. P8324 (funded by) is closest but
        # direction differs (company→investor, not company→round type).
        "chosen_pid":  "P8324",
        "relation":    SKOS.closeMatch,
    },
    {
        "local":       VCKG.acquiredBy,
        "keyword":     "acquisition",
        "pairs":       [],
        # P1642 (acquisition transaction) found by label search; best available match.
        "chosen_pid":  "P1642",
        "relation":    SKOS.closeMatch,
    },
    {
        "local":       VCKG.headquarteredIn,
        "keyword":     "headquarters",
        "pairs":       [],
        # P159 (headquarters location) is the match, but our range is a string
        # literal while P159 points to a QID — marked closeMatch.
        "chosen_pid":  "P159",
        "relation":    SKOS.closeMatch,
    },
    # Inverse / local-only — no Wikidata equivalent exists
    {"local": VCKG.fundedBy,  "keyword": None, "pairs": [], "chosen_pid": None, "relation": None},
    {"local": VCKG.founderOf, "keyword": None, "pairs": [], "chosen_pid": None, "relation": None},
    {"local": VCKG.hadExit,   "keyword": None, "pairs": [], "chosen_pid": None, "relation": None},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared SPARQL helper
# ═══════════════════════════════════════════════════════════════════════════════

def run_sparql(query: str) -> list[dict]:
    try:
        resp = requests.post(
            SPARQL_ENDPOINT,
            data={"query": query, "format": "json"},
            headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]
    except Exception as e:
        logger.warning(f"  SPARQL error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Entity linking
# ═══════════════════════════════════════════════════════════════════════════════

def search_wikidata(label: str, limit: int = 5) -> list[dict]:
    try:
        resp = requests.get(
            WIKIDATA_API,
            params={"action": "wbsearchentities", "search": label,
                    "language": "en", "format": "json", "limit": limit},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("search", [])
    except Exception as e:
        logger.warning(f"  API error for '{label}': {e}")
        return []


def compute_confidence(label: str, ner_label: str, hit: dict) -> float:
    wd_label = hit.get("label", "")
    wd_desc  = hit.get("description", "").lower()
    if wd_label == label:
        score = 1.0
    elif wd_label.lower() == label.lower():
        score = 0.95
    elif label.lower() in wd_label.lower():
        score = 0.80
    else:
        score = 0.60
    hints = TYPE_HINTS.get(ner_label, [])
    if hints and not any(h in wd_desc for h in hints):
        score -= 0.15
    return round(min(max(score, 0.0), 1.0), 3)


def _define_locally(g: Graph, uri: URIRef, class_uri, label: str, reason: str) -> None:
    """Semantically define an entity that has no Wikidata match.

    Uses LOCAL_CLASS_OVERRIDES for a specific subclass when available,
    falling back to the NER-derived class otherwise.
    The subClassOf hierarchy is declared in ontology.ttl and not repeated here
    to avoid writing a wrong parent derived from the NER label.
    """
    specific_class = LOCAL_CLASS_OVERRIDES.get(label)
    assigned_class = specific_class or class_uri
    if assigned_class:
        g.add((uri, RDF.type, assigned_class))
    g.add((uri, RDFS.label, Literal(label, lang="en")))
    g.add((uri, RDFS.comment, Literal(
        f"Locally defined entity: no Wikidata alignment ({reason}).", lang="en"
    )))


def link_entities(g: Graph, ner_file: Path) -> dict:
    with open(ner_file, encoding="utf-8") as f:
        ner_docs = [json.loads(line) for line in f]

    registry = build_entity_registry(ner_docs)
    registry = resolve_aliases(registry)

    # Deduplicate by URI — only link canonical (longest) text per URI
    seen_uris: set[str] = set()
    entities: dict[str, dict] = {}
    for text, info in sorted(registry.items(), key=lambda x: -len(x[0])):
        if info["class_uri"] is None:
            continue
        uri_str = str(info["uri"])
        if uri_str not in seen_uris:
            seen_uris.add(uri_str)
            entities[text] = info

    logger.info(f"\n── Entity Linking ({len(entities)} entities) ──")
    stats = {"same_as": 0, "close_match": 0, "not_found": 0}

    for text, info in entities.items():
        local_uri = info["uri"]
        ner_label = info["label"]

        # Check forced alignments before calling the API
        if text in FORCED_ALIGNMENTS:
            qid = FORCED_ALIGNMENTS[text]
            g.add((local_uri, OWL.sameAs, WD[qid]))
            g.add((local_uri, VCKG.alignmentConfidence, Literal(1.0, datatype=XSD.decimal)))
            g.add((local_uri, VCKG.wikidataId, Literal(qid, datatype=XSD.string)))
            stats["same_as"] += 1
            logger.info(f"  '{text}' → wd:{qid} (forced alignment, conf=1.0)")
            continue

        hits = search_wikidata(text)
        time.sleep(REQUEST_DELAY)

        if not hits:
            _define_locally(g, local_uri, info["class_uri"], text, "no Wikidata candidates returned")
            stats["not_found"] += 1
            logger.info(f"  '{text}' → not found, defined locally")
            continue

        top    = hits[0]
        conf   = compute_confidence(text, ner_label, top)
        wd_uri = WD[top["id"]]
        logger.info(f"  '{text}' → {top['id']} '{top.get('label','')}' conf={conf}")

        g.add((local_uri, VCKG.alignmentConfidence, Literal(conf, datatype=XSD.decimal)))
        g.add((local_uri, VCKG.wikidataId, Literal(top["id"], datatype=XSD.string)))

        if conf >= SAME_AS_THRESHOLD:
            g.add((local_uri, OWL.sameAs, wd_uri))
            stats["same_as"] += 1
        elif conf >= CLOSE_MATCH_THRESHOLD:
            g.add((local_uri, SKOS.closeMatch, wd_uri))
            stats["close_match"] += 1
        else:
            _define_locally(g, local_uri, info["class_uri"], text,
                            f"best Wikidata candidate {top['id']} "
                            f"'{top.get('label','')}' had low confidence ({conf})")
            stats["not_found"] += 1

    logger.info(f"  owl:sameAs={stats['same_as']} closeMatch={stats['close_match']} "
                f"notFound={stats['not_found']}")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Predicate alignment
# ═══════════════════════════════════════════════════════════════════════════════

def label_search(keyword: str) -> list[tuple[str, str]]:
    """Find Wikidata properties whose label contains keyword."""
    query = f"""
    SELECT ?property ?propertyLabel WHERE {{
      ?property a wikibase:Property .
      ?property rdfs:label ?propertyLabel .
      FILTER(CONTAINS(LCASE(?propertyLabel), "{keyword.lower()}"))
      FILTER(LANG(?propertyLabel) = "en")
    }}
    LIMIT 8
    """
    results = run_sparql(query)
    return [(r["property"]["value"].split("/")[-1], r["propertyLabel"]["value"])
            for r in results]


def triple_validation(subj_qid: str, obj_qid: str) -> list[tuple[str, str]]:
    """Find all Wikidata properties linking subj → obj.

    Uses wikibase:directClaim to bridge the direct claim namespace (wdt:P*)
    used in triples to the property entity namespace (wd:P*) where labels live.
    The naive pattern `wd:Qx ?p wd:Qy . ?p a wikibase:Property` fails because
    the triple binds ?p to wdt:P* while wikibase:Property expects wd:P*.
    """
    query = f"""
    SELECT DISTINCT ?property ?propertyLabel WHERE {{
      wd:{subj_qid} ?wdt wd:{obj_qid} .
      ?property wikibase:directClaim ?wdt .
      ?property rdfs:label ?propertyLabel .
      FILTER(LANG(?propertyLabel) = "en")
    }}
    """
    results = run_sparql(query)
    return [(r["property"]["value"].split("/")[-1], r["propertyLabel"]["value"])
            for r in results]


def align_predicates(g: Graph) -> None:
    logger.info(f"\n── Predicate Alignment ({len(PREDICATES)} predicates) ──")

    for entry in PREDICATES:
        local      = entry["local"]
        keyword    = entry["keyword"]
        pairs      = entry["pairs"]
        chosen_pid = entry["chosen_pid"]
        relation   = entry["relation"]
        name       = str(local).split("#")[-1]
        logger.info(f"  {name}")

        g.add((local, RDF.type, OWL.ObjectProperty))

        if not keyword:
            g.add((local, RDFS.comment, Literal(
                "Local-only property — no Wikidata equivalent exists.", lang="en"
            )))
            logger.info(f"    → local only")
            continue

        # Step 1: label-based candidate search
        label_hits = label_search(keyword)
        time.sleep(REQUEST_DELAY)
        label_pids = {pid for pid, _ in label_hits}
        logger.info(f"    label search '{keyword}' → {label_hits[:5]}")

        # Step 2: triple-based validation with known aligned entity pairs
        validated_pids: set[str] = set()
        for subj_qid, obj_qid in pairs:
            found = triple_validation(subj_qid, obj_qid)
            time.sleep(REQUEST_DELAY)
            logger.info(f"    triple validation wd:{subj_qid}→wd:{obj_qid}: {found}")
            validated_pids |= {pid for pid, _ in found}

        # Step 3: cross-check the manually chosen PID against SPARQL evidence
        all_candidates = label_pids | validated_pids
        if chosen_pid in validated_pids:
            support = "confirmed by triple validation"
        elif chosen_pid in label_pids:
            support = "confirmed by label search"
        elif chosen_pid:
            support = "WARNING — not found in label search or triple validation"
            logger.warning(f"    {support}: wdt:{chosen_pid} may be a misalignment")
        else:
            support = "no candidate chosen"

        # Step 4: write alignment triples
        if chosen_pid and relation:
            g.add((local, relation, WDT[chosen_pid]))
            g.add((local, RDFS.comment, Literal(
                f"Manually aligned to wdt:{chosen_pid} after reviewing SPARQL candidates "
                f"{sorted(all_candidates)}. Support: {support}.", lang="en"
            )))
            rel_name = str(relation).split("#")[-1]
            logger.info(f"    → wdt:{chosen_pid} ({rel_name}) [{support}]")
        else:
            g.add((local, RDFS.comment, Literal(
                "No Wikidata alignment. Defined locally.", lang="en"
            )))
            logger.info(f"    → local only")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def build_alignment(ner_file: Path, output_dir: Path) -> Graph:
    """
    Full alignment pipeline:
      1. Entity linking  — Search API per entity node → owl:sameAs / skos:closeMatch
      2. Predicate alignment — SPARQL per vckg: predicate → owl:equivalentProperty
    Both write to kg_artifacts/alignment.ttl.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    g = Graph()
    g.bind("vckg",   VCKG)
    g.bind("wd",     WD)
    g.bind("wdt",    WDT)
    g.bind("owl",    OWL)
    g.bind("skos",   SKOS)
    g.bind("rdfs",   RDFS)
    g.bind("schema", SCHEMA)

    logger.info("=== Alignment: Entities + Predicates ===")

    link_entities(g, ner_file)
    align_predicates(g)

    out_path = output_dir / "alignment.ttl"
    g.serialize(destination=str(out_path), format="turtle")
    logger.info(f"\nTotal triples : {len(g)}")
    logger.info(f"Saved to      : {out_path}")
    return g


if __name__ == "__main__":
    build_alignment(
        ner_file=Path("data/ner/ner_results.jsonl"),
        output_dir=Path("kg_artifacts"),
    )
