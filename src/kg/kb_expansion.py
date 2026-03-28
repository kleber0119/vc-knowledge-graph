"""
Step 4 — KB Expansion via SPARQL

Strategy:
  1. 1-hop outgoing from each aligned entity (KEEP_PREDICATES filter, LIMIT 500)
  2a. Outgoing anchored expansion: for entities discovered in Phase 1, fetch their
      outgoing claims via key predicates (VALUES ?s batch)
  2b. Incoming expansion: for our known orgs, fetch entities that POINT TO them
      e.g. "who works at Google?" (P108 incoming), "which companies did Sequoia fund?" (P1344 incoming)
      This is the main source of new entities (employees, portfolio companies, founders).
  3. 1-hop from all newly discovered entities (up to MAX_NEW_ENTITIES)

Predicate strategy:
  Uses KEEP_PREDICATES allowlist (~60 domain-relevant predicates) to stay within
  the 50–200 relation target. Denylist approach produces 900+ predicates (too many).

Target: 50,000 – 200,000 triples, 5,000 – 30,000 entities, 50 – 200 relations.
"""

import logging
import time
from pathlib import Path
from collections import defaultdict

import requests
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VCKG = Namespace("http://vckg.org/ontology#")
WD   = Namespace("http://www.wikidata.org/entity/")
WDT  = Namespace("http://www.wikidata.org/prop/direct/")

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "KGResearchBot/1.0 (academic KG project; non-commercial)",
    "Accept":     "application/sparql-results+json",
}
REQUEST_DELAY  = 0.3
VALUES_BATCH   = 100      # URIs per VALUES clause
MAX_NEW_ENTITIES = 8000   # cap on new entities expanded in Phase 3

# ── Allowlist of ~60 domain-relevant wdt:P* predicates ────────────────────────
# Covers: org lifecycle, leadership, employment, investment/ownership,
# location, business metrics, person biography, awards.
# ~60 here + ~15 base-graph predicates (rdf:type, rdfs:label, vckg:*, owl:sameAs)
# = ~75 unique predicates total — within the 50–200 target.
KEEP_PREDICATES = {
    # Organisation lifecycle & classification
    "P571",   # inception / founded
    "P576",   # dissolved
    "P31",    # instance of
    "P279",   # subclass of
    # Leadership & governance
    "P112",   # founded by
    "P169",   # CEO
    "P1037",  # director / manager
    "P488",   # chairperson
    "P3320",  # board member
    "P6",     # head of government
    # Employment & membership
    "P108",   # employer
    "P463",   # member of
    "P39",    # position held
    "P106",   # occupation
    # Investment, ownership & corporate structure
    "P1344",  # participant of  (investments / funding rounds)
    "P8324",  # funded by
    "P749",   # parent organisation
    "P355",   # subsidiary
    "P127",   # owned by
    "P1830",  # owner of
    "P199",   # business division
    "P361",   # part of
    "P527",   # has part
    "P1365",  # replaces
    "P1366",  # replaced by
    # Location
    "P159",   # headquarters location
    "P17",    # country
    "P131",   # located in administrative entity
    "P150",   # contains administrative division
    "P276",   # location
    # Business & finance
    "P452",   # industry
    "P414",   # stock exchange
    "P856",   # official website
    "P1128",  # employees
    "P2139",  # total revenue
    "P2226",  # market capitalisation
    "P2295",  # net income
    "P2541",  # operating area
    "P1454",  # legal form
    "P1056",  # product / material produced
    # Person — biography
    "P69",    # educated at (alma mater)
    "P27",    # country of citizenship
    "P19",    # place of birth
    "P569",   # date of birth
    "P570",   # date of death
    "P21",    # sex or gender
    "P26",    # spouse
    "P40",    # child
    "P22",    # father
    "P25",    # mother
    "P3373",  # sibling
    "P101",   # field of work
    "P1412",  # languages spoken
    # Awards & events
    "P166",   # award received
    "P7",     # significant event
    "P737",   # influenced by
}

# ── High-value entities for 2-hop expansion ────────────────────────────────────
TWO_HOP_QIDS = [
    "Q1852025",   # Sequoia Capital
    "Q4034010",   # Andreessen Horowitz
    "Q2616400",   # Y Combinator
    "Q705525",    # Peter Thiel
    "Q62882",     # Marc Andreessen
    "Q7407093",   # Sam Altman
]

# Predicates used for incoming expansion + their per-batch limits
INCOMING_PREDICATES = [
    ("P108",  1000),  # employer  → find employees of our companies
    ("P1344", 2000),  # participant of → find portfolio companies of our VCs
    ("P112",  1000),  # founded by → find companies founded by people we know
    ("P355",  500),   # subsidiary → find subsidiaries of our orgs
    ("P127",  500),   # owned by → find what our orgs own
]

# Wikidata P31 (instance of) values that indicate an entity worth expanding in Phase 3.
# Leaf nodes like countries (Q6256), cities (Q515), awards (Q618779) are excluded.
ORG_PERSON_TYPES: set[str] = {
    "http://www.wikidata.org/entity/Q5",         # human
    "http://www.wikidata.org/entity/Q215627",    # person
    "http://www.wikidata.org/entity/Q4830453",   # business
    "http://www.wikidata.org/entity/Q783794",    # company
    "http://www.wikidata.org/entity/Q891723",    # public company
    "http://www.wikidata.org/entity/Q6881511",   # enterprise
    "http://www.wikidata.org/entity/Q43229",     # organization
    "http://www.wikidata.org/entity/Q484652",    # financial institution
    "http://www.wikidata.org/entity/Q22687",     # bank
    "http://www.wikidata.org/entity/Q740752",    # investment fund
    "http://www.wikidata.org/entity/Q1589009",   # startup company
    "http://www.wikidata.org/entity/Q3661699",   # investment company
    "http://www.wikidata.org/entity/Q206170",    # venture capital company
    "http://www.wikidata.org/entity/Q15913373",  # unicorn company
    "http://www.wikidata.org/entity/Q163740",    # nonprofit organisation
    "http://www.wikidata.org/entity/Q3918",      # university
    "http://www.wikidata.org/entity/Q902104",    # private company
    "http://www.wikidata.org/entity/Q431289",    # enterprise
    "http://www.wikidata.org/entity/Q167396",    # private equity firm
    "http://www.wikidata.org/entity/Q270791",    # sovereign wealth fund
    "http://www.wikidata.org/entity/Q658255",    # subsidiary
    "http://www.wikidata.org/entity/Q1616075",   # holding company
    "http://www.wikidata.org/entity/Q4167836",   # Wikimedia category (skip — caught by absence)
}

_P31_URI = URIRef("http://www.wikidata.org/prop/direct/P31")

def is_org_or_person(g: Graph, qid: str) -> bool:
    """Return True if the entity's P31 (instance of) maps to an org or person type."""
    subj = URIRef(f"http://www.wikidata.org/entity/{qid}")
    return any(str(o) in ORG_PERSON_TYPES for _, _, o in g.triples((subj, _P31_URI, None)))


def run_sparql(query: str, timeout: int = 60) -> list[dict]:
    try:
        resp = requests.post(
            SPARQL_ENDPOINT,
            data={"query": query, "format": "json"},
            headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]
    except Exception as e:
        logger.warning(f"  SPARQL error: {e}")
        return []


LABEL_PREDICATES = {
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://schema.org/name",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
}

def is_keep_predicate(pred_uri: str) -> bool:
    if pred_uri in LABEL_PREDICATES:
        return True
    if "wikidata.org/prop/direct/P" not in pred_uri:
        return False
    pid = pred_uri.split("/")[-1]
    return pid in KEEP_PREDICATES


def make_literal(b_obj: dict) -> Literal | None:
    val   = b_obj.get("value", "")
    lang  = b_obj.get("xml:lang")
    dtype = b_obj.get("datatype")
    if lang:
        return Literal(val, lang=lang)
    if dtype:
        return Literal(val, datatype=URIRef(dtype))
    return Literal(val)


def add_triples(g: Graph, triples: list[tuple]) -> int:
    added = 0
    for t in triples:
        if t not in g:
            g.add(t)
            added += 1
    return added


def one_hop(g: Graph, qid: str, limit: int = 500) -> int:
    """Fetch direct claims + English rdfs:label for a Wikidata entity."""
    query = f"""
    SELECT ?p ?o WHERE {{
      {{
        wd:{qid} ?p ?o .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
      }}
      UNION
      {{
        wd:{qid} rdfs:label ?o .
        BIND(rdfs:label AS ?p)
        FILTER(LANG(?o) = "en")
      }}
    }}
    LIMIT {limit}
    """
    bindings = run_sparql(query)
    subj_uri = f"http://www.wikidata.org/entity/{qid}"
    triples = []
    for b in bindings:
        p_val  = b.get("p", {}).get("value", "")
        o_b    = b.get("o", {})
        o_val  = o_b.get("value", "")
        o_type = o_b.get("type", "")
        if not p_val or not o_val or not is_keep_predicate(p_val):
            continue
        subj = URIRef(subj_uri)
        pred = URIRef(p_val)
        obj  = URIRef(o_val) if o_type == "uri" else make_literal(o_b)
        if obj is not None:
            triples.append((subj, pred, obj))
    return add_triples(g, triples)


def get_wikidata_uris(g: Graph) -> list[str]:
    """All wd:Q* URIs currently in the graph (subjects + objects)."""
    uris = set()
    for s, _, o in g:
        for node in (s, o):
            if isinstance(node, URIRef) and "wikidata.org/entity/Q" in str(node):
                uris.add(str(node))
    return list(uris)


def outgoing_anchored(g: Graph, pid: str, known_uris: list[str],
                      limit_per_batch: int = 500) -> int:
    """
    For entities already in the graph, fetch their outgoing claims via pid.
    VALUES ?s { batch } ?s wdt:pid ?o
    """
    pred_uri = URIRef(f"http://www.wikidata.org/prop/direct/{pid}")
    added = 0
    for i in range(0, len(known_uris), VALUES_BATCH):
        batch = known_uris[i : i + VALUES_BATCH]
        values_str = " ".join(f"<{u}>" for u in batch)
        query = f"""
        SELECT ?s ?o WHERE {{
          VALUES ?s {{ {values_str} }}
          ?s wdt:{pid} ?o .
        }}
        LIMIT {limit_per_batch}
        """
        bindings = run_sparql(query)
        for b in bindings:
            s_val  = b.get("s", {}).get("value", "")
            o_b    = b.get("o", {})
            o_val  = o_b.get("value", "")
            o_type = o_b.get("type", "")
            if not s_val or not o_val:
                continue
            subj = URIRef(s_val)
            obj  = URIRef(o_val) if o_type == "uri" else make_literal(o_b)
            if obj is not None and (subj, pred_uri, obj) not in g:
                g.add((subj, pred_uri, obj))
                added += 1
        time.sleep(REQUEST_DELAY)
    return added


def incoming_expansion(g: Graph, pid: str, org_uris: list[str],
                       limit_per_batch: int) -> tuple[int, set[str]]:
    """
    Discover new entities that point TO our known organisations via pid.
    VALUES ?o { batch } ?s wdt:pid ?o
    Returns (triples_added, new_entity_uris).
    """
    pred_uri    = URIRef(f"http://www.wikidata.org/prop/direct/{pid}")
    new_entities: set[str] = set()
    added = 0
    for i in range(0, len(org_uris), VALUES_BATCH):
        batch = org_uris[i : i + VALUES_BATCH]
        values_str = " ".join(f"<{u}>" for u in batch)
        query = f"""
        SELECT ?s ?o WHERE {{
          VALUES ?o {{ {values_str} }}
          ?s wdt:{pid} ?o .
        }}
        LIMIT {limit_per_batch}
        """
        bindings = run_sparql(query)
        for b in bindings:
            s_val = b.get("s", {}).get("value", "")
            o_val = b.get("o", {}).get("value", "")
            if not s_val or not o_val:
                continue
            subj = URIRef(s_val)
            obj  = URIRef(o_val)
            triple = (subj, pred_uri, obj)
            if triple not in g:
                g.add(triple)
                added += 1
            if "wikidata.org/entity/Q" in s_val:
                new_entities.add(s_val)
        time.sleep(REQUEST_DELAY)
    return added, new_entities


def expand_kb(
    initial_graph_path: Path,
    alignment_path: Path,
    output_dir: Path,
) -> Graph:

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base graph ────────────────────────────────────────────────────────
    g = Graph()
    g.bind("vckg", VCKG)
    g.bind("wd",   WD)
    g.bind("wdt",  WDT)
    g.parse(str(initial_graph_path), format="turtle")
    g.parse(str(alignment_path),     format="turtle")
    logger.info(f"Base graph loaded: {len(g)} triples")

    # ── Collect aligned QIDs ───────────────────────────────────────────────────
    aligned: list[tuple[URIRef, str]] = []
    for subj, _, obj in g.triples((None, OWL.sameAs, None)):
        obj_str = str(obj)
        if "wikidata.org/entity/Q" in obj_str:
            qid = obj_str.split("/")[-1]
            aligned.append((subj, qid))
    logger.info(f"Aligned entities: {len(aligned)}")

    out_path = output_dir / "expanded.nt"
    last_save: list[int] = [len(g)]   # mutable so inner function can update it

    def autosave() -> None:
        """Save to disk whenever 500+ new triples have accumulated."""
        if len(g) - last_save[0] >= 500:
            g.serialize(destination=str(out_path), format="ntriples")
            last_save[0] = len(g)
            logger.info(f"  [saved] {len(g):,} triples → {out_path}")

    # ── Phase 0: Deep expansion for high-value anchor entities ───────────────
    # These 6 entities (Sequoia, a16z, YC, Thiel, Andreessen, Altman) are the
    # densest nodes in the VC domain — fetch up to 5000 triples each.
    logger.info("\n── Phase 0: Deep expansion for high-value anchors ──")
    phase0_total = 0
    for qid in TWO_HOP_QIDS:
        n = one_hop(g, qid, limit=5000)
        phase0_total += n
        logger.info(f"  wd:{qid} → +{n} triples (total: {len(g)})")
        time.sleep(REQUEST_DELAY)
        autosave()
    logger.info(f"Phase 0 done: +{phase0_total} triples → {len(g)} total")

    # ── Phase 1: 1-hop outgoing from aligned entities ─────────────────────────
    logger.info("\n── Phase 1: 1-hop from aligned entities ──")
    phase1_total = 0
    for _, qid in aligned:
        n = one_hop(g, qid, limit=1000)
        phase1_total += n
        logger.info(f"  wd:{qid} → +{n} triples (total: {len(g)})")
        time.sleep(REQUEST_DELAY)
        autosave()
    logger.info(f"Phase 1 done: +{phase1_total} triples → {len(g)} total")

    # ── Phase 2a: Outgoing anchored expansion ─────────────────────────────────
    logger.info("\n── Phase 2a: Outgoing anchored expansion ──")
    known_uris = get_wikidata_uris(g)
    logger.info(f"  Anchor set: {len(known_uris)} Wikidata URIs")
    phase2a_total = 0
    for pid in KEEP_PREDICATES:
        n = outgoing_anchored(g, pid, known_uris, limit_per_batch=1000)
        if n:
            phase2a_total += n
            logger.info(f"  P{pid} → +{n} triples (total: {len(g)})")
        autosave()
    logger.info(f"Phase 2a done: +{phase2a_total} triples → {len(g)} total")

    # ── Phase 2b: Incoming expansion ──────────────────────────────────────────
    # Use ALL Wikidata entities in the graph (not just the 56 sameAs ones) so we
    # discover entities that point to anything we already know about.
    logger.info("\n── Phase 2b: Incoming expansion (new entities → our orgs) ──")
    org_uris = get_wikidata_uris(g)
    logger.info(f"  Org/person anchor set: {len(org_uris)} URIs")

    all_new_entities: set[str] = set()
    phase2b_total = 0
    for pid, limit in INCOMING_PREDICATES:
        logger.info(f"  Incoming P{pid} (limit {limit}/batch)...")
        n, new = incoming_expansion(g, pid, org_uris, limit_per_batch=limit)
        phase2b_total += n
        all_new_entities |= new
        logger.info(f"  → +{n} triples, {len(new)} new entities (total: {len(g)})")
        autosave()
    logger.info(f"Phase 2b done: +{phase2b_total} triples, "
                f"{len(all_new_entities)} new entities → {len(g)} total")

    # ── Phase 3: Smart 1-hop from newly discovered org/person entities ──────────
    logger.info("\n── Phase 3: Smart 1-hop from new org/person entities ──")
    phase1_qids = {qid for _, qid in aligned}
    candidate_qids = [
        uri.split("/")[-1]
        for uri in all_new_entities
        if uri.split("/")[-1] not in phase1_qids
    ]

    # Expand all discovered entities up to cap — no P31 filter.
    # The previous inclusion filter (must be org/person type) was cutting out
    # most candidates because Wikidata uses many specific subtypes not in the list.
    smart_qids = candidate_qids[:MAX_NEW_ENTITIES]
    logger.info(f"  Candidates: {len(candidate_qids)}, expanding: {len(smart_qids)} "
                f"(cap {MAX_NEW_ENTITIES})")

    phase3_total = 0
    for qid in smart_qids:
        n = one_hop(g, qid, limit=500)
        phase3_total += n
        time.sleep(REQUEST_DELAY)
        autosave()
    logger.info(f"Phase 3 done: +{phase3_total} triples → {len(g)} total")

    # Final save
    g.serialize(destination=str(out_path), format="ntriples")
    logger.info(f"\nSaved to {out_path}")

    # ── Final KB statistics ────────────────────────────────────────────────────
    logger.info("\n=== Final KB Statistics ===")
    logger.info(f"Total triples    : {len(g):,}")

    entities: set[str] = set()
    predicates: set[str] = set()
    for s, p, o in g:
        if isinstance(s, URIRef):
            entities.add(str(s))
        if isinstance(o, URIRef):
            entities.add(str(o))
        predicates.add(str(p))
    logger.info(f"Unique entities  : {len(entities):,}")
    logger.info(f"Unique predicates: {len(predicates):,}")

    pred_counts: dict[str, int] = defaultdict(int)
    for _, p, _ in g:
        pred_counts[str(p).split("/")[-1]] += 1
    logger.info("\nTop 20 predicates:")
    for pred, cnt in sorted(pred_counts.items(), key=lambda x: -x[1])[:20]:
        logger.info(f"  {pred:35s}: {cnt:,}")

    return g


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    expand_kb(
        initial_graph_path=Path("kg_artifacts/initial_graph.ttl"),
        alignment_path=Path("kg_artifacts/alignment.ttl"),
        output_dir=Path("kg_artifacts"),
    )
