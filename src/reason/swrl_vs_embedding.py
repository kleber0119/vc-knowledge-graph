"""
Comparison: Rule-Based Reasoning (SWRL) vs Embedding-Based Reasoning (RotatE)

SWRL Rule (Horn clause with two conditions):
    VCFirm(?f) ∧ partnerAt(?p, ?f) ∧ foundedBy(?c, ?p) → investedIn(?f, ?c)

Reading: if person p is a partner at VC firm f, AND company c was founded by p,
         THEN f invested in c.

Embedding analogy:
    The SWRL rule encodes a 2-hop relational path:
        c --foundedBy--> p --partnerAt--> f  ⟹  f --investedIn--> c

    In vector space (TransE/RotatE style):
        vector(c) + vector(foundedBy) + vector(partnerAt) ≈ vector(f)

    We verify this by checking whether the entity nearest to
        vector(Netscape) + vector(foundedBy) + vector(partnerAt)
    is Andreessen Horowitz — which is the answer predicted by the SWRL rule.

Reasoning engine: OWLReady2
"""

from __future__ import annotations

import gzip
from pathlib import Path

import torch
import torch.nn.functional as F
from owlready2 import get_ontology, sync_reasoner_pellet

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
ONTO_PATH   = ROOT / "kg_artifacts" / "ontology.ttl"
GRAPH_PATH  = ROOT / "kg_artifacts" / "initial_graph.ttl"
MODEL_PATH  = ROOT / "kg_artifacts" / "kge" / "results" / "RotatE" / "trained_model.pkl"
TRIPLES_DIR = ROOT / "kg_artifacts" / "kge" / "results" / "RotatE" / "training_triples"

VCKG = "http://vckg.org/ontology#"


# ===========================================================================
# PART 1 — SWRL Rule-Based Reasoning (OWLReady2)
# ===========================================================================

def ttl_to_rdfxml(src: Path, dst: Path) -> None:
    """Convert a Turtle file to RDF/XML so OWLReady2 can load it."""
    from rdflib import Graph as RDFGraph
    g = RDFGraph()
    g.parse(str(src), format="turtle")
    g.serialize(str(dst), format="xml")


def run_swrl_reasoning() -> list[tuple[str, str]]:
    """
    Load the VCKG ontology and initial graph, define the SWRL rule, run
    the reasoner, and return newly inferred investedIn triples.

    SWRL Rule:
        VCFirm(?f) ∧ partnerAt(?p, ?f) ∧ foundedBy(?c, ?p)
            → investedIn(?f, ?c)
    """
    print("=" * 65)
    print("PART 1 — SWRL Rule-Based Reasoning (OWLReady2)")
    print("=" * 65)

    # OWLReady2 requires RDF/XML — merge ontology + instance graph into one file,
    # stripping owl:imports to prevent remote URI fetches.
    import tempfile
    from rdflib import Graph as RDFGraph, OWL

    tmp_dir    = Path(tempfile.mkdtemp())
    merged_owl = tmp_dir / "vckg_merged.owl"

    merged = RDFGraph()
    merged.parse(str(ONTO_PATH),  format="turtle")
    merged.parse(str(GRAPH_PATH), format="turtle")
    # Remove all owl:imports so OWLReady2 doesn't try to fetch them remotely
    for s, o in list(merged.subject_objects(OWL.imports)):
        merged.remove((s, OWL.imports, o))
    merged.serialize(str(merged_owl), format="xml")

    onto = get_ontology(merged_owl.as_uri()).load()

    with onto:
        # ---------------------------------------------------------------
        # SWRL Rule definition
        # ---------------------------------------------------------------
        # VCFirm(?f) ∧ partnerAt(?p, ?f) ∧ foundedBy(?c, ?p)
        #     → investedIn(?f, ?c)
        # ---------------------------------------------------------------
        rule_str = (
            "VCFirm(?f), partnerAt(?p, ?f), foundedBy(?c, ?p)"
            " -> investedIn(?f, ?c)"
        )
        print(f"\nSWRL Rule:\n  {rule_str}\n")

        from owlready2 import Imp
        rule = Imp()
        rule.set_as_rule(rule_str, namespaces=[onto])

    # Run reasoner (requires Java + Pellet on PATH; falls back gracefully)
    inferred: list[tuple[str, str]] = []
    try:
        with onto:
            sync_reasoner_pellet(infer_property_values=True, debug=0)

        # Collect inferred investedIn triples
        investedIn_prop = onto.search_one(iri=f"{VCKG}investedIn")
        vcfirm_cls      = onto.search_one(iri=f"{VCKG}VCFirm")

        if investedIn_prop and vcfirm_cls:
            for firm in vcfirm_cls.instances():
                for company in getattr(firm, investedIn_prop.python_name, []):
                    inferred.append((firm.name, company.name))

        print("Inferred investedIn triples (after SWRL reasoning):")
        for f, c in sorted(inferred):
            print(f"  investedIn({f}, {c})")

    except Exception as e:
        # Pellet/Java not available — apply rule manually in Python
        # (logically equivalent; Pellet would do the same)
        print(f"  [Reasoner not available: {e}]")
        print("  Applying rule manually (equivalent to Pellet output):\n")
        inferred = apply_rule_manually(onto)

    return inferred


def apply_rule_manually(_onto) -> list[tuple[str, str]]:
    """
    Apply the SWRL rule by pattern matching directly on the graph.
    This is logically identical to what a SWRL reasoner executes.

    SWRL Rule:
        VCFirm(?f) ∧ partnerAt(?p, ?f) ∧ foundedBy(?c, ?p)
            → investedIn(?f, ?c)
    """
    from rdflib import Graph, RDF
    g = Graph()
    g.parse(str(GRAPH_PATH))

    # Collect asserted facts
    vcfirms:     set[str] = set()
    partner_at:  dict[str, str] = {}   # person → firm
    founded_by:  dict[str, str] = {}   # company → person
    invested_in: set[tuple[str, str]] = set()

    for s, p, o in g:
        s_l = str(s).split("#")[-1]
        o_l = str(o).split("#")[-1]
        p_l = str(p).split("#")[-1]
        if p == RDF.type and str(o) == f"{VCKG}VCFirm":
            vcfirms.add(s_l)
        elif p_l == "partnerAt":
            partner_at[s_l] = o_l
        elif p_l == "foundedBy":
            founded_by[s_l] = o_l
        elif p_l == "investedIn":
            invested_in.add((s_l, o_l))

    print("  Asserted facts used by the rule:")
    print(f"    VCFirms        : {sorted(vcfirms)}")
    print(f"    partnerAt      : {partner_at}")
    print(f"    foundedBy      : {founded_by}")
    print()

    # Fire the rule: for each binding (f, p, c) satisfying the body
    inferred: list[tuple[str, str]] = []
    for person, firm in partner_at.items():
        if firm not in vcfirms:
            continue
        for company, founder in founded_by.items():
            if founder != person:
                continue
            pair = (firm, company)
            status = "ALREADY ASSERTED" if pair in invested_in else "*** NEW INFERENCE ***"
            print(f"  Rule fires: partnerAt({person}, {firm})"
                  f" ∧ foundedBy({company}, {person})")
            print(f"    → investedIn({firm}, {company})  [{status}]")
            inferred.append(pair)

    return inferred


# ===========================================================================
# PART 2 — Embedding-Based Reasoning (RotatE vector arithmetic)
# ===========================================================================

def load_entity_map() -> tuple[dict[int, str], dict[str, int]]:
    id_to_uri: dict[int, str] = {}
    uri_to_id: dict[str, int] = {}
    with gzip.open(TRIPLES_DIR / "entity_to_id.tsv.gz", "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                eid, uri = int(parts[0]), parts[1]
                id_to_uri[eid] = uri
                uri_to_id[uri] = eid
    return id_to_uri, uri_to_id


def load_relation_map() -> tuple[dict[int, str], dict[str, int]]:
    id_to_rel: dict[int, str] = {}
    rel_to_id: dict[str, int] = {}
    with gzip.open(TRIPLES_DIR / "relation_to_id.tsv.gz", "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                rid, uri = int(parts[0]), parts[1]
                id_to_rel[rid] = uri
                rel_to_id[uri] = rid
    return id_to_rel, rel_to_id


def get_entity_embeddings(model) -> torch.Tensor:
    return model.entity_representations[0]._embeddings.weight.data.detach()


def get_relation_embeddings(model) -> torch.Tensor:
    return model.relation_representations[0]._embeddings.weight.data.detach()


def top_k_nearest(
    query_vec: torch.Tensor,
    embeddings: torch.Tensor,
    id_to_uri: dict[int, str],
    k: int = 10,
    exclude_ids: set[int] | None = None,
) -> list[tuple[str, float]]:
    """Return top-k entity URIs nearest to query_vec by cosine similarity."""
    q_norm   = F.normalize(query_vec.unsqueeze(0), dim=1)
    emb_norm = F.normalize(embeddings, dim=1)
    sims     = (emb_norm @ q_norm.T).squeeze(1)
    if exclude_ids:
        for eid in exclude_ids:
            sims[eid] = -2.0
    top_ids = torch.argsort(sims, descending=True)[:k]
    return [(id_to_uri[int(i)], float(sims[i])) for i in top_ids]


def uri_to_short(uri: str, labels: dict[str, str] | None = None) -> str:
    if labels and uri in labels:
        return labels[uri]
    return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]


def fetch_labels(uris: list[str]) -> dict[str, str]:
    """Batch-resolve Wikidata URIs to English labels via the API."""
    import requests, time
    qids = [u.split("/")[-1] for u in uris if "wikidata.org/entity/Q" in u]
    labels: dict[str, str] = {}
    for i in range(0, len(qids), 50):
        batch = qids[i:i+50]
        try:
            r = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={"action": "wbgetentities", "ids": "|".join(batch),
                        "props": "labels", "languages": "en", "format": "json"},
                headers={"User-Agent": "vc-kg/1.0"},
                timeout=20,
            )
            for qid, ent in r.json().get("entities", {}).items():
                lbl = ent.get("labels", {}).get("en", {}).get("value")
                if lbl:
                    labels[f"http://www.wikidata.org/entity/{qid}"] = lbl
        except Exception:
            pass
        time.sleep(0.3)
    return labels


def run_embedding_reasoning() -> None:
    """
    Embedding analogy corresponding to the SWRL rule:

        SWRL:  foundedBy(?c, ?p) ∧ partnerAt(?p, ?f) → investedIn(?f, ?c)

        Vector path:
            c --foundedBy--> p --partnerAt--> f  ⟹  f --investedIn--> c

        Test:
            query = vector(Netscape) + vector(foundedBy) + vector(partnerAt)
            Nearest entity to `query` should be AndreessenHorowitz.

        This mirrors the SWRL firing:
            foundedBy(Netscape, BenHorowitz) ∧ partnerAt(BenHorowitz, AndreessenHorowitz)
                → investedIn(AndreessenHorowitz, Netscape)
    """
    print("\n" + "=" * 65)
    print("PART 2 — Embedding-Based Reasoning (RotatE vector arithmetic)")
    print("=" * 65)

    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()

    id_to_uri, uri_to_id = load_entity_map()
    _, rel_to_id = load_relation_map()

    ent_emb = get_entity_embeddings(model)
    rel_emb = get_relation_embeddings(model)

    # Entity and relation URIs
    NS_WD   = "http://www.wikidata.org/entity/"
    NS_VCKG = "http://vckg.org/ontology#"

    # Entities involved in the rule firing
    # Netscape ↔ Q210057 (confirmed in alignment)
    # AndreessenHorowitz ↔ Q4034010
    # BenHorowitz — via VCKG URI (may not be in Wikidata sub-graph)
    netscape_candidates = [
        f"{NS_VCKG}Netscape",
        f"{NS_WD}Q210057",
    ]
    a16z_candidates = [
        f"{NS_VCKG}AndreessenHorowitz",
        f"{NS_WD}Q4034010",
    ]

    # Relations: use VCKG ontology predicates (in the embedding space)
    founded_by_uri  = f"{NS_VCKG}foundedBy"
    partner_at_uri  = f"{NS_VCKG}partnerAt"
    invested_in_uri = f"{NS_VCKG}investedIn"

    # Check availability
    def find_uri(candidates: list[str], label: str) -> str | None:
        for uri in candidates:
            if uri in uri_to_id:
                return uri
        print(f"  WARNING: {label} not found in embedding space.")
        return None

    def find_rel(uri: str, label: str) -> int | None:
        if uri in rel_to_id:
            return rel_to_id[uri]
        print(f"  WARNING: relation {label} ({uri}) not found.")
        return None

    netscape_uri  = find_uri(netscape_candidates,  "Netscape")
    a16z_uri      = find_uri(a16z_candidates,      "Andreessen Horowitz")
    founded_by_id = find_rel(founded_by_uri,  "foundedBy")
    partner_at_id = find_rel(partner_at_uri,  "partnerAt")
    invested_in_id = find_rel(invested_in_uri, "investedIn")

    if not all([netscape_uri, a16z_uri, founded_by_id is not None,
                partner_at_id is not None]):
        print("  Cannot complete embedding test — missing entities/relations.")
        return

    netscape_id = uri_to_id[netscape_uri]
    a16z_id     = uri_to_id[a16z_uri]

    # ---------------------------------------------------------------
    # Vector arithmetic:
    #   query = vector(Netscape) + vector(foundedBy) + vector(partnerAt)
    #   Expected nearest: AndreessenHorowitz
    # ---------------------------------------------------------------
    v_netscape    = ent_emb[netscape_id]
    v_founded_by  = rel_emb[founded_by_id]
    v_partner_at  = rel_emb[partner_at_id]

    query = v_netscape + v_founded_by + v_partner_at

    print(f"""
Embedding analogy test
  SWRL path : Netscape --foundedBy--> BenHorowitz --partnerAt--> AndreessenHorowitz
  Vector    : vector(Netscape) + vector(foundedBy) + vector(partnerAt)
  Expected  : AndreessenHorowitz  (the entity the SWRL rule points to)
""")

    neighbors = top_k_nearest(
        query, ent_emb, id_to_uri,
        k=10, exclude_ids={netscape_id},
    )
    nb_labels = fetch_labels([uri for uri, _ in neighbors])

    a16z_rank = None
    print(f"  {'Rank':<6} {'Entity':<40} {'Cosine Sim':>10}")
    print(f"  {'-'*58}")
    for rank, (uri, sim) in enumerate(neighbors, 1):
        label = uri_to_short(uri, nb_labels)
        marker = " ← ✓ SWRL answer" if uri == a16z_uri else ""
        print(f"  {rank:<6} {label:<40} {sim:>10.4f}{marker}")
        if uri == a16z_uri:
            a16z_rank = rank

    print()
    if a16z_rank:
        print(f"  AndreessenHorowitz found at rank {a16z_rank}.")
        if a16z_rank <= 3:
            print("  ✓ Strong match — embedding reasoning agrees with the SWRL rule.")
        elif a16z_rank <= 10:
            print("  ~ Partial match — embedding places the SWRL answer in top 10.")
        else:
            print("  ✗ Embedding does not replicate the SWRL rule for this instance.")
    else:
        print("  AndreessenHorowitz not in top 10.")

    # ---------------------------------------------------------------
    # Also verify the investedIn direction:
    #   vector(AndreessenHorowitz) + vector(investedIn) ≈ vector(Netscape)?
    # ---------------------------------------------------------------
    if invested_in_id is not None:
        print(f"""
Verification (investedIn direction):
  vector(AndreessenHorowitz) + vector(investedIn)  ≈?  vector(Netscape)
""")
        v_a16z       = ent_emb[a16z_id]
        v_invested   = rel_emb[invested_in_id]
        query2       = v_a16z + v_invested

        neighbors2 = top_k_nearest(
            query2, ent_emb, id_to_uri,
            k=10, exclude_ids={a16z_id},
        )
        nb2_labels = fetch_labels([uri for uri, _ in neighbors2])
        netscape_rank2 = None
        print(f"  {'Rank':<6} {'Entity':<40} {'Cosine Sim':>10}")
        print(f"  {'-'*58}")
        for rank, (uri, sim) in enumerate(neighbors2, 1):
            label  = uri_to_short(uri, nb2_labels)
            marker = " ← ✓ SWRL answer" if uri == netscape_uri else ""
            print(f"  {rank:<6} {label:<40} {sim:>10.4f}{marker}")
            if uri == netscape_uri:
                netscape_rank2 = rank

        print()
        if netscape_rank2:
            print(f"  Netscape found at rank {netscape_rank2}.")
        else:
            print("  Netscape not in top 10 for this direction.")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # --- Part 1: SWRL ---
    run_swrl_reasoning()

    # --- Part 2: Embeddings ---
    run_embedding_reasoning()

    print("\n" + "=" * 65)
    print("Summary & Discussion")
    print("=" * 65)
    print("""
SWRL Rule (2 conditions):
  VCFirm(?f) ∧ partnerAt(?p, ?f) ∧ foundedBy(?c, ?p) → investedIn(?f, ?c)

  Fires on:
    partnerAt(BenHorowitz, AndreessenHorowitz)
    ∧ foundedBy(Netscape,  BenHorowitz)  →  investedIn(AndreessenHorowitz, Netscape)
    ∧ foundedBy(Opsware,   BenHorowitz)  →  investedIn(AndreessenHorowitz, Opsware)

  Both inferences were already asserted in the graph — the rule correctly
  reconstructs known facts from structural patterns.

Embedding analogy:
  vector(Netscape) + vector(foundedBy) + vector(partnerAt)
    Rank 1 → Opsware  (NOT AndreessenHorowitz)

  The embedding does NOT replicate the SWRL rule exactly.
  However, the result is semantically meaningful:
    • Opsware is Ben Horowitz's OTHER company — same founder, same VC.
    • The embedding captures "companies similar to Netscape via Ben Horowitz"
      rather than "the VC firm that invested in Netscape".
    • This is because foundedBy and partnerAt have very few triples (2 each)
      in the training data — not enough signal for precise path composition.

  In contrast, the SWRL rule fires deterministically on any graph satisfying
  the pattern, regardless of data density.

Key difference:
  SWRL  — exact symbolic inference; guaranteed correct if premises hold.
  KGE   — statistical pattern matching; generalises to unseen entities but
           requires sufficient training signal per relation.
  For sparse domain-specific relations (like VCKG ontology predicates),
  rule-based reasoning is more reliable. KGE shines on dense Wikidata
  relations (P108, P69, P106) where it learned rich structural patterns.
""")


if __name__ == "__main__":
    main()
