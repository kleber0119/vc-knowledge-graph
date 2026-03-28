"""
KGE Preprocessing: Clean expanded.nt for embedding training.

Steps:
  1. Remove duplicate triplets
  2. Remove inconsistent URIs (malformed, non-HTTP, blank nodes)
  3. Remove literal-heavy predicates (dates, labels, numeric values, etc.)
  4. Remove hub-generating predicates (P21 sex/gender, P31 instance-of)
     and schema/meta relations (owl#, rdf-syntax#, core#)
  5. Prune low-degree entities: drop triples where either entity has
     degree < MIN_ENTITY_DEGREE across the full triple set
  6. Cap over-represented relations at MAX_TRIPLES_PER_RELATION
  7. Ensure all entities and relations are uniquely indexed
  8. 80/10/10 split with cold-start guarantee
  9. Write train.txt, valid.txt, test.txt + index files
"""

from __future__ import annotations

import re
import csv
import random
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
INPUT_NT   = ROOT / "kg_artifacts" / "expanded.nt"
OUTPUT_DIR = ROOT / "kg_artifacts" / "kge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TSV     = OUTPUT_DIR / "train.tsv"
ENTITIES_TSV  = OUTPUT_DIR / "entities.tsv"
RELATIONS_TSV = OUTPUT_DIR / "relations.tsv"
STATS_TXT     = OUTPUT_DIR / "preprocessing_stats.txt"
TRAIN_TXT     = OUTPUT_DIR / "train.txt"
VALID_TXT     = OUTPUT_DIR / "valid.txt"
TEST_TXT      = OUTPUT_DIR / "test.txt"

SPLIT_SEED             = 42
VALID_RATIO            = 0.10
TEST_RATIO             = 0.10

# Fix 1 — min degree threshold: drop any triple whose head or tail
# appears fewer than this many times across the full dataset.
# Kept at 2 to preserve enough triples (target: 50K–200K) while
# still removing pure singletons that contribute no learning signal.
MIN_ENTITY_DEGREE      = 2

# Fix 2 — relation cap: disabled. With only 3 relations exceeding 8K
# triples (P1344, P106, P108), capping causes more damage than benefit
# by dropping total triples below the 50K minimum requirement.
# The imbalance is noted but acceptable given the dataset size.
MAX_TRIPLES_PER_RELATION = None   # None = no cap

# ---------------------------------------------------------------------------
# Fix 3a — Literal-heavy predicates (no structural KGE signal)
# ---------------------------------------------------------------------------
LITERAL_PREDICATES: set[str] = {
    # Dates / times
    "http://www.wikidata.org/prop/direct/P569",
    "http://www.wikidata.org/prop/direct/P570",
    "http://www.wikidata.org/prop/direct/P571",
    "http://www.wikidata.org/prop/direct/P576",
    "http://www.wikidata.org/prop/direct/P580",
    "http://www.wikidata.org/prop/direct/P582",
    "http://www.wikidata.org/prop/direct/P585",
    # Numeric / financial
    "http://www.wikidata.org/prop/direct/P2226",
    "http://www.wikidata.org/prop/direct/P1082",
    "http://www.wikidata.org/prop/direct/P2139",
    "http://www.wikidata.org/prop/direct/P2295",
    "http://www.wikidata.org/prop/direct/P1128",
    # Labels / URLs / images
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://schema.org/description",
    "http://www.wikidata.org/prop/direct/P856",
    "http://www.wikidata.org/prop/direct/P18",
    "http://www.wikidata.org/prop/direct/P154",
    # Language strings
    "http://www.wikidata.org/prop/direct/P1412",
}

# ---------------------------------------------------------------------------
# Fix 3b — Hub object filtering: rather than dropping entire predicates,
# drop only the specific (predicate, object) pairs that create extreme hubs.
#
# P21 (sex/gender) is dropped entirely — its only objects are male/female,
# so it carries zero structural signal.
# P31 (instance of) is kept but only for non-human objects — organisational
# typing (company, university, VC firm…) is useful; "instance of human" is not.
# ---------------------------------------------------------------------------
HUB_PREDICATES: set[str] = {
    "http://www.wikidata.org/prop/direct/P21",   # sex or gender — always male/female
}

# Specific (predicate, object) pairs to drop even when the predicate is kept
HUB_OBJECTS: dict[str, set[str]] = {
    "http://www.wikidata.org/prop/direct/P31": {
        "http://www.wikidata.org/entity/Q5",        # human
        "http://www.wikidata.org/entity/Q6581097",  # male (redundant, also in P21)
        "http://www.wikidata.org/entity/Q6581072",  # female
    },
}

# ---------------------------------------------------------------------------
# Fix 3c — Schema / meta relations: RDF alignment and ontology machinery,
# not factual knowledge suitable for KGE training.
# ---------------------------------------------------------------------------
SCHEMA_PRED_FRAGMENTS: tuple[str, ...] = (
    "22-rdf-syntax-ns#type",
    "owl#sameAs",
    "owl#equivalentProperty",
    "owl#imports",
)

# Predicates exempt from degree pruning — always kept regardless of entity degree.
# Domain-specific ontology relations are the most semantically valuable in this
# graph and must not be silently dropped just because their entities are sparse.
PROTECTED_PRED_FRAGMENTS: tuple[str, ...] = (
    "ontology#",
)

# ---------------------------------------------------------------------------
# URI validation
# ---------------------------------------------------------------------------
URI_RE = re.compile(r'^<(https?://[^\s<>]+)>$')


def extract_uri(token: str) -> str | None:
    m = URI_RE.match(token)
    return m.group(1) if m else None


def is_schema_pred(pred: str) -> bool:
    return any(frag in pred for frag in SCHEMA_PRED_FRAGMENTS)


# ---------------------------------------------------------------------------
# Train / Validation / Test split
# ---------------------------------------------------------------------------

def split_triples(
    triples: list[tuple[str, str, str]],
    valid_ratio: float = VALID_RATIO,
    test_ratio: float  = TEST_RATIO,
    seed: int          = SPLIT_SEED,
) -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
]:
    """
    80/10/10 split with cold-start guarantee:
    any triple in valid/test whose head, tail, or relation is unseen
    in train is rescued back into train until the split is stable.
    """
    rng = random.Random(seed)
    shuffled = triples[:]
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_test  = int(n * test_ratio)
    n_valid = int(n * valid_ratio)
    n_train = n - n_valid - n_test

    train = shuffled[:n_train]
    valid = shuffled[n_train: n_train + n_valid]
    test  = shuffled[n_train + n_valid:]

    for _ in range(10):
        train_ents = {s for s, _, _ in train} | {o for _, _, o in train}
        train_rels = {p for _, p, _ in train}

        rescued    = []
        kept_valid = []
        for s, p, o in valid:
            if s in train_ents and o in train_ents and p in train_rels:
                kept_valid.append((s, p, o))
            else:
                rescued.append((s, p, o))

        kept_test = []
        for s, p, o in test:
            if s in train_ents and o in train_ents and p in train_rels:
                kept_test.append((s, p, o))
            else:
                rescued.append((s, p, o))

        if not rescued:
            break

        train = train + rescued
        valid = kept_valid
        test  = kept_test

    return train, valid, test


def write_split(path: Path, triples: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Reading {INPUT_NT} ...")

    raw_count          = 0
    blank_node_skip    = 0
    literal_pred_skip  = 0
    hub_pred_skip      = 0
    schema_pred_skip   = 0
    literal_obj_skip   = 0
    malformed_skip     = 0
    duplicate_skip     = 0

    seen: set[tuple[str, str, str]] = set()
    triples: list[tuple[str, str, str]] = []

    with INPUT_NT.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw_count += 1
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 3)
            if len(parts) < 3:
                malformed_skip += 1
                continue

            subj_tok, pred_tok, obj_tok = parts[0], parts[1], parts[2]

            subj = extract_uri(subj_tok)
            if subj is None:
                blank_node_skip += 1
                continue

            pred = extract_uri(pred_tok)
            if pred is None:
                malformed_skip += 1
                continue

            # --- Predicate filters ---
            if pred in LITERAL_PREDICATES:
                literal_pred_skip += 1
                continue
            if pred in HUB_PREDICATES:
                hub_pred_skip += 1
                continue
            if is_schema_pred(pred):
                schema_pred_skip += 1
                continue

            obj = extract_uri(obj_tok)
            if obj is None:
                literal_obj_skip += 1
                continue

            # --- Hub object filter (surgical: blocks specific pred+obj pairs) ---
            if pred in HUB_OBJECTS and obj in HUB_OBJECTS[pred]:
                hub_pred_skip += 1
                continue

            triple = (subj, pred, obj)
            if triple in seen:
                duplicate_skip += 1
                continue

            seen.add(triple)
            triples.append(triple)

    after_filter = len(triples)
    print(f"After predicate filtering: {after_filter:,} triples")

    # -------------------------------------------------------------------
    # Fix 1 — Prune low-degree entities (iterative: pruning changes degrees)
    # -------------------------------------------------------------------
    def is_protected(pred: str) -> bool:
        return any(frag in pred for frag in PROTECTED_PRED_FRAGMENTS)

    print(f"Pruning entities with degree < {MIN_ENTITY_DEGREE} ...")
    for _ in range(20):
        degree: Counter = Counter()
        for s, p, o in triples:
            degree[s] += 1
            degree[o] += 1
        keep_ents = {e for e, d in degree.items() if d >= MIN_ENTITY_DEGREE}
        pruned = [(s, p, o) for s, p, o in triples
                  if (s in keep_ents and o in keep_ents) or is_protected(p)]
        if len(pruned) == len(triples):
            break   # stable
        triples = pruned

    after_pruning = len(triples)
    degree_final: Counter = Counter()
    for s, p, o in triples:
        degree_final[s] += 1
        degree_final[o] += 1
    print(f"After degree pruning: {after_pruning:,} triples  "
          f"{len(degree_final):,} entities")

    # -------------------------------------------------------------------
    # Fix 2 — Cap over-represented relations (skipped if None)
    # -------------------------------------------------------------------
    rel_caps: dict[str, int] = {}
    if MAX_TRIPLES_PER_RELATION is not None:
        print(f"Capping relations at {MAX_TRIPLES_PER_RELATION:,} triples each ...")
        rng_cap = random.Random(SPLIT_SEED)
        by_rel: defaultdict[str, list] = defaultdict(list)
        for triple in triples:
            by_rel[triple[1]].append(triple)

        triples = []
        for rel, rel_triples in by_rel.items():
            if len(rel_triples) > MAX_TRIPLES_PER_RELATION:
                sampled = rng_cap.sample(rel_triples, MAX_TRIPLES_PER_RELATION)
                rel_caps[rel.split("/")[-1]] = len(rel_triples)
                triples.extend(sampled)
            else:
                triples.extend(rel_triples)

        if rel_caps:
            print(f"  Capped relations (original → {MAX_TRIPLES_PER_RELATION:,}):")
            for r, orig in sorted(rel_caps.items(), key=lambda x: -x[1]):
                print(f"    {r:<12} {orig:>6,} → {MAX_TRIPLES_PER_RELATION:,}")

    after_cap = len(triples)

    # -------------------------------------------------------------------
    # Build unique indexes
    # -------------------------------------------------------------------
    entity_set: set[str] = set()
    relation_set: set[str] = set()
    for s, p, o in triples:
        entity_set.add(s)
        entity_set.add(o)
        relation_set.add(p)

    entity2id   = {e: i for i, e in enumerate(sorted(entity_set))}
    relation2id = {r: i for i, r in enumerate(sorted(relation_set))}

    # -------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------
    print("Splitting triples (80/10/10) ...")
    train_triples, valid_triples, test_triples = split_triples(triples)

    write_split(TRAIN_TXT, train_triples)
    write_split(VALID_TXT, valid_triples)
    write_split(TEST_TXT,  test_triples)

    # -------------------------------------------------------------------
    # Index files
    # -------------------------------------------------------------------
    with TRAIN_TSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for s, p, o in triples:
            writer.writerow([entity2id[s], relation2id[p], entity2id[o]])

    with ENTITIES_TSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "uri"])
        for uri, idx in sorted(entity2id.items(), key=lambda x: x[1]):
            writer.writerow([idx, uri])

    with RELATIONS_TSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "uri"])
        for uri, idx in sorted(relation2id.items(), key=lambda x: x[1]):
            writer.writerow([idx, uri])

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------
    clean_count = len(triples)
    train_ents = {s for s, _, _ in train_triples} | {o for _, _, o in train_triples}
    train_rels = {p for _, p, _ in train_triples}
    valid_oov  = sum(1 for s, p, o in valid_triples
                     if s not in train_ents or o not in train_ents or p not in train_rels)
    test_oov   = sum(1 for s, p, o in test_triples
                     if s not in train_ents or o not in train_ents or p not in train_rels)

    degrees = sorted(degree_final.values())
    n_ents  = len(degrees)
    singletons_pct = sum(1 for d in degrees if d == 1) / n_ents * 100

    stats_lines = [
        "=== KGE Preprocessing Statistics ===",
        f"Raw lines read               : {raw_count:>10,}",
        "",
        "-- Removed (predicate filters) --",
        f"  Blank nodes / non-URI      : {blank_node_skip:>10,}",
        f"  Literal predicates         : {literal_pred_skip:>10,}",
        f"  Hub predicates (P21, P31)  : {hub_pred_skip:>10,}",
        f"  Schema/meta predicates     : {schema_pred_skip:>10,}",
        f"  Literal objects            : {literal_obj_skip:>10,}",
        f"  Malformed triples          : {malformed_skip:>10,}",
        f"  Duplicates                 : {duplicate_skip:>10,}",
        f"  After predicate filtering  : {after_filter:>10,}",
        "",
        "-- Removed (structural fixes) --",
        f"  Low-degree entity pruning  : {after_filter - after_pruning:>10,}",
        f"  Relation cap               : {after_pruning - after_cap:>10,}",
        "",
        "-- Retained --",
        f"  Clean triples              : {clean_count:>10,}",
        f"  Unique entities            : {len(entity2id):>10,}",
        f"  Unique relations           : {len(relation2id):>10,}",
        f"  Median entity degree       : {degrees[n_ents//2]:>10}",
        f"  Avg entity degree          : {sum(degrees)/n_ents:>10.1f}",
        f"  Singleton entities         : {singletons_pct:>9.1f}%  (target: <5%)",
        "",
        "-- Split (80/10/10) --",
        f"  train.txt                  : {len(train_triples):>10,}  ({len(train_triples)/clean_count*100:.1f}%)",
        f"  valid.txt                  : {len(valid_triples):>10,}  ({len(valid_triples)/clean_count*100:.1f}%)",
        f"  test.txt                   : {len(test_triples):>10,}  ({len(test_triples)/clean_count*100:.1f}%)",
        f"  OOV triples in valid       : {valid_oov:>10,}  (must be 0)",
        f"  OOV triples in test        : {test_oov:>10,}  (must be 0)",
        f"  Random seed                : {SPLIT_SEED}",
        "",
        "-- Output files --",
        f"  {TRAIN_TXT}",
        f"  {VALID_TXT}",
        f"  {TEST_TXT}",
        f"  {TRAIN_TSV}",
        f"  {ENTITIES_TSV}",
        f"  {RELATIONS_TSV}",
        f"  {STATS_TXT}",
        "",
        "-- Relation distribution in train (all) --",
    ]

    pred_counter = Counter(p for _, p, _ in train_triples)
    for pred_uri, cnt in pred_counter.most_common():
        pred_short = pred_uri.split("/")[-1]
        stats_lines.append(f"  {pred_short:<30} {cnt:>8,}")

    report = "\n".join(stats_lines)
    print("\n" + report)

    with STATS_TXT.open("w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\nDone. Outputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
