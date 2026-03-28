"""
Clustering Analysis — t-SNE + Ontology Class Coloring

Steps:
  1. Extract entity embeddings from the trained RotatE model
  2. Classify each entity by ontology class using:
       - VCKG initial_graph.ttl  (authoritative for core entities)
       - P31 (instance of) triples in train.txt (Wikidata entities)
       - Relational heuristics for Persons (P106/P69/P108 subjects)
  3. Apply t-SNE to reduce embeddings to 2D
  4. Plot scatter coloured by class

Output:
  kg_artifacts/kge/analysis/
    tsne_all.png          full scatter (all 11K entities, small dots)
    tsne_labeled.png      zoom on core VCKG entities with name labels
    entity_classes.json   entity → class mapping
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from rdflib import Graph, RDF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
MODEL_PATH   = ROOT / "kg_artifacts" / "kge" / "results" / "RotatE" / "trained_model.pkl"
TRIPLES_DIR  = ROOT / "kg_artifacts" / "kge" / "results" / "RotatE" / "training_triples"
TRAIN_TXT    = ROOT / "kg_artifacts" / "kge" / "train.txt"
INIT_GRAPH   = ROOT / "kg_artifacts" / "initial_graph.ttl"
LABELS_JSON  = ROOT / "kg_artifacts" / "kge" / "analysis" / "nearest_neighbors.json"
OUT_DIR      = ROOT / "kg_artifacts" / "kge" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Class definitions and colours
# ---------------------------------------------------------------------------
CLASSES: dict[str, str] = {
    "VC Firm":      "#e63946",   # red
    "Person":       "#457b9d",   # blue
    "Company":      "#2a9d8f",   # teal
    "Organization": "#e9c46a",   # yellow
    "Tech/Product": "#f4a261",   # orange
    "Other":        "#adb5bd",   # grey
}

# P31 QID → macro class
P31_CLASS_MAP: dict[str, str] = {
    # Company / business
    "Q4830453":  "Company",    # business
    "Q783794":   "Company",    # company
    "Q6881511":  "Company",    # enterprise
    "Q891723":   "Company",    # public company
    "Q210167":   "Company",    # video game developer
    "Q1156831":  "Company",    # investment company
    # Organization
    "Q43229":    "Organization",   # organization
    "Q163740":   "Organization",   # nonprofit
    "Q23002054": "Organization",   # private educational institution
    "Q31855":    "Organization",   # research institute
    "Q38723":    "Organization",   # higher education institution
    "Q3918":     "Organization",   # university
    "Q1076486":  "Organization",   # sports venue / broader org
    # Tech / product
    "Q7397":     "Tech/Product",   # software
    "Q35127":    "Tech/Product",   # website
    "Q1668024":  "Tech/Product",   # internet service
    "Q9143":     "Tech/Product",   # programming language
}

# Predicates that are strong signals for Person subjects
PERSON_PRED_SIGNALS = {
    "http://www.wikidata.org/prop/direct/P106",   # occupation
    "http://www.wikidata.org/prop/direct/P69",    # educated at
    "http://www.wikidata.org/prop/direct/P19",    # place of birth
    "http://www.wikidata.org/prop/direct/P22",    # father
    "http://www.wikidata.org/prop/direct/P25",    # mother
    "http://www.wikidata.org/prop/direct/P26",    # spouse
    "http://www.wikidata.org/prop/direct/P40",    # child
    "http://www.wikidata.org/prop/direct/P3373",  # sibling
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def load_vckg_classes() -> dict[str, str]:
    """Read rdf:type from initial_graph.ttl → uri: class_label."""
    g = Graph()
    g.parse(str(INIT_GRAPH))
    vckg_ns = "http://vckg.org/ontology#"
    mapping: dict[str, str] = {}
    for s, _, o in g.triples((None, RDF.type, None)):
        cls = str(o).replace(vckg_ns, "")
        entity_uri = str(s)
        if cls == "VCFirm":
            mapping[entity_uri] = "VC Firm"
        elif cls == "Person":
            mapping[entity_uri] = "Person"
        elif cls in ("Company", "Organization"):
            mapping[entity_uri] = cls
    return mapping


def build_entity_classes(
    id_to_uri: dict[int, str],
    uri_to_id: dict[str, int],
) -> dict[int, str]:
    """
    Assign each entity an ontology class.
    Priority: VCKG class > P31 mapping > person heuristic > Other
    """
    # 1. VCKG ontology classes (core entities, highest confidence)
    vckg = load_vckg_classes()
    # Map VCKG URIs (vckg.org) to their Wikidata equivalents via owl:sameAs
    # already in alignment — but simpler: match by VCKG URI directly
    classes: dict[int, str] = {}
    for uri, cls in vckg.items():
        if uri in uri_to_id:
            classes[uri_to_id[uri]] = cls

    # 2. P31 (instance of) triples from training data
    p31_pred = "http://www.wikidata.org/prop/direct/P31"
    p31_map: dict[str, str] = {}   # entity_uri → class
    with TRAIN_TXT.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            s, p, o = parts
            if p == p31_pred:
                qid = o.split("/")[-1]
                if qid in P31_CLASS_MAP:
                    # Only set if not already classified by VCKG
                    if s not in vckg:
                        p31_map[s] = P31_CLASS_MAP[qid]

    for uri, cls in p31_map.items():
        if uri in uri_to_id:
            eid = uri_to_id[uri]
            if eid not in classes:
                classes[eid] = cls

    # 3. Person heuristic — subjects of person-specific predicates
    person_subjects: set[str] = set()
    with TRAIN_TXT.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            s, pred, _ = parts
            if pred in PERSON_PRED_SIGNALS:
                person_subjects.add(s)

    for uri in person_subjects:
        if uri in uri_to_id:
            eid = uri_to_id[uri]
            if eid not in classes:
                classes[eid] = "Person"

    # 4. Default
    for eid in id_to_uri:
        if eid not in classes:
            classes[eid] = "Other"

    return classes


def get_entity_embeddings(model) -> torch.Tensor:
    rep = model.entity_representations[0]
    return rep._embeddings.weight.data.detach()


def uri_to_short_label(uri: str, wd_labels: dict[str, str]) -> str:
    qid = uri.split("/")[-1]
    if uri in wd_labels:
        return wd_labels[uri]
    if "vckg.org" in uri:
        return uri.split("#")[-1]
    return qid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading model and entity map ...")
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    id_to_uri, uri_to_id = load_entity_map()

    print("Building entity class assignments ...")
    entity_classes = build_entity_classes(id_to_uri, uri_to_id)

    class_counts = {}
    for cls in entity_classes.values():
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print("  Class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<20} {cnt:>5,}")

    # Save class mapping
    class_json = {id_to_uri[eid]: cls for eid, cls in entity_classes.items()}
    with (OUT_DIR / "entity_classes.json").open("w") as f:
        json.dump(class_json, f, indent=2)

    print("Extracting embeddings ...")
    embeddings = get_entity_embeddings(model)
    emb_np = F.normalize(embeddings, dim=1).numpy()

    num_entities = len(id_to_uri)

    # -------------------------------------------------------------------
    # t-SNE (full 11K entities)
    # -------------------------------------------------------------------
    print("Running t-SNE (this takes ~2–3 min) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
        n_jobs=-1,
    )
    coords = tsne.fit_transform(emb_np)   # (N, 2)
    print("  t-SNE done.")

    # -------------------------------------------------------------------
    # Plot 1 — full scatter (all entities)
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))
    for cls, colour in CLASSES.items():
        mask = [i for i in range(num_entities) if entity_classes[i] == cls]
        if not mask:
            continue
        xs = [coords[i, 0] for i in mask]
        ys = [coords[i, 1] for i in mask]
        ax.scatter(xs, ys, c=colour, s=4, alpha=0.5, linewidths=0, label=cls)

    ax.set_title("Entity Embeddings — t-SNE (RotatE, VC Knowledge Graph)", fontsize=13)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="Ontology Class", markerscale=4, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out1 = OUT_DIR / "tsne_all.png"
    fig.savefig(str(out1), dpi=150)
    plt.close(fig)
    print(f"  Saved {out1.name}")

    # -------------------------------------------------------------------
    # Plot 2 — labeled zoom on core VCKG entities + key Wikidata anchors
    # -------------------------------------------------------------------
    # Load Wikidata labels — anchors + their neighbors from nearest_neighbors.json
    wd_labels: dict[str, str] = {}
    # Hard-code anchor names (they are not in nearest_neighbors.json as entries)
    ANCHOR_NAMES = {
        "http://www.wikidata.org/entity/Q1852025": "Sequoia Capital",
        "http://www.wikidata.org/entity/Q4034010": "Andreessen Horowitz",
        "http://www.wikidata.org/entity/Q2616400": "Y Combinator",
        "http://www.wikidata.org/entity/Q705525":  "Peter Thiel",
        "http://www.wikidata.org/entity/Q62882":   "Marc Andreessen",
        "http://www.wikidata.org/entity/Q7407093": "Sam Altman",
        "http://www.wikidata.org/entity/Q95":      "Google",
    }
    wd_labels.update(ANCHOR_NAMES)
    if LABELS_JSON.exists():
        with LABELS_JSON.open() as f:
            nn_data = json.load(f)
        for _, neighbors in nn_data.items():
            for n in neighbors:
                lbl = n["label"]
                # Skip raw QIDs stored as labels (e.g. "Q2616400") — they are
                # unresolved placeholders and would overwrite correct names
                if lbl.startswith("Q") and lbl[1:].isdigit():
                    continue
                wd_labels[n["uri"]] = lbl

    # Entities to label: VCKG core + key anchors
    ANCHOR_URIS = {
        "http://www.wikidata.org/entity/Q1852025",  # Sequoia
        "http://www.wikidata.org/entity/Q4034010",  # a16z
        "http://www.wikidata.org/entity/Q2616400",  # YC
        "http://www.wikidata.org/entity/Q705525",   # Peter Thiel
        "http://www.wikidata.org/entity/Q62882",    # Marc Andreessen
        "http://www.wikidata.org/entity/Q7407093",  # Sam Altman
        "http://www.wikidata.org/entity/Q95",       # Google
    }
    vckg_uris  = {uri for uri in uri_to_id if "vckg.org" in uri}
    label_uris = vckg_uris | ANCHOR_URIS

    # Fetch labels for any remaining unlabeled Wikidata URIs in label_uris
    import requests, time
    missing_qids = [
        uri.split("/")[-1]
        for uri in label_uris
        if "wikidata.org" in uri and uri not in wd_labels
    ]
    if missing_qids:
        print(f"  Fetching {len(missing_qids)} missing labels from Wikidata ...")
        for i in range(0, len(missing_qids), 50):
            batch = missing_qids[i:i+50]
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
                        wd_labels[f"http://www.wikidata.org/entity/{qid}"] = lbl
            except Exception:
                pass
            time.sleep(0.3)

    fig, ax = plt.subplots(figsize=(16, 12))
    for cls, colour in CLASSES.items():
        mask = [i for i in range(num_entities) if entity_classes[i] == cls]
        if not mask:
            continue
        xs = [coords[i, 0] for i in mask]
        ys = [coords[i, 1] for i in mask]
        ax.scatter(xs, ys, c=colour, s=5, alpha=0.35, linewidths=0)

    # Overlay labeled points — collect for adjustText
    texts = []
    for uri in label_uris:
        if uri not in uri_to_id:
            continue
        eid   = uri_to_id[uri]
        cls   = entity_classes[eid]
        x, y  = coords[eid, 0], coords[eid, 1]
        label = uri_to_short_label(uri, wd_labels)
        ax.scatter(x, y, c=CLASSES[cls], s=60, zorder=5, edgecolors="white", linewidths=0.5)
        texts.append(ax.text(
            x, y, label,
            fontsize=6.5, color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
        ))

    # Push labels apart so none overlap
    adjust_text(
        texts,
        ax=ax,
        expand=(1.3, 1.5),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5),
        force_text=(0.4, 0.6),
        force_points=(0.2, 0.3),
    )

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in CLASSES.items()]
    ax.legend(handles=patches, title="Ontology Class", fontsize=9, loc="best")
    ax.set_title("t-SNE — Core VCKG Entities Labeled (RotatE embeddings)", fontsize=13)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out2 = OUT_DIR / "tsne_labeled.png"
    fig.savefig(str(out2), dpi=150)
    plt.close(fig)
    print(f"  Saved {out2.name}")

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
