"""
Embedding Analysis — Section 6.1: Nearest Neighbors

Loads the best trained model (RotatE, full dataset) and for each
selected VC entity retrieves the K nearest neighbors in embedding
space using cosine similarity on the entity embedding matrix.

Output:
  kg_artifacts/kge/analysis/
    nearest_neighbors.json    structured results
    nearest_neighbors.txt     human-readable report
    nn_heatmap.png            cosine similarity heatmap between anchor entities
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "kg_artifacts" / "kge" / "results" / "RotatE" / "trained_model.pkl"
TRIPLES_DIR = ROOT / "kg_artifacts" / "kge" / "results" / "RotatE" / "training_triples"
OUT_DIR    = ROOT / "kg_artifacts" / "kge" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K          = 10   # number of nearest neighbors to retrieve
LABEL_PRED = "http://www.w3.org/2000/01/rdf-schema#label"
NT_PATH    = ROOT / "kg_artifacts" / "expanded.nt"

# ---------------------------------------------------------------------------
# Anchor entities — key VC players to analyse
# ---------------------------------------------------------------------------
ANCHORS: dict[str, str] = {
    "Sequoia Capital":     "http://www.wikidata.org/entity/Q1852025",
    "Andreessen Horowitz": "http://www.wikidata.org/entity/Q4034010",
    "Y Combinator":        "http://www.wikidata.org/entity/Q2616400",
    "Peter Thiel":         "http://www.wikidata.org/entity/Q705525",
    "Marc Andreessen":     "http://www.wikidata.org/entity/Q62882",
    "Sam Altman":          "http://www.wikidata.org/entity/Q7407093",
    "Google":              "http://www.wikidata.org/entity/Q95",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_entity_map() -> tuple[dict[int, str], dict[str, int]]:
    id_to_uri: dict[int, str] = {}
    uri_to_id: dict[str, int] = {}
    with gzip.open(TRIPLES_DIR / "entity_to_id.tsv.gz", "rt") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                eid, uri = int(parts[0]), parts[1]
                id_to_uri[eid] = uri
                uri_to_id[uri] = eid
    return id_to_uri, uri_to_id


def fetch_wikidata_labels(qids: set[str]) -> dict[str, str]:
    """Batch-resolve Wikidata QIDs to English labels via the Wikidata API."""
    import time
    import requests

    resolved: dict[str, str] = {}
    qid_list = list(qids)
    headers  = {"User-Agent": "vc-kg-analysis/1.0 (research project)"}

    for i in range(0, len(qid_list), 50):
        batch = qid_list[i : i + 50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": "en",
            "format": "json",
        }
        try:
            r = requests.get(
                "https://www.wikidata.org/w/api.php",
                params=params,
                headers=headers,
                timeout=20,
            )
            if r.status_code == 200 and r.text:
                for qid, entity in r.json().get("entities", {}).items():
                    label = entity.get("labels", {}).get("en", {}).get("value")
                    if label:
                        resolved[qid] = label
        except Exception:
            pass
        time.sleep(0.3)

    return resolved


def load_labels() -> dict[str, str]:
    """
    Parse rdfs:label triples from expanded.nt and return uri -> English label.
    Only keeps @en language-tagged literals.
    """
    labels: dict[str, str] = {}
    label_pred = f"<{LABEL_PRED}>"
    with NT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if label_pred not in line:
                continue
            parts = line.strip().split(None, 3)
            if len(parts) < 3:
                continue
            subj_tok, pred_tok, obj_tok = parts[0], parts[1], parts[2]
            if pred_tok != label_pred:
                continue
            # Extract URI
            if not (subj_tok.startswith("<") and subj_tok.endswith(">")):
                continue
            uri = subj_tok[1:-1]
            # Extract @en literal: "Some Name"@en
            if obj_tok.endswith("@en") or obj_tok.endswith("@en ."):
                raw = obj_tok.split('"')
                if len(raw) >= 2:
                    labels[uri] = raw[1]
    return labels


def uri_to_label(uri: str, labels: dict[str, str] | None = None) -> str:
    """Return a readable label: English label if available, else URI suffix."""
    if labels and uri in labels:
        return labels[uri]
    if "vckg.org" in uri:
        return uri.split("#")[-1]
    return uri.split("/")[-1]


def get_entity_embeddings(model) -> torch.Tensor:
    """Extract the entity embedding matrix (num_entities x dim)."""
    rep = model.entity_representations[0]
    # RotatE uses complex embeddings stored as real+imag concatenated
    emb = rep._embeddings.weight.data   # (num_entities, embedding_dim)
    return emb


def nearest_neighbors(
    anchor_id: int,
    embeddings: torch.Tensor,
    k: int = K,
) -> list[tuple[int, float]]:
    """
    Return top-k nearest neighbors by cosine similarity.
    Excludes the anchor itself.
    """
    anchor_vec = embeddings[anchor_id].unsqueeze(0)             # (1, dim)
    # Normalize
    anchor_norm = F.normalize(anchor_vec, dim=1)
    emb_norm    = F.normalize(embeddings, dim=1)
    # Cosine similarity to all entities
    sims = (emb_norm @ anchor_norm.T).squeeze(1)                # (num_entities,)
    sims[anchor_id] = -2.0  # exclude self
    top_ids = torch.argsort(sims, descending=True)[:k]
    return [(int(idx), float(sims[idx])) for idx in top_ids]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading RotatE model ...")
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()

    print("Loading entity map ...")
    id_to_uri, uri_to_id = load_entity_map()

    print("Loading labels from expanded.nt ...")
    labels = load_labels()
    print(f"  Loaded {len(labels):,} English labels")

    print("Extracting entity embeddings ...")
    embeddings = get_entity_embeddings(model)
    print(f"  Embedding matrix: {embeddings.shape}")

    # Resolve any QIDs not covered by rdfs:label triples via Wikidata API
    all_neighbor_uris = {
        id_to_uri[nid]
        for anchor_uri in ANCHORS.values()
        if anchor_uri in uri_to_id
        for nid, _ in nearest_neighbors(uri_to_id[anchor_uri], embeddings, k=K)
    }
    unresolved_qids = {
        uri.split("/")[-1]
        for uri in all_neighbor_uris
        if uri not in labels and "wikidata.org/entity/Q" in uri
    }
    if unresolved_qids:
        print(f"Resolving {len(unresolved_qids)} QIDs via Wikidata API ...")
        wd_labels = fetch_wikidata_labels(unresolved_qids)
        # Map back to full URI
        for qid, lbl in wd_labels.items():
            labels[f"http://www.wikidata.org/entity/{qid}"] = lbl
        print(f"  Resolved {len(wd_labels)} labels")

    results: dict[str, list[dict]] = {}
    report_lines: list[str] = [
        "=" * 70,
        "  Nearest Neighbor Analysis — RotatE Embeddings (VC Knowledge Graph)",
        "=" * 70,
        "",
    ]

    for anchor_name, anchor_uri in ANCHORS.items():
        if anchor_uri not in uri_to_id:
            print(f"  WARNING: {anchor_name} not in embedding space — skipping")
            continue

        anchor_id = uri_to_id[anchor_uri]
        neighbors = nearest_neighbors(anchor_id, embeddings, k=K)

        neighbor_list = []
        for rank, (nid, sim) in enumerate(neighbors, 1):
            nuri   = id_to_uri[nid]
            nlabel = uri_to_label(nuri, labels)
            neighbor_list.append({
                "rank":       rank,
                "uri":        nuri,
                "label":      nlabel,
                "cosine_sim": round(sim, 4),
            })

        results[anchor_name] = neighbor_list

        # --- Format report block ---
        report_lines.append(f"Anchor: {anchor_name}")
        report_lines.append(f"  URI: {anchor_uri}")
        report_lines.append(f"  {'Rank':<6} {'Label':<35} {'Cosine Sim':>10}")
        report_lines.append(f"  {'-'*55}")
        for n in neighbor_list:
            report_lines.append(
                f"  {n['rank']:<6} {n['label']:<35} {n['cosine_sim']:>10.4f}"
            )
        report_lines.append("")

    # --- Save JSON ---
    json_out = OUT_DIR / "nearest_neighbors.json"
    with json_out.open("w") as f:
        json.dump(results, f, indent=2)

    # --- Save text report ---
    report = "\n".join(report_lines)
    txt_out = OUT_DIR / "nearest_neighbors.txt"
    with txt_out.open("w") as f:
        f.write(report)
    print("\n" + report)

    # --- Cosine similarity heatmap between anchor entities ---
    anchor_names  = list(results.keys())
    anchor_ids    = [uri_to_id[ANCHORS[n]] for n in anchor_names]
    anchor_embs   = F.normalize(embeddings[anchor_ids], dim=1)
    sim_matrix    = (anchor_embs @ anchor_embs.T).detach().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(anchor_names)))
    ax.set_yticks(range(len(anchor_names)))
    ax.set_xticklabels(anchor_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(anchor_names, fontsize=9)
    for i in range(len(anchor_names)):
        for j in range(len(anchor_names)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(sim_matrix[i, j]) < 0.5 else "white")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Anchor Entity Similarity — RotatE Embeddings", fontsize=11)
    fig.tight_layout()
    heatmap_out = OUT_DIR / "nn_heatmap.png"
    fig.savefig(str(heatmap_out), dpi=150)
    plt.close(fig)

    print(f"\nOutputs written to {OUT_DIR}/")
    print(f"  {json_out.name}")
    print(f"  {txt_out.name}")
    print(f"  {heatmap_out.name}")


if __name__ == "__main__":
    main()
