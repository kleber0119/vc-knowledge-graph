"""
KGE Training: Train TransE, DistMult, ComplEx, and RotatE on the VC knowledge graph
using the PyKEEN framework.

Input:  kg_artifacts/kge/train.txt, valid.txt, test.txt  (tab-separated URI triples)
Output: kg_artifacts/kge/results/<model>/
          checkpoints/trained_model.pkl
          results.json          (MRR, Hits@1/3/10)
          training_loss.png
        kg_artifacts/kge/comparison.json  (side-by-side metrics)
        kg_artifacts/kge/comparison.png   (bar chart)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory as TF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
KGE_DIR   = ROOT / "kg_artifacts" / "kge"
TRAIN_TXT = KGE_DIR / "train.txt"
VALID_TXT = KGE_DIR / "valid.txt"
TEST_TXT  = KGE_DIR / "test.txt"
RESULTS_DIR = KGE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Model configurations
# Each entry: model_name -> pykeen kwargs forwarded to pipeline()
# ---------------------------------------------------------------------------
MODEL_CONFIGS: dict[str, dict] = {
    # TransE — translational model; self-adversarial sampling provides hard negatives.
    "TransE": {
        "model": "TransE",
        "model_kwargs": {"embedding_dim": 256},
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": 1e-3},
        "training_loop": "slcwa",
        "training_kwargs": {"num_epochs": 300, "batch_size": 512},
        "negative_sampler": "bernoulli",
        "negative_sampler_kwargs": {"num_negs_per_pos": 256},
        "loss": "NSSALoss",
        "loss_kwargs": {"margin": 9.0, "adversarial_temperature": 1.0},
    },
    # DistMult — bilinear model; SoftplusLoss with bernoulli sampler.
    "DistMult": {
        "model": "DistMult",
        "model_kwargs": {"embedding_dim": 256},
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": 1e-3},
        "training_loop": "slcwa",
        "training_kwargs": {"num_epochs": 300, "batch_size": 512},
        "negative_sampler": "bernoulli",
        "negative_sampler_kwargs": {"num_negs_per_pos": 256},
        "loss": "SoftplusLoss",
    },
    # ComplEx — same setup as DistMult.
    "ComplEx": {
        "model": "ComplEx",
        "model_kwargs": {"embedding_dim": 256},
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": 1e-3},
        "training_loop": "slcwa",
        "training_kwargs": {"num_epochs": 300, "batch_size": 512},
        "negative_sampler": "bernoulli",
        "negative_sampler_kwargs": {"num_negs_per_pos": 256},
        "loss": "SoftplusLoss",
    },
    # RotatE — rotation model; self-adversarial NSSALoss is the standard setup.
    "RotatE": {
        "model": "RotatE",
        "model_kwargs": {"embedding_dim": 256},
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": 1e-3},
        "training_loop": "slcwa",
        "training_kwargs": {"num_epochs": 70, "batch_size": 512},
        "negative_sampler": "bernoulli",
        "negative_sampler_kwargs": {"num_negs_per_pos": 256},
        "loss": "NSSALoss",
        "loss_kwargs": {"margin": 6.0, "adversarial_temperature": 1.0},
    },
}

# ---------------------------------------------------------------------------
# Load triples
# ---------------------------------------------------------------------------

def load_factories() -> tuple[TF, TF, TF]:
    """Load train/valid/test as TriplesFactory objects sharing the same entity/relation map."""
    train_tf = TF.from_path(TRAIN_TXT)
    valid_tf = TF.from_path(
        VALID_TXT,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )
    test_tf = TF.from_path(
        TEST_TXT,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )
    print(
        f"Loaded  train={train_tf.num_triples:,}  "
        f"valid={valid_tf.num_triples:,}  "
        f"test={test_tf.num_triples:,}  "
        f"entities={train_tf.num_entities:,}  "
        f"relations={train_tf.num_relations:,}"
    )
    return train_tf, valid_tf, test_tf


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def train_model(
    name: str,
    cfg: dict,
    train_tf: TF,
    valid_tf: TF,
    test_tf: TF,
) -> dict:
    """
    Run the PyKEEN pipeline for one model.
    Returns a metrics dict: {model, MRR, Hits@1, Hits@3, Hits@10, train_time_s}.
    """
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training {name}")
    print(f"{'='*60}")

    t0 = time.time()

    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        device=DEVICE,
        random_seed=42,
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={"filtered": True},
        **cfg,
    )

    elapsed = time.time() - t0

    # --- Save model checkpoint ---
    result.save_to_directory(str(out_dir))

    # --- Extract metrics ---
    metrics = result.metric_results.to_dict()
    mrr    = metrics["both"]["realistic"]["inverse_harmonic_mean_rank"]
    hits1  = metrics["both"]["realistic"]["hits_at_1"]
    hits3  = metrics["both"]["realistic"]["hits_at_3"]
    hits10 = metrics["both"]["realistic"]["hits_at_10"]

    summary = {
        "model":        name,
        "MRR":          round(mrr,   4),
        "Hits@1":       round(hits1, 4),
        "Hits@3":       round(hits3, 4),
        "Hits@10":      round(hits10, 4),
        "train_time_s": round(elapsed, 1),
        "epochs":       cfg["training_kwargs"]["num_epochs"],
        "embedding_dim": cfg["model_kwargs"]["embedding_dim"],
    }

    # --- Save per-model results.json ---
    with (out_dir / "results.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Plot training loss ---
    losses = result.losses
    if losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{name} — Training Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_dir / "training_loss.png"), dpi=150)
        plt.close(fig)

    print(
        f"  MRR={mrr:.4f}  Hits@1={hits1:.4f}  "
        f"Hits@3={hits3:.4f}  Hits@10={hits10:.4f}  "
        f"({elapsed:.0f}s)"
    )
    return summary


# ---------------------------------------------------------------------------
# Comparison chart
# ---------------------------------------------------------------------------

def plot_comparison(summaries: list[dict]) -> None:
    models  = [s["model"]  for s in summaries]
    mrr     = [s["MRR"]    for s in summaries]
    hits1   = [s["Hits@1"] for s in summaries]
    hits3   = [s["Hits@3"] for s in summaries]
    hits10  = [s["Hits@10"] for s in summaries]

    x      = range(len(models))
    width  = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar([i - 1.5 * width for i in x], mrr,    width, label="MRR")
    ax.bar([i - 0.5 * width for i in x], hits1,  width, label="Hits@1")
    ax.bar([i + 0.5 * width for i in x], hits3,  width, label="Hits@3")
    ax.bar([i + 1.5 * width for i in x], hits10, width, label="Hits@10")

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("KGE Model Comparison — VC Knowledge Graph")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(KGE_DIR / "comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nComparison chart saved to {KGE_DIR / 'comparison.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    train_tf, valid_tf, test_tf = load_factories()

    summaries: list[dict] = []

    skip_models = {"TransE", "DistMult", "ComplEx"}   # skip if results already exist
    for name, cfg in MODEL_CONFIGS.items():
        results_path = RESULTS_DIR / name / "results.json"
        if name in skip_models and results_path.exists():
            print(f"\nSkipping {name} — results already exist.")
            with results_path.open() as f:
                summaries.append(json.load(f))
            continue
        summary = train_model(name, cfg, train_tf, valid_tf, test_tf)
        summaries.append(summary)

    # --- Side-by-side comparison ---
    comparison_path = KGE_DIR / "comparison.json"
    with comparison_path.open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nComparison JSON saved to {comparison_path}")

    plot_comparison(summaries)

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print(f"{'Model':<12} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>8} {'Time(s)':>9}")
    print("-" * 70)
    for s in summaries:
        print(
            f"{s['model']:<12} {s['MRR']:>8.4f} {s['Hits@1']:>8.4f} "
            f"{s['Hits@3']:>8.4f} {s['Hits@10']:>8.4f} {s['train_time_s']:>9.1f}"
        )
    print("=" * 70)
    print(f"\nAll results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
