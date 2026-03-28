"""
KB Size Sensitivity Analysis (Section 5.2)

Trains RotatE (best model) on three dataset sizes to observe how
performance scales with the number of triples:

    20K  →  35K  →  52K (full)

For each size:
  - Subsample from the full clean dataset (reproducible, fixed seed)
  - Re-split 80/10/10 with cold-start guarantee
  - Train RotatE with identical hyperparameters
  - Record MRR, Hits@1/3/10

Output:
  kg_artifacts/kge/sensitivity/
    results_<size>.json      per-size metrics
    sensitivity.json         all sizes combined
    sensitivity.png          scaling curve
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory as TF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
KGE_DIR     = ROOT / "kg_artifacts" / "kge"
TRAIN_TXT   = KGE_DIR / "train.txt"
VALID_TXT   = KGE_DIR / "valid.txt"
TEST_TXT    = KGE_DIR / "test.txt"
OUT_DIR     = KGE_DIR / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
VALID_RATIO = 0.10
TEST_RATIO  = 0.10

# Subset sizes to evaluate — full dataset is included automatically
SUBSET_SIZES = [20_000, 50_000]   # full (~52K) is added at runtime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RotatE config — identical across all sizes for a fair comparison
ROTATE_CFG: dict = {
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
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_triples() -> list[tuple[str, str, str]]:
    """Load and merge train + valid + test into one pool."""
    triples: list[tuple[str, str, str]] = []
    for path in (TRAIN_TXT, VALID_TXT, TEST_TXT):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) == 3:
                        triples.append((parts[0], parts[1], parts[2]))
    return triples


def split_triples(
    triples: list[tuple[str, str, str]],
    seed: int = SEED,
) -> tuple[list, list, list]:
    """80/10/10 split with cold-start guarantee (no OOV in valid/test)."""
    rng = random.Random(seed)
    shuffled = triples[:]
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_test  = int(n * TEST_RATIO)
    n_valid = int(n * VALID_RATIO)
    n_train = n - n_valid - n_test

    train = shuffled[:n_train]
    valid = shuffled[n_train: n_train + n_valid]
    test  = shuffled[n_train + n_valid:]

    for _ in range(10):
        train_ents = {s for s, _, _ in train} | {o for _, _, o in train}
        train_rels = {p for _, p, _ in train}
        rescued, kept_valid, kept_test = [], [], []
        for s, p, o in valid:
            (kept_valid if s in train_ents and o in train_ents and p in train_rels
             else rescued).append((s, p, o))
        for s, p, o in test:
            (kept_test if s in train_ents and o in train_ents and p in train_rels
             else rescued).append((s, p, o))
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


def subsample(
    triples: list[tuple[str, str, str]],
    size: int,
    seed: int = SEED,
) -> list[tuple[str, str, str]]:
    """Randomly sample `size` triples from the pool."""
    rng = random.Random(seed)
    if size >= len(triples):
        return triples[:]
    return rng.sample(triples, size)


def train_rotate(
    train_tf: TF,
    valid_tf: TF,
    test_tf:  TF,
    label:    str,
) -> dict:
    """Train RotatE and return metrics dict."""
    print(f"\n{'='*60}")
    print(f"  RotatE  |  size={label}  |  "
          f"train={train_tf.num_triples:,}  "
          f"entities={train_tf.num_entities:,}")
    print(f"{'='*60}")

    t0 = time.time()
    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        device=DEVICE,
        random_seed=SEED,
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={"filtered": True},
        **ROTATE_CFG,
    )
    elapsed = time.time() - t0

    m      = result.metric_results.to_dict()["both"]["realistic"]
    mrr    = m["inverse_harmonic_mean_rank"]
    hits1  = m["hits_at_1"]
    hits3  = m["hits_at_3"]
    hits10 = m["hits_at_10"]

    summary = {
        "size_label":   label,
        "num_triples":  train_tf.num_triples + valid_tf.num_triples + test_tf.num_triples,
        "num_train":    train_tf.num_triples,
        "num_entities": train_tf.num_entities,
        "num_relations": train_tf.num_relations,
        "MRR":          round(mrr,   4),
        "Hits@1":       round(hits1, 4),
        "Hits@3":       round(hits3, 4),
        "Hits@10":      round(hits10, 4),
        "train_time_s": round(elapsed, 1),
    }

    print(f"  MRR={mrr:.4f}  Hits@1={hits1:.4f}  "
          f"Hits@3={hits3:.4f}  Hits@10={hits10:.4f}  ({elapsed:.0f}s)")

    # save model + per-size result
    model_dir = OUT_DIR / f"RotatE_{label}"
    model_dir.mkdir(exist_ok=True)
    result.save_to_directory(str(model_dir))
    with (OUT_DIR / f"results_{label}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # save loss curve
    if result.losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result.losses, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"RotatE — Training Loss ({label} triples)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(OUT_DIR / f"loss_{label}.png"), dpi=150)
        plt.close(fig)

    return summary


def plot_sensitivity(results: list[dict]) -> None:
    sizes  = [r["num_triples"] for r in results]
    mrr    = [r["MRR"]    for r in results]
    hits1  = [r["Hits@1"] for r in results]
    hits3  = [r["Hits@3"] for r in results]
    hits10 = [r["Hits@10"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, mrr,    "o-", label="MRR",     linewidth=2)
    ax.plot(sizes, hits1,  "s-", label="Hits@1",  linewidth=2)
    ax.plot(sizes, hits3,  "^-", label="Hits@3",  linewidth=2)
    ax.plot(sizes, hits10, "D-", label="Hits@10", linewidth=2)

    ax.set_xlabel("Total Triples", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("RotatE — Performance vs KB Size", fontsize=13)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "sensitivity.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"\nScaling curve saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Using device: {DEVICE}")
    print("Loading full clean dataset ...")
    all_triples = load_all_triples()
    full_size   = len(all_triples)
    print(f"Full dataset: {full_size:,} triples")

    sizes = sorted(set(SUBSET_SIZES + [full_size]))
    results: list[dict] = []

    for size in sizes:
        label = f"{size // 1000}k" if size < full_size else "full"
        out_path = OUT_DIR / f"results_{label}.json"

        if out_path.exists():
            print(f"\nSkipping size={label} — results already exist.")
            with out_path.open() as f:
                results.append(json.load(f))
            continue

        # subsample → split → write temp files → load factories
        subset = subsample(all_triples, size)
        train_t, valid_t, test_t = split_triples(subset)

        tmp_train = OUT_DIR / f"_tmp_train_{label}.txt"
        tmp_valid = OUT_DIR / f"_tmp_valid_{label}.txt"
        tmp_test  = OUT_DIR / f"_tmp_test_{label}.txt"
        write_split(tmp_train, train_t)
        write_split(tmp_valid, valid_t)
        write_split(tmp_test,  test_t)

        train_tf = TF.from_path(tmp_train)
        valid_tf = TF.from_path(
            tmp_valid,
            entity_to_id=train_tf.entity_to_id,
            relation_to_id=train_tf.relation_to_id,
        )
        test_tf = TF.from_path(
            tmp_test,
            entity_to_id=train_tf.entity_to_id,
            relation_to_id=train_tf.relation_to_id,
        )

        summary = train_rotate(train_tf, valid_tf, test_tf, label)
        results.append(summary)

        # clean up temp files
        tmp_train.unlink()
        tmp_valid.unlink()
        tmp_test.unlink()

    # save combined results
    combined_path = OUT_DIR / "sensitivity.json"
    with combined_path.open("w") as f:
        json.dump(results, f, indent=2)

    plot_sensitivity(results)

    # print summary table
    print("\n" + "=" * 65)
    print(f"{'Size':<10} {'Triples':>10} {'Entities':>10} "
          f"{'MRR':>8} {'Hits@1':>8} {'Hits@10':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['size_label']:<10} {r['num_triples']:>10,} "
              f"{r['num_entities']:>10,} {r['MRR']:>8.4f} "
              f"{r['Hits@1']:>8.4f} {r['Hits@10']:>8.4f}")
    print("=" * 65)
    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
