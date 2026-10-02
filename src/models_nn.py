"""
models_nn.py — neural network (MLP) models.
Owner: Leora.

Trains a small feed-forward network on k-mer TF-IDF features for every
split x representation, and writes one row per run to
results/results_leora.csv (README contract 4).

Run from the repo root:
    python src/models_nn.py            # full grid
    python src/models_nn.py --quick    # species split, k=3 only — smoke test
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataio import load_split_data      # noqa: E402
from src.represent import vectorize         # noqa: E402

RESULTS = ROOT / "results" / "results_leora.csv"
FIGURES = ROOT / "figures"

SEED = 0
SPLITS = ["random", "nodup", "species"]

# (label written to results, k passed to vectorize, overlap)
REPRESENTATIONS = [
    ("3",     3,      True),
    ("4",     4,      True),
    ("6",     6,      True),
    ("3-6",   (3, 6), True),
    ("codon", 3,      False),
]


def make_model():
    """One hidden layer, early stopping on a held-out slice of TRAIN only."""
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        alpha=1e-4,                 # L2 penalty
        early_stopping=True,        # stops when validation score plateaus
        validation_fraction=0.1,    # carved from training data, never test
        n_iter_no_change=10,
        max_iter=200,
        random_state=SEED,
    )

def fit_score(Xtr, ytr, Xte, yte):
    """Train the standard MLP on any features. Returns (row_metrics, pred)."""
    model = make_model()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        "n_features": Xtr.shape[1],
        "seed": SEED,
        "epochs": model.n_iter_,
    }, pred

def run_one(split, k_label, k, overlap):
    tr, te = load_split_data(split)
    Xtr, Xte, names = vectorize(
        tr.sequence.tolist(), te.sequence.tolist(),
        k=k, mode="tfidf", overlap=overlap,
    )
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()

    model = make_model()
    t0 = time.time()
    model.fit(Xtr, ytr)
    secs = time.time() - t0
    pred = model.predict(Xte)

    converged = model.n_iter_ < model.max_iter
    note = f"epochs={model.n_iter_}; {secs:.0f}s"
    if not converged:
        note += "; hit max_iter"

    row = {
        "representation": "tfidf",
        "k": k_label,
        "model": "mlp",
        "split_type": split,
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        "n_features": Xtr.shape[1],
        "seed": SEED,
        "notes": note,
    }
    return row, (yte, pred)


def save_confusion(yte, pred, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(set(yte) | set(pred))
    cm = confusion_matrix(yte, pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(name.replace("_", " "))
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def main(quick=False):
    splits = ["species"] if quick else SPLITS
    reps = REPRESENTATIONS[:1] if quick else REPRESENTATIONS

    rows, best = [], None
    for split in splits:
        for k_label, k, overlap in reps:
            row, (yte, pred) = run_one(split, k_label, k, overlap)
            rows.append(row)
            print(f"{split:8s} k={k_label:5s}  acc {row['accuracy']:.3f}  "
                  f"macroF1 {row['macro_f1']:.3f}  ({row['notes']})")
            if split == "species" and (best is None or row["macro_f1"] > best[0]):
                best = (row["macro_f1"], k_label, yte, pred)

    out = pd.DataFrame(rows)
    RESULTS.parent.mkdir(exist_ok=True)
    out.to_csv(RESULTS, index=False)
    print(f"\nwrote {len(out)} rows -> {RESULTS.relative_to(ROOT)}")

    if best:
        _, k_label, yte, pred = best
        name = f"confusion_mlp_k{k_label}_species"
        save_confusion(yte, pred, name)
        print(f"wrote figures/{name}.png")

    print("\nmacro-F1 by split (rows) and k (columns):")
    print(out.pivot(index="split_type", columns="k", values="macro_f1")
             .reindex(index=[s for s in SPLITS if s in splits]))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)