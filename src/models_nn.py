"""
models_nn.py — neural network (MLP) models.
Owner: Leora.

Trains a small feed-forward network on k-mer TF-IDF features for every
split x representation of the disease dataset, and writes one row per run
to results/disease/results_leora.csv (README contract 4).

Run from the repo root:
    python src/models_nn.py            # full grid
    python src/models_nn.py --quick    # lenmatch split, k=3 only — smoke test

Class imbalance: only 29% of genes are disease genes. sklearn's MLP has no
class_weight option, so without help it learns to predict "non-disease" for
everything (71% accuracy, macro-F1 0.42, no DNA read at all). We therefore
oversample disease genes in the TRAINING set until the classes are equal.
The test set is never touched, so the numbers stay honest.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.disease_data import load_disease_split_data   # noqa: E402
from src.represent import vectorize                    # noqa: E402

RESULTS = ROOT / "results" / "disease" / "results_leora.csv"
FIGURES = ROOT / "figures" / "disease"

SEED = 0
# Hardest last: random (paralogs leak) -> family (paralogs kept apart)
# -> lenmatch (family split, disease and non-disease matched on length and GC)
SPLITS = ["random", "family", "lenmatch"]
HARDEST = "lenmatch"
CLASS_NAMES = {0: "non-disease", 1: "disease"}

# (label written to results, k passed to vectorize, overlap)
REPRESENTATIONS = [
    ("3",     3,      True),
    ("4",     4,      True),
    ("6",     6,      True),
    ("3-6",   (3, 6), True),
    ("codon", 3,      False),
]

COLUMNS = [
    "representation", "k", "model", "split_type",
    "accuracy", "macro_f1", "roc_auc", "n_features", "seed", "notes",
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


def oversample(X, y, seed=SEED):
    """Repeat random minority-class rows until both classes are equal size.

    Call on TRAINING data only. Works on sparse matrices.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    small = classes[counts.argmin()]
    small_idx = np.where(y == small)[0]
    extra = rng.choice(small_idx, counts.max() - counts.min(), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(idx)
    return X[idx], y[idx]


def scores(yte, pred, prob):
    return {
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        "roc_auc": round(roc_auc_score(yte, prob), 4),
    }


def fit_score(Xtr, ytr, Xte, yte):
    """Train the standard MLP on any features. Returns (row_metrics, pred)."""
    model = make_model()
    Xo, yo = oversample(Xtr, ytr)
    model.fit(Xo, yo)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    return {
        **scores(yte, pred, prob),
        "n_features": Xtr.shape[1],
        "seed": SEED,
        "epochs": model.n_iter_,
    }, pred


def run_one(split, k_label, k, overlap):
    tr, te = load_disease_split_data(split)
    Xtr, Xte, names = vectorize(
        tr.sequence.tolist(), te.sequence.tolist(),
        k=k, mode="tfidf", overlap=overlap,
    )
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()

    model = make_model()
    t0 = time.time()
    Xo, yo = oversample(Xtr, ytr)
    model.fit(Xo, yo)
    secs = time.time() - t0
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]

    converged = model.n_iter_ < model.max_iter
    note = f"epochs={model.n_iter_}; oversampled; {secs:.0f}s"
    if not converged:
        note += "; hit max_iter"

    row = {
        "representation": "tfidf",
        "k": k_label,
        "model": "mlp",
        "split_type": split,
        **scores(yte, pred, prob),
        "n_features": Xtr.shape[1],
        "seed": SEED,
        "notes": note,
    }
    return row, (yte, pred)


def save_confusion(yte, pred, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [0, 1]
    cm = confusion_matrix(yte, pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ticks = [CLASS_NAMES[l] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), ticks)
    ax.set_yticks(range(len(labels)), ticks)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(name.replace("_", " "))
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def main(quick=False):
    splits = [HARDEST] if quick else SPLITS
    reps = REPRESENTATIONS[:1] if quick else REPRESENTATIONS

    rows, best = [], None
    for split in splits:
        for k_label, k, overlap in reps:
            row, (yte, pred) = run_one(split, k_label, k, overlap)
            rows.append(row)
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows)[COLUMNS].to_csv(RESULTS, index=False)
            print(f"{split:8s} k={k_label:5s}  acc {row['accuracy']:.3f}  "
                  f"macroF1 {row['macro_f1']:.3f}  AUC {row['roc_auc']:.3f}  "
                  f"({row['notes']})", flush=True)
            if split == HARDEST and (best is None or row["roc_auc"] > best[0]):
                best = (row["roc_auc"], k_label, yte, pred)

    out = pd.DataFrame(rows)[COLUMNS]
    out.to_csv(RESULTS, index=False)
    print(f"\nwrote {len(out)} rows -> {RESULTS.relative_to(ROOT)}")

    if best:
        _, k_label, yte, pred = best
        name = f"confusion_mlp_k{k_label}_{HARDEST}"
        save_confusion(yte, pred, name)
        print(f"wrote {(FIGURES / name).relative_to(ROOT)}.png")

    for metric in ["roc_auc", "macro_f1"]:
        print(f"\n{metric} by split (rows) and k (columns):")
        print(out.pivot(index="split_type", columns="k", values=metric)
                 .reindex(index=[s for s in SPLITS if s in splits]))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
