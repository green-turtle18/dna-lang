"""
models.py — logistic regression and random forest.

Owner: Pranay (NAME3).

Trains logistic regression and random forest on k-mer TF-IDF features for
every split x representation, and writes one row per run to
results/results_pranay.csv (README contract 4).

Run from the repo root:
    python src/models.py            # full grid
    python src/models.py --quick    # species split, k=3 only — smoke test
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataio import load_split_data      # noqa: E402
from src.represent import vectorize         # noqa: E402

RESULTS = ROOT / "results" / "results_pranay.csv"
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

MODELS = ["logreg", "rf"]

COLUMNS = [
    "representation", "k", "model", "split_type",
    "accuracy", "macro_f1", "n_features", "seed", "notes",
]


def make_model(kind):
    """The standard classifier behind each model name in the results table."""
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=SEED,
        )
    raise ValueError(f"unknown model {kind!r}")


def run(Xtr, ytr, Xte, yte, model, name, **meta):
    """Fit `model` and return (contract-4 row without notes, predictions).

    NAME4's controls feed any feature matrix in here — it is just a
    different X.
    """
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    row = {
        "model": name,
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
        "n_features": Xtr.shape[1],
        "seed": SEED,
        **meta,
    }
    return row, pred


def run_one(split, k_label, k, overlap, kind):
    tr, te = load_split_data(split)
    Xtr, Xte, names = vectorize(
        tr.sequence.tolist(), te.sequence.tolist(),
        k=k, mode="tfidf", overlap=overlap,
    )
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()

    model = make_model(kind)
    t0 = time.time()
    row, pred = run(
        Xtr, ytr, Xte, yte, model, kind,
        representation="tfidf", k=k_label, split_type=split,
    )
    row["notes"] = f"{time.time() - t0:.0f}s"
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
    models = MODELS[:1] if quick else MODELS

    rows, best = [], {}
    for split in splits:
        for k_label, k, overlap in reps:
            for kind in models:
                row, (yte, pred) = run_one(split, k_label, k, overlap, kind)
                rows.append(row)
                # Write after every run so a long grid survives an interruption.
                RESULTS.parent.mkdir(exist_ok=True)
                pd.DataFrame(rows)[COLUMNS].to_csv(RESULTS, index=False)
                print(f"{split:8s} k={k_label:5s} {kind:6s}  "
                      f"acc {row['accuracy']:.3f}  macroF1 {row['macro_f1']:.3f}  "
                      f"({row['notes']})", flush=True)
                if split == "species":
                    prev = best.get(kind)
                    if prev is None or row["macro_f1"] > prev[0]:
                        best[kind] = (row["macro_f1"], k_label, yte, pred)

    out = pd.DataFrame(rows)[COLUMNS]
    RESULTS.parent.mkdir(exist_ok=True)
    out.to_csv(RESULTS, index=False)
    print(f"\nwrote {len(out)} rows -> {RESULTS.relative_to(ROOT)}")

    for kind, (_, k_label, yte, pred) in best.items():
        name = f"confusion_{kind}_k{k_label}_species"
        save_confusion(yte, pred, name)
        print(f"wrote figures/{name}.png")

    for kind in models:
        sub = out[out.model == kind]
        print(f"\nmacro-F1 by split (rows) and k (columns) — {kind}:")
        print(sub.pivot(index="split_type", columns="k", values="macro_f1")
                 .reindex(index=[s for s in SPLITS if s in splits]))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
