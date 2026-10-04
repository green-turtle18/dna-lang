"""
models.py — logistic regression and random forest.

Owner: Pranay (NAME3).

Trains logistic regression and random forest on k-mer TF-IDF features for
every split x representation, and writes one row per run to
results/results_pranay.csv (README contract 4).

Run from the repo root:
    python src/models.py            # full grid
    python src/models.py --quick    # species split, k=3 only — smoke test

Why these choices:
  * Only k-mer TF-IDF features are fed in. The raw sequence -> sparse
    matrix step lives in represent.py (NAME2) and is fitted on the
    training fold only, so no test vocabulary leaks into training.
  * Macro-F1 is reported next to accuracy because the 7 gene-family
    classes are imbalanced (~5:1). A model that just over-predicts the
    biggest class can post a good accuracy and a bad macro-F1, so
    macro-F1 is the number to trust in the talk.
  * Every model gets a fixed random_state (SEED) so the results file is
    reproducible on any machine.
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

# One seed for the whole project so two people's numbers are comparable.
SEED = 0

# The three shared splits (README contract 2), hardest last:
#   random  -> plain random (near-duplicates leak across folds)
#   nodup   -> near-duplicates kept on one side
#   species -> train human+chimp, test dog (cross-species generalisation)
SPLITS = ["random", "nodup", "species"]

# (label written to results, k passed to vectorize, overlap)
# "codon" is non-overlapping frame-0 triplets -- the biologically correct
# reading frame for coding DNA, so it is a fair tokenisation to compare.
REPRESENTATIONS = [
    ("3",     3,      True),
    ("4",     4,      True),
    ("6",     6,      True),
    ("3-6",   (3, 6), True),
    ("codon", 3,      False),
]

MODELS = ["logreg", "rf"]

# Exact column order required by README contract 4.
COLUMNS = [
    "representation", "k", "model", "split_type",
    "accuracy", "macro_f1", "n_features", "seed", "notes",
]


def make_model(kind):
    """Build the standard classifier behind each model name.

    logreg : multinomial logistic regression on the sparse TF-IDF matrix.
             max_iter raised so it converges on the larger k=6 / 3-6 vocab.
    rf     : 200-tree random forest. n_jobs=-1 uses all cores because the
             biggest representation (3-6, ~5,400 features) is slow.
    """
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

    This is the reusable entry point: NAME4's controls call it with a
    different X (length/GC instead of k-mers) and everything else is
    identical. Returns the predictions as well so the caller can build a
    confusion matrix without fitting twice.
    """
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    row = {
        "model": name,
        # zero_division=0 avoids warnings if a class is never predicted.
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
        "n_features": Xtr.shape[1],
        "seed": SEED,
        **meta,
    }
    return row, pred


def run_one(split, k_label, k, overlap, kind):
    """Vectorise one split x representation, train one model, score it."""
    # Shared, pre-committed split -- never made locally (README rule).
    tr, te = load_split_data(split)

    # Fit the k-mer vocabulary on TRAIN only, then transform test.
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
    # Record wall-clock time so slow representations are visible in the CSV.
    row["notes"] = f"{time.time() - t0:.0f}s"
    return row, (yte, pred)


def save_confusion(yte, pred, name):
    """Save a row-normalised confusion matrix as a 150 dpi PNG.

    Normalising by true-class count makes the minority classes readable
    even though they have far fewer test examples than class 6.
    """
    import matplotlib
    matplotlib.use("Agg")           # no display needed on servers/CI
    import matplotlib.pyplot as plt

    labels = sorted(set(yte) | set(pred))
    cm = confusion_matrix(yte, pred, labels=labels)
    # Each row sums to 1: cell (i, j) = fraction of true class i -> pred j.
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    # Print raw counts on top of the colour, readable on light/dark cells.
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
    """Run the full grid (or a one-cell smoke test) and write results.

    Full grid = 3 splits x 5 representations x 2 models = 30 rows.
    """
    splits = ["species"] if quick else SPLITS
    reps = REPRESENTATIONS[:1] if quick else REPRESENTATIONS
    models = MODELS[:1] if quick else MODELS

    rows, best = [], {}
    for split in splits:
        for k_label, k, overlap in reps:
            for kind in models:
                row, (yte, pred) = run_one(split, k_label, k, overlap, kind)
                rows.append(row)

                # Write after every run so a long grid survives an
                # interruption and the CSV is never left half-written.
                RESULTS.parent.mkdir(exist_ok=True)
                pd.DataFrame(rows)[COLUMNS].to_csv(RESULTS, index=False)

                print(f"{split:8s} k={k_label:5s} {kind:6s}  "
                      f"acc {row['accuracy']:.3f}  macroF1 {row['macro_f1']:.3f}  "
                      f"({row['notes']})", flush=True)

                # Track the best macro-F1 per model on the hardest split
                # so we know which k to draw the confusion matrix for.
                if split == "species":
                    prev = best.get(kind)
                    if prev is None or row["macro_f1"] > prev[0]:
                        best[kind] = (row["macro_f1"], k_label, yte, pred)

    out = pd.DataFrame(rows)[COLUMNS]
    RESULTS.parent.mkdir(exist_ok=True)
    out.to_csv(RESULTS, index=False)
    print(f"\nwrote {len(out)} rows -> {RESULTS.relative_to(ROOT)}")

    # One confusion matrix per model, for the best-k species run.
    for kind, (_, k_label, yte, pred) in best.items():
        name = f"confusion_{kind}_k{k_label}_species"
        save_confusion(yte, pred, name)
        print(f"wrote figures/{name}.png")

    # Pivot table: rows = split, columns = k. This is slide 1 material.
    for kind in models:
        sub = out[out.model == kind]
        print(f"\nmacro-F1 by split (rows) and k (columns) — {kind}:")
        print(sub.pivot(index="split_type", columns="k", values="macro_f1")
                 .reindex(index=[s for s in SPLITS if s in splits]))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
