"""
controls.py — control experiments and PCA visualisation.

Owner: Nidhi.

This module tests whether classification performance can be explained by
simple sequence properties (length and GC content), rather than k-mer
sequence patterns.

It also produces a PCA visualisation of the TF-IDF k-mer representation.

Usage from repository root:

    python src/controls.py

Quick version:

    python src/controls.py --quick

Outputs:

    results/results_name4.csv
    figures/pca_tfidf_k6.png

The controls use the shared train/test splits produced by dataio.py.
No split is created locally.

Conclusion from comparison analysis:
    The strong performance of the k-mer models cannot be explained by sequence length or GC content alone.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.dataio import load_split_data, SPLITS
from src.represent import vectorize


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results_nidhi.csv"
FIGURES = ROOT / "figures"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 0

# The README/project currently considers k=6 an important representation.
PCA_K = 6

# Run all three shared splits for the controls.
DEFAULT_SPLITS = ["random", "nodup", "species"]

# Quick mode is useful while developing.
QUICK_SPLITS = ["species"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true, predictions):
    """Return accuracy and macro-F1 for a set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Simple feature controls
# ---------------------------------------------------------------------------

def make_control_features(train, test, control):
    """
    Return train/test feature matrices for a simple control.

    control:
        "length"
        "gc"
        "length_gc"
    """
    if control == "length":
        columns = ["length"]

    elif control == "gc":
        columns = ["gc"]

    elif control == "length_gc":
        columns = ["length", "gc"]

    else:
        raise ValueError(
            "control must be one of: length, gc, length_gc"
        )

    return (
        train[columns].to_numpy(dtype=float),
        test[columns].to_numpy(dtype=float),
    )


def run_control(split, control, model_name):
    """
    Train a logistic-regression classifier using only simple sequence
    properties and evaluate it on the shared test split.
    """
    train, test = load_split_data(split)

    X_train, X_test = make_control_features(
        train,
        test,
        control,
    )

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # Length is on a much larger numerical scale than GC.
    # Standardisation makes the two-feature control well behaved.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def make_model(kind):
        if kind == "logreg":
            return LogisticRegression(
                max_iter=2000,
                random_state=SEED
            )

        if kind == "rf":
            return RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                random_state=SEED
            )

        raise ValueError(f"Unknown model: {kind}")

    model = make_model(model_name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = evaluate_predictions(y_test, predictions)

    return {
        "representation": "composition_only",
        "k": "none",
        "model": model_name,
        "split_type": split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "n_features": X_train.shape[1],
        "seed": SEED,
        "notes": f"control={control}",
    }

# ---------------------------------------------------------------------------
# Shuffled-base control
# ---------------------------------------------------------------------------

def shuffle_sequence(sequence, rng):
    """Randomly permute bases within one sequence."""
    bases = list(sequence)
    rng.shuffle(bases)
    return "".join(bases)


def run_shuffled_bases(model_name, seed=SEED):
    """
    Shuffle bases within each sequence, preserving length and base
    composition while destroying original k-mer order.

    Uses the species split: train human+chimp, test dog.
    """
    train, test = load_split_data("species")

    rng = np.random.default_rng(seed)

    train = train.copy()
    test = test.copy()

    train["sequence"] = train["sequence"].map(
        lambda s: shuffle_sequence(s, rng)
    )

    test["sequence"] = test["sequence"].map(
        lambda s: shuffle_sequence(s, rng)
    )

    X_train, X_test, features = vectorize(
        train["sequence"].tolist(),
        test["sequence"].tolist(),
        k=6,
        mode="tfidf",
        overlap=True,
    )

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    if model_name == "logreg":
        model = LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = evaluate_predictions(y_test, predictions)

    return {
        "representation": "tfidf",
        "k": 6,
        "model": model_name,
        "split_type": "shuffled_bases",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "n_features": X_train.shape[1],
        "seed": seed,
        "notes": "bases shuffled within each sequence; species split",
    }

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def make_pca_plot(
    split="species",
    k=PCA_K,
    seed=SEED,
):
    """
    Create a 2D PCA plot of the TF-IDF k-mer representation.

    PCA is fitted only on the training data. Test sequences are transformed
    using the same fitted vectorizer and PCA transformation.

    The resulting figure is saved under figures/.
    """
    train, test = load_split_data(split)

    X_train, X_test, features = vectorize(
        train["sequence"].tolist(),
        test["sequence"].tolist(),
        k=k,
        mode="tfidf",
        overlap=True,
    )

    # PCA requires a dense matrix. k=6 has at most 4096 possible features,
    # which is manageable for this dataset.
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    pca = PCA(
        n_components=2,
        random_state=seed,
    )

    train_2d = pca.fit_transform(X_train_dense)
    test_2d = pca.transform(X_test_dense)

    FIGURES.mkdir(parents=True, exist_ok=True)

    # Use a non-interactive backend so this works on servers/CI too.
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))

    labels = sorted(
        set(train["label"].tolist()) |
        set(test["label"].tolist())
    )

    # Plot each class separately so the legend identifies gene families.
    for label in labels:
        train_mask = train["label"].to_numpy() == label
        test_mask = test["label"].to_numpy() == label

        if train_mask.any():
            ax.scatter(
                train_2d[train_mask, 0],
                train_2d[train_mask, 1],
                alpha=0.45,
                s=18,
                label=f"train class {label}",
            )

        if test_mask.any():
            ax.scatter(
                test_2d[test_mask, 0],
                test_2d[test_mask, 1],
                alpha=0.8,
                s=28,
                marker="x",
                label=f"test class {label}",
            )

    explained = pca.explained_variance_ratio_

    ax.set_xlabel(
        f"PC1 ({explained[0] * 100:.1f}% variance)"
    )
    ax.set_ylabel(
        f"PC2 ({explained[1] * 100:.1f}% variance)"
    )

    ax.set_title(
        f"TF-IDF {k}-mer PCA — {split} split"
    )

    ax.legend(
        fontsize=7,
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    output = FIGURES / f"pca_tfidf_k{k}.png"
    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "path": output,
        "explained_pc1": explained[0],
        "explained_pc2": explained[1],
        "n_features": len(features),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(quick=False):
    """
    Run all control experiments and create the PCA figure.
    """
    splits = QUICK_SPLITS if quick else DEFAULT_SPLITS

    controls = [
        "length",
        "gc",
        "length_gc",
    ]

    rows = []

    print("=" * 70)
    print("DNA LANGUAGE — CONTROL EXPERIMENTS")
    print("=" * 70)

    for split in splits:
        print(f"\n--- split: {split} ---")

        for control in controls:
            for model_name in ["logreg", "rf"]:
                row = run_control(split, control, model_name)
                rows.append(row)

                print(
                    f"{control:12s} "
                    f"{model_name:7s} "
                    f"accuracy={row['accuracy']:.3f} "
                    f"macro-F1={row['macro_f1']:.3f}"
                )

    # Shuffled-base control on the species split.
    print("\n--- shuffled bases ---")

    for model_name in ["logreg", "rf"]:
        row = run_shuffled_bases(model_name)
        rows.append(row)

        print(
            f"{model_name:7s} "
            f"accuracy={row['accuracy']:.3f} "
            f"macro-F1={row['macro_f1']:.3f}"
        )
        
    # Save control results.
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        RESULTS,
        index=False,
    )

    print(
        f"\nWrote {len(results_df)} control rows -> "
        f"{RESULTS.relative_to(ROOT)}"
    )

    # PCA is intended primarily as the visualisation for the hardest,
    # cross-species split.
    print("\n--- PCA ---")

    pca_info = make_pca_plot(
        split="species",
        k=PCA_K,
    )

    print(
        f"PC1 explains {pca_info['explained_pc1'] * 100:.2f}% variance"
    )
    print(
        f"PC2 explains {pca_info['explained_pc2'] * 100:.2f}% variance"
    )
    print(
        f"TF-IDF feature count: {pca_info['n_features']}"
    )
    print(
        f"Wrote PCA figure -> "
        f"{pca_info['path'].relative_to(ROOT)}"
    )

    print("\nControl results:")
    print(
        results_df[
            [
                "split_type",
                "notes",
                "accuracy",
                "macro_f1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)