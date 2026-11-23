"""
integrate.py — combine everyone's results and pick the final model
(proposal tasks 4 and 7).


Reads every results/disease/results_*.csv (README contract 4), writes one
combined table, and draws the two summary figures for the talk:

  summary_auc_by_representation.png  every representation x model, per split
  summary_leaky_to_honest.png        best k-mer model vs the length+GC control
                                     as the split gets harder

Run from the repo root, after the other modules:
    python src/integrate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plotstyle as ps   # noqa: E402

RESULTS = ROOT / "results" / "disease"
SPLITS = ["random", "family", "lenmatch"]
MODEL_STYLE = {                     # colour + shape, so it's never colour alone
    "logreg": dict(color=ps.TEXT_SECONDARY, marker="o", label="logistic regression"),
    "rf":     dict(color=ps.THIRD, marker="s", label="random forest"),
    "mlp":    dict(color="#6250d6", marker="^", label="MLP"),
    "transformer": dict(color="#eda100", marker="D", label="transformer"),
}


def load():
    frames = []
    for f in sorted(RESULTS.glob("results_*.csv")):
        d = pd.read_csv(f, dtype={"k": str})
        d["source"] = f.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["k"] = df.k.astype(str)
    df["control"] = df.notes.astype(str).str.extract(r"control=(\w+)")[0]
    df["label"] = np.select(
        [df.representation == "transformer_mlm",
         df.representation == "transformer_scratch",
         df.representation == "composition_only",
         df.representation == "w2v_mean",
         df.split_type == "shuffled_bases"],
        ["transformer, pretrained (codons)",
         "transformer, no pretraining (codons)",
         "control: " + df.control.fillna("").str.replace("_", " + "),
         "word2vec mean, " + df.k.str.replace("codon", "codons").where(df.k == "codon",
                                                                       df.k + "-mers"),
         "control: shuffled bases (k=6)"],
        default="TF-IDF " + np.where(df.k == "codon", "codons", "k=" + df.k))
    return df


ORDER = ["TF-IDF k=3", "TF-IDF k=4", "TF-IDF k=6", "TF-IDF k=3-6", "TF-IDF codons",
         "word2vec mean, codons", "word2vec mean, 6-mers",
         "transformer, pretrained (codons)", "transformer, no pretraining (codons)",
         "control: length", "control: gc", "control: length + gc"]


def plot_by_representation(df):
    plt = ps.setup()
    d = df[df.split_type.isin(["family", "lenmatch"]) & df.label.isin(ORDER)]
    labels = [l for l in ORDER if l in set(d.label)][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.38 * len(labels) + 1.6), sharey=True)
    for ax, split in zip(axes, ["family", "lenmatch"]):
        s = d[d.split_type == split]
        ctrl = s[s.label == "control: length + gc"].roc_auc.max()
        if not np.isnan(ctrl):
            ax.axvline(ctrl, color=ps.NEUTRAL, lw=1)
            ax.text(ctrl, len(labels) - 0.4, "best length+GC ", color=ps.TEXT_SECONDARY,
                    fontsize=7.5, va="bottom", ha="right")
        ax.axvline(0.5, color=ps.NEUTRAL, lw=1, ls="--")
        ax.text(0.5, -0.75, " chance", color=ps.TEXT_SECONDARY, fontsize=7.5,
                va="bottom", ha="left")
        offsets = {"logreg": -0.22, "rf": 0.0, "mlp": 0.22, "transformer": 0.0}
        for model, st in MODEL_STYLE.items():
            m = s[s.model == model]
            y = [labels.index(l) + offsets[model] for l in m.label]
            ax.scatter(m.roc_auc, y, color=st["color"], marker=st["marker"], s=34,
                       label=st["label"], edgecolor="white", linewidth=0.8, zorder=3)
        for i, l in enumerate(labels):
            if l.startswith("control"):
                ax.axhspan(i - 0.45, i + 0.45, color="#f4f3f0", zorder=0)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{split} split" + ("  (length & GC matched)" if split == "lenmatch" else
                                         "  (paralogs kept apart)"))
        ax.set_xlabel("ROC-AUC on test set")
        ax.grid(axis="y", visible=False)
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=4, fontsize=8,
               bbox_to_anchor=(0.995, 0.995))
    lo = max(0.4, d.roc_auc.min() - 0.02)
    for ax in axes:
        ax.set_xlim(lo, d.roc_auc.max() + 0.02)
        ax.set_ylim(-0.8, len(labels) - 0.2)
    fig.suptitle("Every representation and model, on the two honest splits",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return ps.save(fig, "summary_auc_by_representation")


def plot_leaky_to_honest(df):
    plt = ps.setup()
    kmer = df[df.representation.isin(["tfidf", "w2v_mean", "transformer_mlm",
                                      "transformer_scratch"]) & df.split_type.isin(SPLITS)]
    ctrl = df[df.label == "control: length + gc"]
    best_k = kmer.groupby("split_type").roc_auc.max().reindex(SPLITS)
    best_c = ctrl.groupby("split_type").roc_auc.max().reindex(SPLITS)
    x = np.arange(len(SPLITS))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, best_k.values, color=ps.THIRD, marker="s", ms=8, label="best sequence model")
    ax.plot(x, best_c.values, color=ps.TEXT_SECONDARY, marker="o", ms=8,
            label="length + GC only (control)")
    for xi, v in zip(x, best_k.values):
        ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=ps.TEXT,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    for xi, v in zip(x, best_c.values):
        ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8, color=ps.TEXT_SECONDARY,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.axhline(0.5, color=ps.NEUTRAL, lw=1, ls="--")
    ax.text(2.25, 0.503, "chance", color=ps.TEXT_SECONDARY, fontsize=8, ha="right")
    ax.set_xticks(x, ["random\n(leaky)", "family\n(paralogs apart)",
                      "lenmatch\n(length & GC matched)"])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0.46, max(best_k.max(), best_c.max()) + 0.02)
    ax.set_ylabel("best ROC-AUC")
    ax.set_title("Length explains the easy wins; a sequence signal survives")
    ax.legend(loc="lower left")
    return ps.save(fig, "summary_leaky_to_honest")


def main():
    df = load()
    df.drop(columns=["control"]).to_csv(RESULTS / "all_results.csv", index=False)
    print(f"combined {len(df)} rows from {df.source.nunique()} files "
          f"({', '.join(sorted(df.source.unique()))})\n")

    main_rows = df[df.split_type.isin(SPLITS)]
    pivot = main_rows.pivot_table(index=["label", "model"], columns="split_type",
                                  values="roc_auc").reindex(columns=SPLITS).round(3)
    pivot.to_csv(RESULTS / "summary_auc_pivot.csv")
    with pd.option_context("display.width", 160):
        print("ROC-AUC by representation x model (columns = split):")
        print(pivot.to_string(), "\n")

    lm = df[(df.split_type == "lenmatch") & ~df.label.str.startswith("control")]
    best = lm.sort_values("roc_auc", ascending=False).head(5)
    print("top 5 on lenmatch (the headline split):")
    print(best[["label", "model", "roc_auc", "macro_f1", "accuracy"]].to_string(index=False))
    best.to_csv(RESULTS / "final_model_candidates.csv", index=False)

    plot_by_representation(df)
    plot_leaky_to_honest(df)
    print("\nwrote results/disease/all_results.csv, summary_auc_pivot.csv, "
          "final_model_candidates.csv, figures/disease/summary_*.png")


if __name__ == "__main__":
    main()
