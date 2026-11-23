"""
vocab.py — exploring the DNA "language" (proposal task 2).
Owner: Leora.

Treats each gene as a document and k-mers as words, then asks:
  1. Sequence statistics per class (length, GC, GC at codon position 3, CpG).
  2. Which words are frequent and which are rare, and is it the same in both
     classes? (rank-frequency "Zipf" curve, top/bottom words)
  3. Which words differ between disease and non-disease genes?
     (per-gene relative frequency, Mann-Whitney U test, Benjamini-Hochberg FDR)

Step 3 uses the lenmatch subset, where both classes have the same length and
GC distribution, so a word can't look "disease-like" just because disease
genes are longer. Frequencies are per gene (count / total words in that
gene), so long genes don't dominate either.

Run from the repo root:
    python src/vocab.py

Outputs: results/disease/vocab/*.csv and figures/disease/vocab_*.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.disease_data import load_disease_data, load_disease_splits, CODON  # noqa: E402
from src import plotstyle as ps                                            # noqa: E402

OUT = ROOT / "results" / "disease" / "vocab"
SENSE_CODONS = sorted(c for c, aa in CODON.items() if aa != "*")


# ------------------------------------------------------------- features

def codons(seq):
    return [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]


def word_counts(seqs, k):
    """Sparse count matrix and vocabulary. k=int for overlapping k-mers,
    k="codon" for in-frame codons."""
    if k == "codon":
        vec = CountVectorizer(analyzer=codons, vocabulary=SENSE_CODONS, lowercase=False)
    else:
        vec = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
    X = vec.fit_transform(seqs)
    return X, np.array(vec.get_feature_names_out())


def relative(X):
    """Row-normalise counts to per-gene relative frequencies (dense)."""
    X = X.astype(float).toarray()
    return X / X.sum(axis=1, keepdims=True)


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    return out


def gc3(seq):
    third = seq[2::3]
    return (third.count("G") + third.count("C")) / len(third)


def cpg_oe(seq):
    """CpG observed/expected: (#CG * L) / (#C * #G). ~1 means no depletion."""
    c, g = seq.count("C"), seq.count("G")
    return seq.count("CG") * len(seq) / (c * g) if c and g else np.nan


# --------------------------------------------------------------- analyses

def sequence_stats(df):
    d = df.assign(gc3=df.sequence.map(gc3), cpg_oe=df.sequence.map(cpg_oe))
    rows = []
    for subset, sub in [("all genes", d), ("lenmatch", d[d.lenmatch])]:
        for lab in [0, 1]:
            s = sub[sub.label == lab]
            rows.append({"subset": subset, "class": ps.CLASS_NAMES[lab], "n": len(s),
                         "median_length": int(s.length.median()),
                         "mean_gc": round(s.gc.mean(), 4),
                         "mean_gc3": round(s.gc3.mean(), 4),
                         "mean_cpg_obs_exp": round(s.cpg_oe.mean(), 4)})
        for col in ["length", "gc", "gc3", "cpg_oe"]:
            p = mannwhitneyu(sub[sub.label == 1][col], sub[sub.label == 0][col]).pvalue
            rows.append({"subset": subset, "class": f"p-value {col}", "n": np.nan,
                         "median_length": np.nan, "mean_gc": np.nan, "mean_gc3": np.nan,
                         "mean_cpg_obs_exp": round(p, 6)})
    return pd.DataFrame(rows)


def frequent_and_rare(df, k=6, n=15):
    """Top/bottom words per class by mean relative frequency."""
    X, vocab = word_counts(df.sequence.tolist(), k)
    R = relative(X)
    rows, means = [], {}
    for lab in [0, 1]:
        m = R[df.label.to_numpy() == lab].mean(axis=0)
        means[lab] = m
        order = np.argsort(-m)
        for rank, i in enumerate(order[:n], 1):
            rows.append({"class": ps.CLASS_NAMES[lab], "kind": "most frequent",
                         "rank": rank, "word": vocab[i], "rel_freq": m[i]})
        for rank, i in enumerate(order[::-1][:n], 1):
            rows.append({"class": ps.CLASS_NAMES[lab], "kind": "rarest",
                         "rank": rank, "word": vocab[i], "rel_freq": m[i]})
    top = {lab: set(vocab[np.argsort(-means[lab])[:100]]) for lab in [0, 1]}
    bottom = {lab: set(vocab[np.argsort(means[lab])[:100]]) for lab in [0, 1]}
    allm = (means[0] + means[1]) / 2
    o = np.argsort(-allm)
    summary = {
        "k": k,
        "vocabulary_seen": int((X.sum(axis=0) > 0).sum()),
        "possible_words": 4 ** k,
        "top100_shared_between_classes": len(top[0] & top[1]),
        "bottom100_shared_between_classes": len(bottom[0] & bottom[1]),
        "share_with_CG_in_top100": np.mean(["CG" in w for w in vocab[o[:100]]]),
        "share_with_CG_in_bottom100": np.mean(["CG" in w for w in vocab[o[::-1][:100]]]),
        "share_with_CG_all_words": np.mean(["CG" in w for w in vocab]),
        "spearman_rank_corr_between_classes": pd.Series(means[0]).corr(
            pd.Series(means[1]), method="spearman"),
    }
    return pd.DataFrame(rows), summary, means, vocab


def differential(df, k):
    """Words whose per-gene frequency differs between classes."""
    X, vocab = word_counts(df.sequence.tolist(), k)
    R = relative(X)
    y = df.label.to_numpy()
    a, b = R[y == 1], R[y == 0]
    p = mannwhitneyu(a, b, axis=0).pvalue
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    eps = 1e-9
    out = pd.DataFrame({
        "word": vocab,
        "mean_freq_disease": ma, "mean_freq_non_disease": mb,
        "log2_ratio": np.log2((ma + eps) / (mb + eps)),
        "p_value": p, "q_value": bh_fdr(p),
    })
    if k == "codon":
        out.insert(1, "amino_acid", [CODON[w] for w in vocab])
    return out.sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------- figures

def plot_zipf(means, vocab, k):
    plt = ps.setup()
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab in [0, 1]:
        m = np.sort(means[lab])[::-1]
        m = m[m > 0]
        ax.loglog(np.arange(1, len(m) + 1), m, color=ps.CLASS_COLOURS[lab],
                  label=ps.CLASS_NAMES[lab], lw=2)
    ax.set_xlabel(f"rank of {k}-mer (most to least frequent)")
    ax.set_ylabel("mean relative frequency per gene")
    ax.set_title(f"Both classes use the same {k}-mer vocabulary")
    ax.legend()
    ax.grid(True, which="major")
    ax.grid(False, which="minor")
    return ps.save(fig, f"vocab_zipf_k{k}")


def plot_differential_codons(diff, n=8):
    plt = ps.setup()
    sig = diff[diff.q_value < 0.05]
    up = sig[sig.log2_ratio > 0].nlargest(n, "log2_ratio")
    down = sig[sig.log2_ratio < 0].nsmallest(n, "log2_ratio")
    show = pd.concat([down, up]).sort_values("log2_ratio")
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(show) + 1.2))
    colours = [ps.DISEASE if v > 0 else ps.NON_DISEASE for v in show.log2_ratio]
    ax.barh(range(len(show)), show.log2_ratio, color=colours, height=0.7)
    ax.set_yticks(range(len(show)),
                  [f"{w}  ({aa})" for w, aa in zip(show.word, show.amino_acid)])
    ax.axvline(0, color=ps.NEUTRAL, lw=1)
    ax.set_xlabel("log2(frequency in disease / non-disease)")
    ax.set_title("Codons that differ most (length & GC matched, FDR < 5%)")
    ax.grid(axis="y", visible=False)
    lim = np.abs(show.log2_ratio).max() * 1.15
    ax.set_xlim(-lim, lim)
    ax.text(lim * 0.97, -0.9, "more in disease genes →", ha="right",
            color=ps.TEXT_SECONDARY, fontsize=8)
    ax.text(-lim * 0.97, len(show) - 0.2, "← more in non-disease genes", ha="left",
            color=ps.TEXT_SECONDARY, fontsize=8)
    return ps.save(fig, "vocab_differential_codons")


def plot_volcano(diff, k):
    plt = ps.setup()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    q = diff.q_value.clip(lower=1e-300)
    sig = diff.q_value < 0.05
    ax.scatter(diff.log2_ratio[~sig], -np.log10(q[~sig]), s=8, color=ps.NEUTRAL,
               alpha=0.5, label="not significant", linewidths=0)
    for direction, col, name in [(1, ps.DISEASE, "more in disease"),
                                 (-1, ps.NON_DISEASE, "more in non-disease")]:
        m = sig & (np.sign(diff.log2_ratio) == direction)
        ax.scatter(diff.log2_ratio[m], -np.log10(q[m]), s=10, color=col,
                   label=f"{name} ({m.sum()})", linewidths=0)
    ax.axhline(-np.log10(0.05), color=ps.NEUTRAL, lw=1, ls="--")
    ax.set_xlabel("log2(frequency in disease / non-disease)")
    ax.set_ylabel("-log10(FDR q-value)")
    ax.set_title(f"Which {k}-mers differ between classes (length & GC matched)")
    ax.legend(loc="upper left", fontsize=8)
    return ps.save(fig, f"vocab_volcano_k{k}")


# ------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_disease_data()
    df["lenmatch"] = df.seq_id.isin(set(load_disease_splits("lenmatch").seq_id))
    lm = df[df.lenmatch].reset_index(drop=True)

    print("--- sequence statistics ---")
    stats = sequence_stats(df)
    stats.to_csv(OUT / "sequence_stats.csv", index=False)
    print(stats.to_string(index=False), "\n")

    print("--- frequent and rare words ---")
    summaries = []
    for k in [4, 6]:
        words, summary, means, vocab = frequent_and_rare(lm, k)
        words.to_csv(OUT / f"frequent_rare_k{k}.csv", index=False)
        summaries.append(summary)
        if k == 6:
            plot_zipf(means, vocab, k)
    summ = pd.DataFrame(summaries)
    summ.to_csv(OUT / "frequent_rare_summary.csv", index=False)
    print(summ.T.to_string(header=False), "\n")

    print("--- differential words (lenmatch subset) ---")
    for k in ["codon", 4, 6]:
        diff = differential(lm, k)
        diff.to_csv(OUT / f"differential_{k}.csv", index=False)
        sig = diff[diff.q_value < 0.05]
        print(f"k={k}: {len(sig)}/{len(diff)} words differ at FDR 5% "
              f"({(sig.log2_ratio > 0).sum()} up in disease, "
              f"{(sig.log2_ratio < 0).sum()} down)")
        print(diff.head(8)[["word", "log2_ratio", "q_value"] +
                           (["amino_acid"] if k == "codon" else [])].to_string(index=False))
        if k == "codon":
            plot_differential_codons(diff)
        if k == 6:
            plot_volcano(diff, k)
        print()
    print(f"wrote {OUT.relative_to(ROOT)}/ and figures/disease/vocab_*.png")


if __name__ == "__main__":
    main()
