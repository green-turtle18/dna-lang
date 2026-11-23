"""
interpret.py — what have the models actually learned? (proposal task 5)
Owner: Leora.

Three questions, all on the lenmatch split (length and GC matched, paralog
families kept apart), with the family split alongside for reference:

  1. Which features carry the signal?  Train the same model on different
     feature sets, each isolating one kind of information:
       length+GC             -> composition control (should be chance)
       CpG obs/exp           -> one number per gene
       dinucleotides (16)    -> local base pairs, includes CpG
       amino-acid comp. (20) -> PROTEIN only, no codon choice
       synonymous codon use  -> codon CHOICE only, protein removed (RSCU)
       codon usage (61)      -> protein + codon choice
       codons, synonyms shuffled -> protein kept, codon choice randomised
     If amino-acid composition alone matches codon usage, the signal is in
     the protein the gene encodes, not in DNA "spelling".

  2. Which codons/k-mers does the codon model weight most?

  3. Is it just gene families? Zinc fingers, olfactory receptors and
     keratin-associated proteins are almost all non-disease. Drop them and
     re-test.

Every AUC comes with a 95% bootstrap confidence interval on the test set,
so you can tell which gaps mean something.

Run from the repo root:
    python src/interpret.py

Outputs: results/disease/interpret/*.csv, figures/disease/interpret_*.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.disease_data import load_disease_split_data, CODON, translate  # noqa: E402
from src import plotstyle as ps                                       # noqa: E402

OUT = ROOT / "results" / "disease" / "interpret"
SEED = 0
N_BOOT = 1000

SENSE = sorted(c for c, a in CODON.items() if a != "*")
AMINO = sorted(set(CODON[c] for c in SENSE))
SYN = {a: [c for c in SENSE if CODON[c] == a] for a in AMINO}
MULTI = [a for a in AMINO if len(SYN[a]) > 1]          # excludes M, W
DINUC = [a + b for a in "ACGT" for b in "ACGT"]
FAMILY_PATTERN = r"^ZNF|^ZSCAN|^ZFP|^ZKSCAN|^ZIK|^ZBTB|^OR\d|^KRTAP"


# --------------------------------------------------------------- features

def codon_counts(seqs):
    idx = {c: i for i, c in enumerate(SENSE)}
    X = np.zeros((len(seqs), len(SENSE)))
    for r, s in enumerate(seqs):
        for i in range(0, len(s) - 2, 3):
            j = idx.get(s[i:i + 3])
            if j is not None:
                X[r, j] += 1
    return X


def f_codon_usage(seqs):
    X = codon_counts(seqs)
    return X / X.sum(axis=1, keepdims=True), SENSE


def f_amino_acids(seqs):
    X = codon_counts(seqs)
    A = np.stack([X[:, [SENSE.index(c) for c in SYN[a]]].sum(axis=1) for a in AMINO], 1)
    return A / A.sum(axis=1, keepdims=True), AMINO


def f_rscu(seqs):
    """Share of each codon within its amino acid's synonyms. Protein-free:
    a gene's amino-acid content does not change these numbers."""
    X = codon_counts(seqs)
    cols, names = [], []
    for a in MULTI:
        idx = [SENSE.index(c) for c in SYN[a]]
        tot = X[:, idx].sum(axis=1, keepdims=True)
        share = np.divide(X[:, idx], tot, out=np.full((len(X), len(idx)), np.nan),
                          where=tot > 0)
        cols.append(share)
        names += SYN[a]
    R = np.hstack(cols)
    col_mean = np.nanmean(R, axis=0)
    return np.where(np.isnan(R), col_mean, R), names


def f_dinucleotides(seqs):
    X = np.array([[s.count(d) for d in DINUC] for s in seqs], float)
    return X / X.sum(axis=1, keepdims=True), DINUC


def f_cpg(seqs):
    v = []
    for s in seqs:
        c, g = s.count("C"), s.count("G")
        v.append(s.count("CG") * len(s) / (c * g) if c and g else 0.0)
    return np.array(v)[:, None], ["CpG_obs_exp"]


def synonymous_shuffle(seqs, usage, seed=SEED):
    """Replace every codon by a random synonym drawn from genome-wide usage
    within its amino acid. The protein is unchanged; codon choice is noise."""
    rng = np.random.default_rng(seed)
    probs = {a: np.array([usage[c] for c in SYN[a]]) / sum(usage[c] for c in SYN[a])
             for a in AMINO}
    out = []
    for s in seqs:
        cods = [s[i:i + 3] for i in range(0, len(s) - 2, 3)]
        new = []
        for c in cods:
            a = CODON[c]
            if a == "*" or len(SYN[a]) == 1:
                new.append(c)
            else:
                new.append(SYN[a][rng.choice(len(SYN[a]), p=probs[a])])
        out.append("".join(new))
    return out


# ------------------------------------------------------------------ model

def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, C=0.1,
                                            class_weight="balanced",
                                            random_state=SEED))


def auc_ci(y, s, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def evaluate(Xtr, ytr, Xte, yte):
    m = model().fit(Xtr, ytr)
    return m, auc_ci(yte, m.predict_proba(Xte)[:, 1])


# --------------------------------------------------------------- analyses

def feature_sets(split):
    tr, te = load_disease_split_data(split)
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()
    trs, tes = tr.sequence.tolist(), te.sequence.tolist()

    cu_tr, _ = f_codon_usage(trs)
    usage = dict(zip(SENSE, cu_tr.mean(axis=0)))
    shuf_tr, shuf_te = synonymous_shuffle(trs, usage, SEED), synonymous_shuffle(tes, usage, SEED + 1)

    sets = {
        "length + GC (control)": (np.c_[np.log10(tr.length), tr.gc],
                                  np.c_[np.log10(te.length), te.gc]),
        "CpG obs/exp (1)": (f_cpg(trs)[0], f_cpg(tes)[0]),
        "dinucleotides (16)": (f_dinucleotides(trs)[0], f_dinucleotides(tes)[0]),
        "synonymous codon choice only (59)": (f_rscu(trs)[0], f_rscu(tes)[0]),
        "amino-acid composition (20)": (f_amino_acids(trs)[0], f_amino_acids(tes)[0]),
        "codon usage (61)": (cu_tr, f_codon_usage(tes)[0]),
        "codon usage, synonyms shuffled (61)": (f_codon_usage(shuf_tr)[0],
                                                f_codon_usage(shuf_te)[0]),
        "amino acids + CpG (21)": (np.c_[f_amino_acids(trs)[0], f_cpg(trs)[0]],
                                   np.c_[f_amino_acids(tes)[0], f_cpg(tes)[0]]),
    }
    rows = []
    for name, (Xtr, Xte) in sets.items():
        _, (auc, lo, hi) = evaluate(Xtr, ytr, Xte, yte)
        rows.append({"split": split, "feature_set": name, "n_features": Xtr.shape[1],
                     "roc_auc": round(auc, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
        print(f"{split:8s} {name:38s} AUC {auc:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)
    return rows


def top_codons(split="lenmatch", n=12):
    tr, te = load_disease_split_data(split)
    Xtr, names = f_codon_usage(tr.sequence.tolist())
    m = model().fit(Xtr, tr.label)
    coef = m[-1].coef_[0]
    out = pd.DataFrame({"codon": names, "amino_acid": [CODON[c] for c in names],
                        "contains_CG": ["CG" in c for c in names],
                        "coefficient": coef}).sort_values("coefficient", ascending=False)
    out["direction"] = np.where(out.coefficient > 0, "disease", "non-disease")
    return out.reset_index(drop=True)


def family_confound(split="lenmatch"):
    tr, te = load_disease_split_data(split)
    rows = []
    for name, keep in [("all genes", lambda d: d.index == d.index),
                       ("without zinc-finger / OR / KRTAP families",
                        lambda d: ~d.gene_symbol.str.contains(FAMILY_PATTERN))]:
        a, b = tr[keep(tr)], te[keep(te)]
        for fs, fn in [("codon usage (61)", f_codon_usage),
                       ("amino-acid composition (20)", f_amino_acids)]:
            _, (auc, lo, hi) = evaluate(fn(a.sequence.tolist())[0], a.label,
                                        fn(b.sequence.tolist())[0], b.label)
            rows.append({"split": split, "genes": name, "feature_set": fs,
                         "n_train": len(a), "n_test": len(b),
                         "roc_auc": round(auc, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4)})
            print(f"{name:42s} {fs:28s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  n_test={len(b)}")
    return pd.DataFrame(rows)


def amino_acid_differences(split="lenmatch"):
    """Mean amino-acid share per class across the whole split (descriptive)."""
    tr, te = load_disease_split_data(split)
    d = pd.concat([tr, te])
    A, names = f_amino_acids(d.sequence.tolist())
    y = d.label.to_numpy()
    from scipy.stats import mannwhitneyu
    p = mannwhitneyu(A[y == 1], A[y == 0], axis=0).pvalue
    return pd.DataFrame({"amino_acid": names,
                         "share_disease": A[y == 1].mean(0),
                         "share_non_disease": A[y == 0].mean(0),
                         "log2_ratio": np.log2(A[y == 1].mean(0) / A[y == 0].mean(0)),
                         "p_value": p}).sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------- figures

def plot_feature_sets(df):
    plt = ps.setup()
    d = df[df.split == "lenmatch"].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(d) + 1.3))
    for i, r in d.iterrows():
        control = "control" in r.feature_set
        col = ps.NEUTRAL if control else ps.DISEASE
        ax.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.roc_auc, i, "o", color=col, ms=8, mec="white", mew=1.5)
        ax.text(r.ci_high + 0.004, i, f"{r.roc_auc:.3f}", va="center",
                color=ps.TEXT_SECONDARY, fontsize=8)
    ax.axvline(0.5, color=ps.NEUTRAL, lw=1, ls="--")
    ax.text(0.5, len(d) - 0.35, "chance", color=ps.TEXT_SECONDARY, fontsize=8, ha="center")
    ax.set_yticks(range(len(d)), d.feature_set)
    ax.set_xlabel("ROC-AUC on lenmatch test set (95% bootstrap CI)")
    ax.set_title("Protein content and codon choice each carry part of the signal")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(d) - 0.1)
    return ps.save(fig, "interpret_feature_sets")


def plot_top_codons(coef, n=10):
    plt = ps.setup()
    show = pd.concat([coef.head(n), coef.tail(n)]).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(show) + 1.2))
    colours = [ps.DISEASE if v > 0 else ps.NON_DISEASE for v in show.coefficient]
    ax.barh(range(len(show)), show.coefficient, color=colours, height=0.7)
    ax.set_yticks(range(len(show)), [f"{c} ({a}){'  CG' if cg else ''}" for c, a, cg in
                                     zip(show.codon, show.amino_acid, show.contains_CG)])
    ax.axvline(0, color=ps.NEUTRAL, lw=1)
    ax.set_xlabel("logistic-regression weight (standardised codon frequency)")
    ax.set_title("Codons the model relies on (lenmatch)")
    ax.grid(axis="y", visible=False)
    return ps.save(fig, "interpret_top_codons")


# ------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("--- 1. which features carry the signal ---")
    fs = pd.DataFrame(feature_sets("lenmatch") + feature_sets("family"))
    fs.to_csv(OUT / "feature_sets.csv", index=False)
    plot_feature_sets(fs)

    print("\n--- 2. codons the model weights most (lenmatch) ---")
    coef = top_codons()
    coef.to_csv(OUT / "codon_coefficients.csv", index=False)
    print(pd.concat([coef.head(8), coef.tail(8)]).to_string(index=False))
    plot_top_codons(coef)

    aa = amino_acid_differences()
    aa.to_csv(OUT / "amino_acid_differences.csv", index=False)
    print("\namino acids differing most:\n", aa.head(10).round(5).to_string(index=False))

    print("\n--- 3. is it just gene families? ---")
    fam = family_confound()
    fam.to_csv(OUT / "family_confound.csv", index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}/ and figures/disease/interpret_*.png")


if __name__ == "__main__":
    main()
