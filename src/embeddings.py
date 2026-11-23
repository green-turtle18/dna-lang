"""
embeddings.py — word2vec for DNA (proposal task 3).
Owner: Leora.

Treats each gene as a sentence and trains word2vec (skip-gram) on its words:
  * codons   : in-frame, non-overlapping triplets (61 sense codons)
  * 6-mers   : overlapping, like dna2vec

Then asks the proposal's three questions:
  1. Do words used in similar contexts get similar vectors?
     For codons there's a natural "meaning" to test: synonymous codons
     encode the same amino acid. We compare cosine similarity of synonymous
     pairs with non-synonymous pairs, and with pairs that merely share the
     same third base type (G/C vs A/T), because GC-rich genes use all the
     G/C-ending codons together — similarity could come from that instead.
  2. Can a whole gene be one vector? Gene vector = mean of its word vectors
     (the DNA version of averaging word vectors into a sentence vector).
  3. Do those gene vectors classify disease genes as well as TF-IDF?

word2vec is trained on TRAINING genes only, separately for each split
partition, so test genes are never seen, not even without labels.

Run from the repo root (needs gensim: pip install gensim):
    python src/embeddings.py            # everything
    python src/embeddings.py --quick    # codons only, lenmatch only

Outputs:
    results/disease/results_leora_w2v.csv        (README contract 4)
    results/disease/embeddings/*.csv
    figures/disease/embedding_*.png
"""

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.disease_data import load_disease_split_data, CODON   # noqa: E402
from src.models_nn import oversample                          # noqa: E402
from src import plotstyle as ps                               # noqa: E402

RESULTS = ROOT / "results" / "disease" / "results_leora_w2v.csv"
OUT = ROOT / "results" / "disease" / "embeddings"
SEED = 0
SPLITS = ["random", "family", "lenmatch"]
COLUMNS = ["representation", "k", "model", "split_type", "accuracy", "macro_f1",
           "roc_auc", "n_features", "seed", "notes"]

# (k label, tokeniser, vector size, window, epochs)
TOKENISERS = {
    "codon": dict(dim=32, window=5, epochs=5),
    "6":     dict(dim=64, window=10, epochs=3),
}


def tokens(seq, k):
    if k == "codon":
        return [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
    k = int(k)
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def train_w2v(seqs, k, shuffle_order=False):
    """shuffle_order=True randomises word order inside each gene: same words,
    same genes, no local context. A control for what word2vec learns from."""
    from gensim.models import Word2Vec
    p = TOKENISERS[k]
    sents = [tokens(s, k) for s in seqs]
    if shuffle_order:
        rng = np.random.default_rng(SEED)
        sents = [list(rng.permutation(t)) for t in sents]
    return Word2Vec(
        sentences=sents,
        vector_size=p["dim"], window=p["window"], min_count=1,
        sg=1, negative=5, epochs=p["epochs"],
        seed=SEED, workers=1,           # one worker = reproducible
    )


def gene_vectors(w2v, seqs, k):
    wv = w2v.wv
    out = np.zeros((len(seqs), wv.vector_size))
    for i, s in enumerate(seqs):
        toks = [t for t in tokens(s, k) if t in wv.key_to_index]
        if toks:
            out[i] = wv[toks].mean(axis=0)
    return out


# --------------------------------------------------- 1. semantic structure

def codon_similarity(w2v):
    """Cosine similarity for every pair of sense codons, with pair types."""
    wv = w2v.wv
    sense = [c for c in sorted(CODON) if CODON[c] != "*" and c in wv.key_to_index]
    rows = []
    for a, b in combinations(sense, 2):
        rows.append({
            "codon_a": a, "codon_b": b,
            "synonymous": CODON[a] == CODON[b],
            "same_third_base_type": (a[2] in "GC") == (b[2] in "GC"),
            "cosine": float(wv.similarity(a, b)),
        })
    return pd.DataFrame(rows)


def semantic_tests(sim, n_perm=2000, seed=SEED):
    """Is synonym similarity real, and is it more than third-base GC?"""
    rng = np.random.default_rng(seed)
    syn = sim[sim.synonymous]
    non = sim[~sim.synonymous]
    observed = syn.cosine.mean() - non.cosine.mean()

    # permutation: shuffle which codons belong to which amino acid,
    # keeping the number of codons per amino acid fixed
    codons = sorted(set(sim.codon_a) | set(sim.codon_b))
    aas = [CODON[c] for c in codons]
    lookup = {(a, b): v for a, b, v in zip(sim.codon_a, sim.codon_b, sim.cosine)}
    perm = []
    for _ in range(n_perm):
        shuffled = dict(zip(codons, rng.permutation(aas)))
        same = np.array([shuffled[a] == shuffled[b] for a, b in lookup])
        vals = np.array(list(lookup.values()))
        perm.append(vals[same].mean() - vals[~same].mean())
    p_perm = (np.sum(np.array(perm) >= observed) + 1) / (n_perm + 1)

    # within the same third-base type, are synonyms still closer?
    same3 = sim[sim.same_third_base_type]
    p_within = mannwhitneyu(same3[same3.synonymous].cosine,
                            same3[~same3.synonymous].cosine,
                            alternative="greater").pvalue
    groups = sim.assign(group=np.select(
        [sim.synonymous & sim.same_third_base_type,
         sim.synonymous & ~sim.same_third_base_type,
         ~sim.synonymous & sim.same_third_base_type],
        ["synonymous, same 3rd-base type", "synonymous, different 3rd-base type",
         "not synonymous, same 3rd-base type"],
        default="not synonymous, different 3rd-base type"))
    table = groups.groupby("group").cosine.agg(["count", "mean", "median"]).round(4)

    # nearest neighbour of each codon: is it a synonym?
    nn_syn, nn_same3 = [], []
    for c in codons:
        sub = sim[(sim.codon_a == c) | (sim.codon_b == c)]
        best = sub.loc[sub.cosine.idxmax()]
        nn_syn.append(bool(best.synonymous))
        nn_same3.append(bool(best.same_third_base_type))
    has_syn = [c for c in codons if sum(CODON[x] == CODON[c] for x in codons) > 1]
    expected_nn_syn = np.mean([(sum(CODON[x] == CODON[c] for x in codons) - 1) /
                               (len(codons) - 1) for c in has_syn])
    summary = {
        "mean_cosine_synonymous": round(syn.cosine.mean(), 4),
        "mean_cosine_non_synonymous": round(non.cosine.mean(), 4),
        "difference": round(observed, 4),
        "permutation_p_value": round(p_perm, 5),
        "synonyms_closer_within_same_3rd_base_type_p": float(f"{p_within:.3g}"),
        "nearest_neighbour_is_synonym": round(float(np.mean(nn_syn)), 3),
        "nearest_neighbour_is_synonym_by_chance": round(float(expected_nn_syn), 3),
        "nearest_neighbour_same_3rd_base_type": round(float(np.mean(nn_same3)), 3),
    }
    return summary, table.reset_index()


# --------------------------------------------------- 3. gene-vector models

def classify(Xtr, ytr, Xte, yte, kind):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "logreg":
        m = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
        m.fit(Xtr, ytr)
    elif kind == "rf":
        m = RandomForestClassifier(200, n_jobs=-1, random_state=SEED,
                                   class_weight="balanced", min_samples_leaf=5)
        m.fit(Xtr, ytr)
    else:
        m = MLPClassifier((128,), alpha=1e-4, early_stopping=True,
                          validation_fraction=0.1, n_iter_no_change=10,
                          max_iter=200, random_state=SEED)
        m.fit(*oversample(Xtr, ytr))
    pred = m.predict(Xte)
    prob = m.predict_proba(Xte)[:, 1]
    return {"accuracy": round(accuracy_score(yte, pred), 4),
            "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
            "roc_auc": round(roc_auc_score(yte, prob), 4)}


# ---------------------------------------------------------------- figures

def plot_codon_space(w2v):
    plt = ps.setup()
    wv = w2v.wv
    sense = [c for c in sorted(CODON) if CODON[c] != "*" and c in wv.key_to_index]
    xy = PCA(2, random_state=SEED).fit_transform(wv[sense])
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for gc3, col, name in [(True, ps.THIRD, "ends in G or C"),
                           (False, ps.TEXT_SECONDARY, "ends in A or T")]:
        m = np.array([(c[2] in "GC") == gc3 for c in sense])
        ax.scatter(xy[m, 0], xy[m, 1], s=42, color=col, label=name,
                   edgecolor="white", linewidth=1.2, zorder=3)
    texts = [ax.text(x, y, f"{c} {CODON[c]}", fontsize=6.5, color=ps.TEXT, zorder=4)
             for (x, y), c in zip(xy, sense)]
    try:                                           # optional: nicer label placement
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=ps.NEUTRAL, lw=0.5))
    except ImportError:
        pass
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Learned codon space: the main axis is third-base GC")
    ax.legend(loc="best", fontsize=8)
    ax.text(0, -0.1, "word2vec codon vectors (32-d), first 2 principal components. "
            "Label = codon and the amino acid it encodes.",
            transform=ax.transAxes, fontsize=7.5, color=ps.TEXT_SECONDARY)
    return ps.save(fig, "embedding_codon_space")


def plot_similarity_heatmap(w2v):
    """61x61 cosine similarity, codons ordered by amino acid. Synonym blocks
    on the diagonal = the genetic code, learned from context alone."""
    plt = ps.setup()
    from matplotlib.colors import LinearSegmentedColormap
    wv = w2v.wv
    sense = sorted((c for c in CODON if CODON[c] != "*" and c in wv.key_to_index),
                   key=lambda c: (CODON[c], c))
    S = np.array([[wv.similarity(a, b) for b in sense] for a in sense])
    np.fill_diagonal(S, np.nan)
    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(S, cmap=cmap, vmin=np.nanpercentile(S, 2), vmax=np.nanmax(S))
    ax.grid(False)
    labels = [f"{CODON[c]} {c}" for c in sense]
    ax.set_xticks(range(len(sense)), labels, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(sense)), labels, fontsize=5.5)
    start = 0                                       # outline each amino acid block
    for aa in sorted(set(CODON[c] for c in sense)):
        n = sum(CODON[c] == aa for c in sense)
        if n > 1:
            ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), n, n, fill=False,
                                       edgecolor=ps.TEXT, lw=1.0))
        start += n
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("cosine similarity", color=ps.TEXT_SECONDARY)
    ax.set_title("Codon similarity learned by word2vec, ordered by amino acid")
    ax.text(0, -0.09, "Outlined squares = codons for the same amino acid. word2vec was "
            "never told the genetic code; darker blocks mean it learned it from context.",
            transform=ax.transAxes, fontsize=7.5, color=ps.TEXT_SECONDARY)
    return ps.save(fig, "embedding_codon_similarity_heatmap")


def plot_similarity_groups(table):
    plt = ps.setup()
    order = ["synonymous, same 3rd-base type", "synonymous, different 3rd-base type",
             "not synonymous, same 3rd-base type", "not synonymous, different 3rd-base type"]
    t = table.set_index("group").reindex(order).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    cols = [ps.THIRD if g.startswith("synonymous") else ps.NEUTRAL for g in t.index]
    ax.barh(range(len(t)), t["mean"], color=cols, height=0.62)
    for i, (v, n) in enumerate(zip(t["mean"], t["count"])):
        ax.text(v + 0.01, i, f"{v:.2f}  (n={n})", va="center", fontsize=8,
                color=ps.TEXT_SECONDARY)
    ax.set_yticks(range(len(t)), t.index)
    ax.set_xlabel("mean cosine similarity between codon vectors")
    ax.set_title("Synonymous codons get similar vectors, beyond third-base GC")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(t["mean"]) * 1.35)
    return ps.save(fig, "embedding_similarity_groups")


# ------------------------------------------------------------------- main

def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    splits = ["lenmatch"] if quick else SPLITS
    ks = ["codon"] if quick else list(TOKENISERS)
    rows, cache = [], {}
    for split in splits:
        tr, te = load_disease_split_data(split)
        # lenmatch reuses the family partition, so reuse its word2vec model
        part = "family" if split == "lenmatch" else split
        for k in ks:
            if (part, k) not in cache:
                ref_tr, _ = load_disease_split_data(part)
                t0 = time.time()
                cache[(part, k)] = train_w2v(ref_tr.sequence.tolist(), k)
                print(f"trained word2vec k={k} on {part} train "
                      f"({len(ref_tr)} genes) in {time.time() - t0:.0f}s", flush=True)
            w2v = cache[(part, k)]
            Xtr = gene_vectors(w2v, tr.sequence.tolist(), k)
            Xte = gene_vectors(w2v, te.sequence.tolist(), k)
            for kind in ["logreg", "rf", "mlp"]:
                t0 = time.time()
                r = classify(Xtr, tr.label.to_numpy(), Xte, te.label.to_numpy(), kind)
                rows.append({"representation": "w2v_mean", "k": k, "model": kind,
                             "split_type": split, **r, "n_features": Xtr.shape[1],
                             "seed": SEED, "notes": f"mean of word vectors; {time.time() - t0:.0f}s"})
                print(f"{split:8s} k={k:5s} {kind:6s} AUC {r['roc_auc']:.3f}  "
                      f"macroF1 {r['macro_f1']:.3f}", flush=True)
                RESULTS.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows)[COLUMNS].to_csv(RESULTS, index=False)

    print("\n--- do synonymous codons get similar vectors? ---")
    w2v = cache[("family", "codon")]
    w2v.save(str(OUT / "word2vec_codon_family_train.model"))
    sim = codon_similarity(w2v)
    sim.to_csv(OUT / "codon_pair_similarity.csv", index=False)
    summary, table = semantic_tests(sim)
    table.to_csv(OUT / "codon_similarity_groups.csv", index=False)
    for key, v in summary.items():
        print(f"{key:48s} {v}")
    print(table.to_string(index=False))

    # context control: same genes, codon order shuffled inside each gene
    print("\n--- control: codon order shuffled within each gene ---")
    fam_tr, _ = load_disease_split_data("family")
    w2v_shuf = train_w2v(fam_tr.sequence.tolist(), "codon", shuffle_order=True)
    summary_shuf, table_shuf = semantic_tests(codon_similarity(w2v_shuf))
    both = pd.DataFrame({"real codon order": summary,
                         "codon order shuffled": summary_shuf})
    both.to_csv(OUT / "codon_semantics_summary.csv")
    table_shuf.to_csv(OUT / "codon_similarity_groups_shuffled_order.csv", index=False)
    print(both.to_string())
    plot_codon_space(w2v)
    plot_similarity_heatmap(w2v)
    plot_similarity_groups(table)
    print(f"\nwrote {RESULTS.relative_to(ROOT)}, {OUT.relative_to(ROOT)}/, "
          "figures/disease/embedding_*.png")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
