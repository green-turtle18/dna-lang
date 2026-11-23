"""
verify_claims.py — check AI-generated explanations against the data
(proposal task 6).
Owner: Leora.

Chatbots asked to explain our results make claims like "disease genes are
CpG-rich because CpG sites mutate often" or "disease genes have more repeats".
Each claim here is turned into a number computed per gene and tested on the
lenmatch subset (disease vs non-disease, length and GC matched, 8,902 genes):

    direction   : is the claimed difference in the claimed direction?
    p / q       : Mann-Whitney U test, Benjamini-Hochberg FDR over all claims
    effect      : log2(mean in disease / mean in non-disease)
    auc         : how well that one number alone separates the classes
    verdict     : SUPPORTED, OPPOSITE, or NOT SUPPORTED (q >= 0.05)

To test a new claim from a chatbot, add a function to CLAIMS at the bottom:
    "short claim text": (feature_function, expected_direction)
where feature_function(seq) -> float, and expected_direction is
+1 (higher in disease) or -1 (lower in disease).

Run from the repo root:
    python src/verify_claims.py

Output: results/disease/ai_claims/claim_tests.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.disease_data import load_disease_data, load_disease_splits, CODON, translate  # noqa: E402

OUT = ROOT / "results" / "disease" / "ai_claims"


# ------------------------------------------------------------ helpers

def codons(s):
    return [s[i:i + 3] for i in range(0, len(s) - 2, 3)]


def protein(s):
    return translate(s)


def aa_share(letters):
    def f(s):
        p = protein(s)
        return sum(p.count(a) for a in letters) / len(p)
    return f


def per_kb(count):
    return lambda s: count(s) * 1000 / len(s)


# ------------------------------------------------------------ features

def cpg_obs_exp(s):
    c, g = s.count("C"), s.count("G")
    return s.count("CG") * len(s) / (c * g) if c and g else 0.0


def dinuc_share(d):
    def f(s):
        return s.count(d) / (len(s) - 1)    # str.count is non-overlapping; fine
    return f                                # for heterodinucleotides like TG, CA


def cgn_arginine_share(s):
    """Share of arginine codons that start with CG (CGA/CGC/CGG/CGT)."""
    arg = [c for c in codons(s) if CODON.get(c) == "R"]
    return sum(c.startswith("CG") for c in arg) / len(arg) if arg else np.nan


def cga_per_kb(s):
    """CGA is one C->T mutation away from TGA (stop): a nonsense hotspot."""
    return codons(s).count("CGA") * 1000 / len(s)


STOP_ONE_TRANSITION = {"CGA", "CAG", "CAA", "TGG"}   # C->T or G->A gives a stop


def stop_gain_codons_share(s):
    c = codons(s)
    return sum(x in STOP_ONE_TRANSITION for x in c) / len(c)


def gc_at(pos):
    def f(s):
        t = s[pos::3]
        return (t.count("G") + t.count("C")) / len(t)
    return f


gc3 = gc_at(2)


def longest_aa_run(s):
    """Longest run of one amino acid (poly-Q, poly-A, poly-E ...)."""
    return max(len(m.group(0)) for m in re.finditer(r"(.)\1*", protein(s)))


def has_aa_run_5(s):
    return float(longest_aa_run(s) >= 5)


def cag_repeat(s):
    """Longest in-frame run of CAG/CAA (glutamine codons)."""
    best = cur = 0
    for c in codons(s):
        cur = cur + 1 if c in ("CAG", "CAA") else 0
        best = max(best, cur)
    return best


def low_complexity_share(s, w=12, max_distinct=3):
    """Share of protein windows using <= 3 distinct amino acids."""
    p = protein(s)
    if len(p) < w:
        return 0.0
    return np.mean([len(set(p[i:i + w])) <= max_distinct for i in range(len(p) - w + 1)])


def collagen_gxy(s):
    """Gly-X-Y repeats per 100 residues (collagen signature)."""
    return len(re.findall(r"(?=G..G..G..G)", protein(s))) * 100 / len(protein(s))


def zinc_finger_motifs(s):
    """C2H2 zinc-finger motifs C-x(2,4)-C-x(12)-H-x(3,5)-H per 100 residues."""
    return len(re.findall(r"C.{2,4}C.{12}H.{3,5}H", protein(s))) * 100 / len(protein(s))


def hydrophobic_share(s):
    return aa_share("AILMFVW")(s)


def transmembrane_like(s, w=19, cut=0.75):
    """Any 19-residue window that is >= 75% hydrophobic (crude TM helix)."""
    p = protein(s)
    hyd = np.array([a in "AILMFVWC" for a in p], float)
    if len(hyd) < w:
        return 0.0
    win = np.convolve(hyd, np.ones(w) / w, mode="valid")
    return float((win >= cut).any())


def rare_codon_share(s, rare=("CGA", "TCG", "CCG", "ACG", "GCG", "TTA", "CTA", "ATA")):
    c = codons(s)
    return sum(x in rare for x in c) / len(c)


# claim text -> (function, expected direction: +1 higher in disease, -1 lower)
CLAIMS = {
    "Disease genes are less CpG-depleted (more CpG = mutation hotspots)": (cpg_obs_exp, +1),
    "Disease genes use CG-starting arginine codons more (CGN vs AGR)": (cgn_arginine_share, +1),
    "Disease genes have more CGA codons (one C->T from a stop codon)": (cga_per_kb, +1),
    "Disease genes have more codons one transition from a stop": (stop_gain_codons_share, +1),
    "Disease genes differ in GC at codon position 3": (gc3, +1),
    "Disease genes have higher GC3 (GC-biased gene conversion)": (gc3, +1),
    "Disease genes differ in GC at codon position 1": (gc_at(0), +1),
    "Disease genes differ in GC at codon position 2": (gc_at(1), +1),
    "Disease genes have fewer cysteines (zinc-finger families are non-disease)": (aa_share("C"), -1),
    "Disease genes have fewer C2H2 zinc-finger motifs": (zinc_finger_motifs, -1),
    "Disease genes have more hydrophobic residues": (hydrophobic_share, +1),
    "Disease genes are more often membrane proteins (TM-like segment)": (transmembrane_like, +1),
    "Disease genes have more charged residues (D, E, K, R)": (aa_share("DEKR"), +1),
    "Disease genes have fewer serines": (aa_share("S"), -1),
    "Disease genes have more glycine (structural proteins, collagen)": (aa_share("G"), +1),
    "Disease genes have more collagen-like Gly-X-Y repeats": (collagen_gxy, +1),
    "Disease genes have longer single-amino-acid runs (poly-Q etc.)": (longest_aa_run, +1),
    "Disease genes more often have a run of >= 5 identical amino acids": (has_aa_run_5, +1),
    "Disease genes have longer CAG/CAA (glutamine) codon runs": (cag_repeat, +1),
    "Disease genes have more low-complexity protein regions": (low_complexity_share, +1),
    "Disease genes use more rare codons": (rare_codon_share, +1),
    "Disease genes have fewer TpG (CpG deamination product)": (dinuc_share("TG"), -1),
    "Disease genes have fewer CpA (CpG deamination product)": (dinuc_share("CA"), -1),
}


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    r = p[o] * len(p) / np.arange(1, len(p) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(len(p))
    out[o] = np.minimum(r, 1)
    return out


def composition_control():
    """ChatGPT's 'most important experiment': control for GC1, GC2, GC3,
    CpG obs/exp and all 16 dinucleotides, then retrain the codon model.

    Three models on the lenmatch split (logistic regression, 95% bootstrap CI):
      composition only      : the 20 composition features
      codon usage           : 61 codon frequencies, as before
      codon usage, residual : each codon frequency with the part a linear
                              model can predict from composition removed
                              (fitted on TRAIN only), so what's left is
                              codon information composition can't explain
    """
    from sklearn.linear_model import LinearRegression
    from src.disease_data import load_disease_split_data
    from src.interpret import (f_codon_usage, f_dinucleotides, f_cpg, model, auc_ci)

    tr, te = load_disease_split_data("lenmatch")

    def comp(seqs):
        pos = np.array([[gc_at(i)(x) for i in range(3)] for x in seqs])
        return np.hstack([pos, f_cpg(seqs)[0], f_dinucleotides(seqs)[0]])

    Ctr, Cte = comp(tr.sequence.tolist()), comp(te.sequence.tolist())
    Xtr, Xte = f_codon_usage(tr.sequence.tolist())[0], f_codon_usage(te.sequence.tolist())[0]
    reg = LinearRegression().fit(Ctr, Xtr)
    Rtr, Rte = Xtr - reg.predict(Ctr), Xte - reg.predict(Cte)
    rows = []
    for name, (a, b) in {
        "composition only (GC1-3, CpG, 16 dinucleotides)": (Ctr, Cte),
        "codon usage": (Xtr, Xte),
        "codon usage, composition regressed out": (Rtr, Rte),
        "codon usage + composition": (np.hstack([Xtr, Ctr]), np.hstack([Xte, Cte])),
    }.items():
        m = model().fit(a, tr.label)
        auc, lo, hi = auc_ci(te.label, m.predict_proba(b)[:, 1])
        rows.append({"model": name, "n_features": a.shape[1], "roc_auc": round(auc, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
        print(f"{name:50s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "composition_control.csv", index=False)
    return out


def enriched_codon_gc():
    """Are the codons enriched in disease genes the GC-rich ones?"""
    path = ROOT / "results" / "disease" / "vocab" / "differential_codon.csv"
    d = pd.read_csv(path)
    d = d[d.q_value < 0.05].copy()
    d["codon_gc"] = d.word.map(lambda c: (c.count("G") + c.count("C")) / 3)
    d["third_base_gc"] = d.word.str[2].isin(["G", "C"])
    up, down = d[d.log2_ratio > 0], d[d.log2_ratio < 0]
    p = mannwhitneyu(up.codon_gc, down.codon_gc).pvalue
    res = {"n_up_in_disease": len(up), "n_down_in_disease": len(down),
           "mean_gc_up": round(up.codon_gc.mean(), 3),
           "mean_gc_down": round(down.codon_gc.mean(), 3),
           "share_third_base_gc_up": round(up.third_base_gc.mean(), 3),
           "share_third_base_gc_down": round(down.third_base_gc.mean(), 3),
           "p_value_gc_up_vs_down": round(p, 4)}
    pd.Series(res).to_csv(OUT / "enriched_codon_gc.csv", header=["value"])
    for k, v in res.items():
        print(f"{k:28s} {v}")
    return res


def _effect(v, y):
    a, b = v[y == 1], v[y == 0]
    return np.log2(a.mean() / b.mean()), mannwhitneyu(a, b).pvalue


def _cpg_oe(s):
    c, g = s.count("C"), s.count("G")
    return s.count("CG") * len(s) / (c * g) if c and g and len(s) > 1 else np.nan


def cpg_by_position(df):
    """Gemini: CpG excess should be stronger at codon positions 1-2 (changes
    the protein) than where it touches the synonymous third position, and
    concentrated in the first ~500 bp (CpG-island overlap), fading
    downstream."""
    y = df.label.to_numpy()
    rows = []

    def frame_cpg(s, start):            # CG starting at codon position `start`
        n = sum(1 for i in range(start, len(s) - 1, 3) if s[i:i + 2] == "CG")
        return n * 1000 / len(s)

    for name, fn in [("CpG at codon positions 1-2 (non-synonymous)", lambda s: frame_cpg(s, 0)),
                     ("CpG at codon positions 2-3", lambda s: frame_cpg(s, 1)),
                     ("CpG at position 3 -> next codon 1", lambda s: frame_cpg(s, 2)),
                     ("CpG obs/exp, first 500 bp", lambda s: _cpg_oe(s[:500])),
                     ("CpG obs/exp, after 500 bp", lambda s: _cpg_oe(s[500:])),
                     ("CpG obs/exp, last 500 bp", lambda s: _cpg_oe(s[-500:]))]:
        v = df.sequence.map(fn).to_numpy(float)
        ok = ~np.isnan(v)
        eff, p = _effect(v[ok], y[ok])
        rows.append({"measure": name, "mean_disease": v[ok & (y == 1)].mean(),
                     "mean_non_disease": v[ok & (y == 0)].mean(),
                     "log2_effect": eff, "p_value": p, "n": int(ok.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cpg_by_position.csv", index=False)
    print(out.round(5).to_string(index=False))
    return out


def paralog_families(df_all, lm_ids):
    """Gemini: disease genes should have fewer close paralogs. Family size
    comes from the homology groups in disease_data.py (all genes)."""
    from src.disease_data import load_disease_data
    full = load_disease_data(include_ambiguous=True)
    size = full.group_id.map(full.group_id.value_counts())
    d = df_all.assign(family_size=df_all.group_id.map(
        full.groupby("group_id").size()))
    d = d[d.seq_id.isin(lm_ids)]
    y = d.label.to_numpy()
    multi = (d.family_size > 1).to_numpy(float)
    rows = []
    for name, v in [("share in a multi-gene family", multi),
                    ("family size", d.family_size.to_numpy(float))]:
        eff, p = _effect(v, y)
        rows.append({"measure": name, "mean_disease": v[y == 1].mean(),
                     "mean_non_disease": v[y == 0].mean(), "log2_effect": eff, "p_value": p})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "paralog_families.csv", index=False)
    print(out.round(5).to_string(index=False))
    return out


def family_removal():
    """Gemini: removing zinc fingers, olfactory receptors, keratin-associated
    proteins and collagens should sharply cut the Cys/Gly effects and the
    amino-acid-only AUC."""
    from src.disease_data import load_disease_split_data
    from src.interpret import f_amino_acids, model, auc_ci, AMINO
    tr, te = load_disease_split_data("lenmatch")
    pattern = r"^ZNF|^ZSCAN|^ZFP|^ZKSCAN|^ZIK|^ZBTB|^OR\d|^KRTAP|^COL\d"
    rows = []
    for name, keep in [("all genes", None), ("without ZNF/OR/KRTAP/collagen families", pattern)]:
        a = tr if keep is None else tr[~tr.gene_symbol.str.contains(keep)]
        b = te if keep is None else te[~te.gene_symbol.str.contains(keep)]
        Xa, Xb = f_amino_acids(a.sequence.tolist())[0], f_amino_acids(b.sequence.tolist())[0]
        m = model().fit(Xa, a.label)
        auc, lo, hi = auc_ci(b.label, m.predict_proba(Xb)[:, 1])
        both = np.vstack([Xa, Xb])
        yb = np.r_[a.label.to_numpy(), b.label.to_numpy()]
        c, g = AMINO.index("C"), AMINO.index("G")
        rows.append({"genes": name, "n_genes": len(yb),
                     "aa_only_auc": round(auc, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "cys_log2_effect": round(_effect(both[:, c], yb)[0], 4),
                     "gly_log2_effect": round(_effect(both[:, g], yb)[0], 4)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "family_removal.csv", index=False)
    print(out.to_string(index=False))
    return out


def main(claims=CLAIMS):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_disease_data()
    df = df[df.seq_id.isin(set(load_disease_splits("lenmatch").seq_id))]
    y = df.label.to_numpy()
    rows = []
    for text, (fn, direction) in claims.items():
        v = df.sequence.map(fn).to_numpy(float)
        ok = ~np.isnan(v)
        a, b = v[ok & (y == 1)], v[ok & (y == 0)]
        p = mannwhitneyu(a, b).pvalue
        ma, mb = a.mean(), b.mean()
        auc = roc_auc_score(y[ok], v[ok])
        rows.append({"claim": text, "expected": "higher" if direction > 0 else "lower",
                     "mean_disease": ma, "mean_non_disease": mb,
                     "log2_effect": np.log2(ma / mb) if ma > 0 and mb > 0 else np.nan,
                     "observed": "higher" if ma > mb else "lower",
                     "auc_single_feature": auc if direction > 0 else 1 - auc,
                     "p_value": p})
    out = pd.DataFrame(rows)
    out["q_value"] = bh(out.p_value)
    out["verdict"] = np.select(
        [(out.q_value < 0.05) & (out.observed == out.expected), out.q_value < 0.05],
        ["SUPPORTED", "OPPOSITE"], default="NOT SUPPORTED")
    out = out.sort_values(["verdict", "q_value"]).reset_index(drop=True)
    out.to_csv(OUT / "claim_tests.csv", index=False)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(out[["claim", "observed", "log2_effect", "auc_single_feature",
                   "q_value", "verdict"]].round(4).to_string(index=False))
    print("\n--- are disease-enriched codons the GC-rich ones? ---")
    enriched_codon_gc()
    print("\n--- composition-controlled retraining (lenmatch) ---")
    composition_control()
    print("\n--- Gemini: where is the CpG excess? ---")
    cpg_by_position(df)
    print("\n--- Gemini: do disease genes have fewer paralogs? ---")
    paralog_families(load_disease_data(), set(df.seq_id))
    print("\n--- Gemini: remove the big families ---")
    family_removal()
    print(f"\nwrote {OUT.relative_to(ROOT)}/claim_tests.csv, enriched_codon_gc.csv, "
          "composition_control.csv")
    return out


if __name__ == "__main__":
    main()
