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


def gc3(s):
    t = s[2::3]
    return (t.count("G") + t.count("C")) / len(t)


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
}


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    r = p[o] * len(p) / np.arange(1, len(p) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(len(p))
    out[o] = np.minimum(r, 1)
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
    print(f"\nwrote {(OUT / 'claim_tests.csv').relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    main()
