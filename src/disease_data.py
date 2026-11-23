"""
disease_data.py — curated disease vs non-disease human gene dataset.

The proposal's framing question is disease vs non-disease. The Kaggle
corpus only has gene-family labels, so this module builds a second,
independent dataset with binary disease labels, in the same shape as
contract 1 so the rest of the pipeline (represent.py, models.py,
controls.py) can run on it unchanged.

    label 1  disease gene      OMIM Mendelian gene–phenotype link (via HPO)
    label 0  non-disease gene  no disease link in ANY source used here
    (genes with only weak or partial evidence are set aside, not labelled)

Sequences are full-length human protein-coding ORFs (start codon to the
last sense codon, stop codon removed) from the human ORFeome 9.1
reference, one per gene. Full details and caveats: docs/disease_dataset.md

Build (downloads ~30 MB from GitHub into data/disease/raw/):

    python src/disease_data.py

Use, from the repo root:

    from src.disease_data import load_disease_split_data
    train, test = load_disease_split_data("family")   # or "random", "lenmatch"
"""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import urllib.request

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "disease"
RAW = DATA / "raw"
OUT = DATA / "disease_genes.csv"
OUT_AMBIGUOUS = DATA / "disease_genes_ambiguous.csv"

SEED = 0

# ------------------------------------------------------------- sources
# Pinned to exact commits / releases so every teammate gets identical data.

_BOGO = ("https://raw.githubusercontent.com/Harry77j/BOGO_analysis/"
         "d031fa74331db157ae33cb5d8fc95cc8ec943fdb/reference/")
_GL = ("https://raw.githubusercontent.com/macarthur-lab/gene_lists/"
       "fcc849637bd71e683bffc618e1a48081a8df08f8/lists/")
_HPO = ("https://github.com/obophenotype/human-phenotype-ontology/"
        "releases/download/v2026-09-01/")

SOURCES = {
    # sequences: human ORFeome 9.1 reference (BOGO preprint, MIT licence)
    "orfeome.fa.gz": _BOGO + "9%271_delta_final.fa.gz",
    # labels: HPO annotation release 2026-09-01 (OMIM via mim2gene_medgen + Orphanet)
    "genes_to_disease.txt": _HPO + "genes_to_disease.txt",
}

# MacArthur lab gene lists (MIT licence). Disease lists are used only to
# keep known disease genes OUT of the negative class.
DISEASE_LISTS = {
    "clinvar_plp":   ["clinvar_path_likelypath.tsv", "x-linked_clinvar.tsv"],
    "omim_curated":  ["all_ad.tsv", "all_ar.tsv", "berg_ad.tsv", "berg_ar.tsv",
                      "berg_xd.tsv", "berg_xr.tsv", "blekhman_ad.tsv",
                      "blekhman_ar.tsv", "blekhman_x.tsv"],
    "clingen_hi":    ["clingen_level3_genes_2018_09_13.tsv",
                      "clingen_level3_genes_2015_02_27.tsv"],
    "curated_hi":    ["haploinsufficiency_severe_curated_2016.tsv",
                      "haploinsufficiency_moderate_curated_2016.tsv",
                      "haploinsufficiency_mild_curated_2016.tsv"],
    "clinical_panel": ["ACMG_2_0.tsv", "BROCA_Cancer_Risk_Panel.tsv"],
}
# Metadata-only lists: never change a label, but let you filter or stratify.
FLAG_LISTS = {
    "lof_tolerant":     "homozygous_lof_tolerant_twohit.tsv",  # ExAC hom-LoF seen in healthy people
    "mouse_lethal":     "mgi_essential.tsv",                   # knockout lethal in mice
    "gwas_nearest":     "gwascatalog.tsv",                     # nearest gene to a GWAS hit
    "olfactory":        "olfactory_receptors.tsv",
    "protein_coding":   "universe.tsv",                        # HGNC protein-coding, 2015
}
for _files in list(DISEASE_LISTS.values()) + [[f] for f in FLAG_LISTS.values()]:
    for _f in _files:
        SOURCES[_f] = _GL + _f

MIN_LEN = 150          # nt; 50 codons
MAX_DF = 30            # protein 5-mers shared by more genes than this are ignored
MIN_SHARED = 10        # homology link needs >= this many shared 5-mers ...
MIN_CONTAIN = 0.20     # ... covering >= 20% of the shorter protein


# ------------------------------------------------------------ download

def download(force=False, verbose=True):
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = RAW / name
        if path.exists() and not force:
            continue
        if verbose:
            print(f"downloading {name}")
        urllib.request.urlretrieve(url, path)


def _gene_list(fname):
    path = RAW / fname
    vals = pd.read_csv(path, sep="\t", header=None, dtype=str)[0].str.strip()
    return set(vals[vals != ""]) - {"gene", "symbol", "Gene", "Symbol"}


# ------------------------------------------------------------- helpers

def gc_fraction(s):
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


_BASES = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: _AA[16 * i + 4 * j + k]
         for i, a in enumerate(_BASES) for j, b in enumerate(_BASES)
         for k, c in enumerate(_BASES)}
STOPS = {"TAA", "TAG", "TGA"}


def translate(s):
    return "".join(CODON[s[i:i + 3]] for i in range(0, len(s) - 2, 3))


def read_orfeome(path):
    rows, head, buf = [], None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if head:
                    rows.append((head, "".join(buf)))
                head, buf = line[1:], []
            elif line:
                buf.append(line)
    rows.append((head, "".join(buf)))
    out = []
    for head, seq in rows:
        # header: <index>_<entrez>_<well>_<symbol>
        idx, entrez, _well, symbol = head.split("_", 3)
        out.append({"entrez_id": int(entrez), "gene_symbol": symbol,
                    "orfeome_id": head, "sequence": seq.upper()})
    return pd.DataFrame(out)


def qc(df):
    """Keep complete, clean ORFs; remove the terminal stop codon."""
    n0 = len(df)
    df = df[df.sequence.map(lambda s: set(s) <= set("ACGT"))].copy()
    n_acgt = n0 - len(df)
    df["sequence"] = [s[:-3] if s[-3:] in STOPS else s for s in df.sequence]
    ok = (df.sequence.str.startswith("ATG")
          & (df.sequence.str.len() % 3 == 0)
          & (df.sequence.str.len() >= MIN_LEN))
    n_shape = (~ok).sum()
    df = df[ok]
    internal = df.sequence.map(
        lambda s: any(s[i:i + 3] in STOPS for i in range(0, len(s), 3)))
    n_stop = internal.sum()
    df = df[~internal]
    n_dup = df.sequence.duplicated().sum()
    df = df.drop_duplicates("sequence")
    log = {"orfeome_raw": n0, "dropped_non_acgt": n_acgt,
           "dropped_not_complete_orf": int(n_shape),
           "dropped_internal_stop": int(n_stop),
           "dropped_exact_duplicate": int(n_dup)}
    return df.reset_index(drop=True), log


def homology_groups(proteins):
    """Union-find over proteins sharing many amino-acid 5-mers.

    Groups clear paralogs (e.g. GAPDH/GAPDHS, MYH6/MYH7, histone clusters)
    without linking unrelated proteins. Used so a split never
    puts a gene in train and its close paralog in test.
    """
    sets = []
    for p in proteins:
        ks = {p[i:i + 5] for i in range(len(p) - 4)}
        sets.append({k for k in ks if len(set(k)) > 2})   # drop low-complexity
    index = defaultdict(list)
    for i, ks in enumerate(sets):
        for k in ks:
            index[k].append(i)
    shared = Counter()
    for members in index.values():
        if 2 <= len(members) <= MAX_DF:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    shared[(members[a], members[b])] += 1

    parent = list(range(len(proteins)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), n in shared.items():
        smaller = max(1, min(len(sets[a]), len(sets[b])))
        if n >= MIN_SHARED and n / smaller >= MIN_CONTAIN:
            parent[find(a)] = find(b)
    roots = [find(i) for i in range(len(proteins))]
    return pd.Series(roots).factorize()[0]


# --------------------------------------------------------------- build

def build(verbose=True):
    download(verbose=verbose)
    df, log = qc(read_orfeome(RAW / "orfeome.fa.gz"))

    hpo = pd.read_csv(RAW / "genes_to_disease.txt", sep="\t")
    hpo["entrez_id"] = hpo.ncbi_gene_id.str.replace("NCBIGene:", "").astype(int)
    hpo["db"] = hpo.disease_id.str.split(":").str[0]
    omim_mend = hpo[(hpo.db == "OMIM") & (hpo.association_type == "MENDELIAN")]
    n_omim = omim_mend.groupby("entrez_id").disease_id.nunique()
    n_orpha = hpo[hpo.db == "ORPHA"].groupby("entrez_id").disease_id.nunique()
    any_hpo = set(hpo.entrez_id)
    hpo_symbols = set(hpo.gene_symbol)

    df["n_omim_mendelian"] = df.entrez_id.map(n_omim).fillna(0).astype(int)
    df["n_orphanet"] = df.entrez_id.map(n_orpha).fillna(0).astype(int)
    df["omim_polygenic_only"] = (df.entrez_id.isin(
        hpo[(hpo.db == "OMIM") & (hpo.association_type == "POLYGENIC")].entrez_id)
        & (df.n_omim_mendelian == 0))

    for col, files in DISEASE_LISTS.items():
        genes = set().union(*(_gene_list(f) for f in files))
        df["in_" + col] = df.gene_symbol.isin(genes)
    for col, f in FLAG_LISTS.items():
        df[col] = df.gene_symbol.isin(_gene_list(f))

    # protein-coding gene filter: HGNC protein-coding list, or has any HPO link
    keep = df.protein_coding | df.entrez_id.isin(any_hpo) | df.gene_symbol.isin(hpo_symbols)
    log["dropped_not_protein_coding"] = int((~keep).sum())
    df[~keep][["entrez_id", "gene_symbol"]].to_csv(RAW / "dropped_not_protein_coding.csv", index=False)
    df = df[keep].drop(columns="protein_coding").reset_index(drop=True)

    any_list = df[["in_" + c for c in DISEASE_LISTS]].any(axis=1)
    has_any = (df.entrez_id.isin(any_hpo) | df.gene_symbol.isin(hpo_symbols)
               | any_list)
    df["n_evidence_sources"] = ((df.n_omim_mendelian > 0).astype(int)
                                + (df.n_orphanet > 0)
                                + df[["in_" + c for c in DISEASE_LISTS]].sum(axis=1))

    pos = df.n_omim_mendelian > 0
    neg = ~has_any
    df["label"] = np.select([pos, neg], [1, 0], default=-1)
    df["ambiguous_reason"] = np.select(
        [df.label != -1, df.omim_polygenic_only, df.n_orphanet > 0],
        ["", "omim_polygenic_only", "orphanet_only"], default="curated_list_only")

    df["length"] = df.sequence.str.len()
    df["gc"] = df.sequence.map(gc_fraction)
    df["species"] = "human"

    # homology families over ALL kept genes, so ambiguous genes are grouped too
    df["group_id"] = homology_groups([translate(s) for s in df.sequence])
    df = df.sort_values(["entrez_id"]).reset_index(drop=True)
    df["seq_id"] = [f"dg_{i:05d}" for i in range(len(df))]

    lead = ["seq_id", "sequence", "label", "species", "length", "gc", "group_id"]
    meta = ["gene_symbol", "entrez_id", "n_omim_mendelian", "n_orphanet",
            "n_evidence_sources"] + ["in_" + c for c in DISEASE_LISTS] + \
           list(k for k in FLAG_LISTS if k != "protein_coding") + ["orfeome_id"]
    labelled = df[df.label >= 0][lead + meta].reset_index(drop=True)
    ambiguous = df[df.label < 0][lead + meta + ["ambiguous_reason"]].reset_index(drop=True)
    labelled.to_csv(OUT, index=False)
    ambiguous.to_csv(OUT_AMBIGUOUS, index=False)

    if verbose:
        for k, v in log.items():
            print(f"{k:28s} {v}")
        print(f"{'ambiguous (set aside)':28s} {len(ambiguous)}")
        print(f"{'final labelled rows':28s} {len(labelled)}\n")
        print(labelled.label.value_counts().rename({1: "disease", 0: "non-disease"}), "\n")
        print(labelled.groupby("label")[["length", "gc"]].agg(["mean", "median"]).round(3), "\n")
        sizes = labelled.group_id.value_counts()
        print(f"homology groups {labelled.group_id.nunique()}  "
              f"genes in multi-gene groups {(labelled.group_id.map(sizes) > 1).sum()}  "
              f"largest {sizes.max()}")
    return labelled


# -------------------------------------------------------------- splits

SPLITS = {
    "random":   "splits_disease_random.csv",    # leaky baseline
    "family":   "splits_disease_family.csv",    # paralogs kept on one side
    "lenmatch": "splits_disease_lenmatch.csv",  # family split, length+GC matched subset
}


def _length_gc_matched(df, seed=SEED, n_len=20, n_gc=5):
    """Balanced subset where both classes have the same length × GC histogram."""
    rng = np.random.default_rng(seed)
    d = df.assign(lb=pd.qcut(np.log10(df.length), n_len, labels=False),
                  gb=pd.qcut(df.gc, n_gc, labels=False))
    keep = []
    for _, cell in d.groupby(["lb", "gb"]):
        p, n = cell[cell.label == 1], cell[cell.label == 0]
        m = min(len(p), len(n))
        if m:
            keep += list(rng.choice(p.seq_id, m, replace=False))
            keep += list(rng.choice(n.seq_id, m, replace=False))
    return set(keep)


def make_splits(df=None, seed=SEED, verbose=True):
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    if df is None:
        df = pd.read_csv(OUT)
    df = df.reset_index(drop=True)

    _, te = train_test_split(df.seq_id, test_size=0.2, random_state=seed,
                             stratify=df.label)
    random = pd.DataFrame({"seq_id": df.seq_id,
                           "split": np.where(df.seq_id.isin(set(te)), "test", "train")})

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, te_idx = next(sgkf.split(df, df.label, groups=df.group_id))
    family = pd.DataFrame({"seq_id": df.seq_id,
                           "split": np.where(df.index.isin(te_idx), "test", "train")})

    matched = _length_gc_matched(df, seed)
    lenmatch = family[family.seq_id.isin(matched)].reset_index(drop=True)

    for name, s in [("random", random), ("family", family), ("lenmatch", lenmatch)]:
        s.to_csv(DATA / SPLITS[name], index=False)
        if verbose:
            j = s.merge(df[["seq_id", "label", "group_id", "length"]], on="seq_id")
            tr, tst = j[j.split == "train"], j[j.split == "test"]
            leak = len(set(tr.group_id) & set(tst.group_id))
            print(f"{name:9s} train {len(tr):5d} (pos {tr.label.mean():.2f})  "
                  f"test {len(tst):5d} (pos {tst.label.mean():.2f})  "
                  f"families on both sides {leak}")
    return random, family, lenmatch


# ---------------------------------------------------------------- load

def load_disease_data(include_ambiguous=False):
    df = pd.read_csv(OUT)
    if include_ambiguous:
        df = pd.concat([df, pd.read_csv(OUT_AMBIGUOUS)], ignore_index=True)
    return df


def load_disease_splits(split="family"):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    return pd.read_csv(DATA / SPLITS[split])


def load_disease_split_data(split="family"):
    """Returns (train_df, test_df). Defaults to the paralog-safe split."""
    df = load_disease_data().merge(load_disease_splits(split), on="seq_id", how="inner")
    return df[df.split == "train"].copy(), df[df.split == "test"].copy()


if __name__ == "__main__":
    data = build()
    print()
    make_splits(data)
