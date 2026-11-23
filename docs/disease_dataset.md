# Disease vs non-disease gene dataset

This dataset answers the proposal's original question. The Kaggle corpus
only gives gene-family labels; this one gives every gene a binary
**disease / non-disease** label. It follows contract 1, so `represent.py`,
`models.py` and `controls.py` work on it unchanged. Only the loader import
changes:

```python
from src.disease_data import load_disease_split_data
train, test = load_disease_split_data("family")   # "random" | "family" | "lenmatch"
```

All files live in `data/disease/`. Rebuild from scratch (downloads about 30 MB
from GitHub into `data/disease/raw/`, takes about a minute):

```bash
python src/disease_data.py
```

## What's in it

**15,665 human genes. Each row is one full-length protein-coding ORF.**

| label | meaning | count | median length (nt) | mean GC |
|---|---|---|---|---|
| 1 | disease gene | 4,483 (29%) | 1,425 | 0.529 |
| 0 | non-disease gene | 11,182 (71%) | 1,074 | 0.527 |

Another 782 genes have only weak or partial evidence. They are set aside
in `data/disease/disease_genes_ambiguous.csv` and get no label (details below).

### Sequences

The sequences come from the human ORFeome 9.1 reference FASTA in the
BOGO repository (`Harry77j/BOGO_analysis`, MIT licence, pinned commit).
Each one is a cloned coding sequence that runs from ATG to the last sense
codon. There is one ORF per gene, keyed by NCBI Entrez ID.

Cleaning steps:
18,388 raw → 1 non-ACGT → 195 not a complete ORF (no ATG start, not a
multiple of 3, or under 150 nt) → 36 with an internal stop → 78 exact
duplicates → 1,631 not protein-coding (pseudogenes, lncRNA ORFs, and
symbols missing from HGNC's protein-coding list) → 782 ambiguous → **15,665**.

**The terminal stop codon is removed from every sequence.** About 13% of
ORFeome clones are "fusion-ready" and were cloned without a stop. If the
stop were left in, whether the last three bases are a stop codon would be
a clone artefact that a model could pick up.

### Labels

| | rule | source |
|---|---|---|
| **1 disease** | at least one OMIM gene–phenotype link classed as Mendelian | HPO annotations, release 2026-09-01 (`genes_to_disease.txt`, from NCBI mim2gene_medgen) |
| **0 non-disease** | no link of any kind in any source below | all of the sources below |
| set aside | some evidence, but not OMIM Mendelian | — |

To be labelled 0, a gene must be absent from **every** source:
- HPO: OMIM (Mendelian or polygenic) and Orphanet, any association type
- ClinVar genes with at least one pathogenic or likely-pathogenic variant (2015 snapshot)
- OMIM dominant, recessive and X-linked gene lists curated by Berg 2013 and Blekhman 2008
- ClinGen haploinsufficient genes (2015 and 2018) and curated haploinsufficiency lists (2016)
- ACMG secondary-findings genes and the BROCA cancer panel

All of these except HPO come from the MacArthur lab's `gene_lists` (MIT licence).

The 782 set-aside genes break down as follows:
- 395 Orphanet-only: Orphanet mixes "disease-causing" with "candidate"
  and "biomarker" links, and HPO doesn't separate them
- 222 OMIM polygenic or susceptibility only
- 165 found only in an older curated list

Spot check: CFTR, BRCA1/2, DMD, HBB, FBN1, TP53, MECP2, LMNA, KRT5 and
APOE are all labelled 1. GAPDH, the olfactory receptors and ZNF76 are
labelled 0. CCR5 and HLA-B are set aside.

## Columns

The first seven columns match contract 1. In this dataset `group_id`
means homology family (see Splits).

| column | notes |
|---|---|
| seq_id | `dg_00000`… |
| sequence | ATGC only, ATG start, stop codon removed |
| label | 1 = disease, 0 = non-disease |
| species | always `human` |
| length, gc | as in contract 1 |
| group_id | homology family: paralogs share an id |
| gene_symbol, entrez_id | gene identity, for looking things up and sanity checks |
| n_omim_mendelian, n_orphanet | how many diseases the gene is linked to |
| n_evidence_sources | number of independent sources that call it a disease gene (0 for negatives) |
| in_clinvar_plp, in_omim_curated, in_clingen_hi, in_curated_hi, in_clinical_panel | evidence flags for each source |
| lof_tolerant | healthy people carry homozygous loss-of-function variants (ExAC). Strong sign the gene is dispensable |
| mouse_lethal | knockout is lethal in mice. A negative with this flag is likely an **undiscovered** disease gene |
| gwas_nearest | nearest gene to a GWAS hit (weak, common-disease evidence) |
| olfactory | olfactory receptor (all of them are negatives; see caveats) |

Flags never change a label. They are there so you can build stricter
subsets. Two examples:
- **High-confidence negatives:** `label==0 & ~mouse_lethal & ~gwas_nearest` gives 7,192 genes.
- **High-confidence positives:** `n_evidence_sources >= 3` gives 2,657 genes.

## Splits

All splits are committed and identical for everyone. As before, nobody
makes their own.

| name | file | what it does |
|---|---|---|
| `random` | `splits_disease_random.csv` | stratified 80/20. Paralogs can land on both sides (leaky baseline) |
| `family` | `splits_disease_family.csv` | stratified 80/20 that keeps every homology family on one side. **Default.** |
| `lenmatch` | `splits_disease_lenmatch.csv` | the `family` split restricted to a balanced subset (8,902 genes) where disease and non-disease have the same length × GC distribution |

**How homology families are built.** Each ORF is translated. Two proteins
are linked when they share at least 10 amino-acid 5-mers, covering at
least 20% of the shorter protein. Families are the connected components
of these links. This groups clear paralogs (GAPDH/GAPDHS, MYH6/MYH7, the histone H2A
and H2B clusters) and doesn't link unrelated genes. Distant zinc-finger
paralogs, which mostly share only repeated finger motifs, are left in
separate families.
The result is 13,786 families. 3,083 genes are in multi-gene families,
and the largest family has 18 genes. The `family` and `lenmatch` splits
have 0 families on both sides.

## Baselines: read this before the talk

Disease genes are **longer** (median 1,425 vs 1,074 nt), as the
literature also reports. So length alone carries signal, which is the
same trap as in the gene-family data. The table below shows results with
seed 0, class-balanced weights, and AUC on the test split.

| split | length+GC only (RF) | TF-IDF k=3 | TF-IDF k=6 | codons (k=3, no overlap) |
|---|---|---|---|---|
| random | 0.63 | 0.62 | 0.64 | 0.62 |
| family | 0.59 | 0.62 | 0.64 | 0.64 |
| lenmatch | **0.48** (chance) | 0.58 | 0.57 | **0.60** |

What the numbers say:
1. **This task is much harder than gene families.** Expect AUC around
   0.6, not 0.95. That is a real finding, not a bug. Whether a gene
   causes disease depends on its role in the body (dosage sensitivity,
   tissue, network position), and most of that is not written in the
   coding sequence.
2. **Once length and GC are matched, composition is at chance. K-mers
   still reach 0.57–0.60.** So there is a small sequence signal beyond
   length (test n = 1,797, so the AUC standard error is about 0.013).
   Codon tokenisation does best, which fits contract 3's note that
   frame-0 codons are the biologically correct unit.
3. Random vs family barely differs here, so paralog leakage isn't
   inflating these numbers. Still report `family`.

Report `lenmatch` as the headline. It is the only split where length
can't explain the result. The full table is in `results/disease_baselines.csv`.

## Caveats (say these before anyone asks)

- **"Non-disease" means no known disease link as of Sept 2026. It does
  not mean proven harmless.** Some negatives are undiscovered disease
  genes. `mouse_lethal` flags the likeliest ones (7.7% of negatives vs
  27% of positives). This label noise caps achievable accuracy.
- **Discovery bias.** Well-studied genes are more likely to have a known
  disease. Part of what separates the classes is "how studied", not
  biology.
- **Very long genes are missing.** Genes such as TTN, SCN1A and HTT are
  hard to clone, so the ORFeome doesn't contain them. The longest ORF is
  13.7 kb.
- **Olfactory receptors** (292 genes) are all negatives and are
  very similar to each other. The `family` split keeps each OR family on
  one side, but you could drop them as a robustness check.
- **Old symbols.** The secondary lists are matched by gene symbol (2015
  era), so a few renamed disease genes could slip into the negatives.
  The main source, HPO, is matched by Entrez ID, so it isn't affected.
- **Clone sequences.** ORFeome clones can carry a few silent differences
  from the reference genome. This doesn't matter for k-mer statistics.
- **This is gene-level, not organism-level.** It labels human genes, not
  pathogenic vs non-pathogenic organisms.

## Sources and citations

- ORFeome 9.1 reference FASTA: Jo et al. 2025, *BOGO*, bioRxiv 10.1101/2025.09.02.673780;
  github.com/Harry77j/BOGO_analysis @ d031fa7
- HPO annotations v2026-09-01: Gargano et al. 2024, *Nucleic Acids Res.*;
  github.com/obophenotype/human-phenotype-ontology
- MacArthur lab gene lists @ fcc8496 (ClinVar: Landrum 2014; Berg 2013;
  Blekhman 2008; ClinGen: Rehm 2015; ExAC: Lek 2016; MGI: Blake 2011 / Georgi 2013)
