# DNA as Language — CS F407 Course Project

Can an AI learn the "language" of DNA well enough to tell **disease genes**
from **non-disease genes**? We treat sequences as text, tokenise with
k-mers, and test whether classifiers learn real sequence structure or just
length and base composition.

Before midsem we built and validated the pipeline on a 7-class gene-family
dataset (Kaggle, human/chimp/dog). Since midsem the project runs on a
curated disease vs non-disease dataset of human genes, which answers the
proposal's actual question. The gene-family material is kept at the bottom
of this file for reference.


## Setup

```bash
git clone https://github.com/green-turtle18/dna-lang.git
cd dna-lang
pip install -r requirements.txt
```

The disease dataset is committed, so there is nothing to download. Check it
works (run from the repo root, not inside `src/`):

```bash
python -c "from src.disease_data import load_disease_split_data; tr,te=load_disease_split_data(); print(len(tr),len(te))"
```

Should print `12531 3134`.

Then run any module directly, for example `python src/models.py --quick`.

## Layout

    data/
      disease/      disease vs non-disease genes + splits (committed), see docs/disease_dataset.md
      gene_family/  pre-midsem Kaggle human/chimp/dog data: sequences.csv (gitignored), split files (committed)
    src/            one module per person
    docs/           disease_dataset.md (datasheet); gene_family/ holds the pre-midsem per-person guides
    figures/
      disease/      current PNGs for slides, 150 dpi
      gene_family/  pre-midsem PNGs
    results/
      disease/      current results CSVs (contract 4)
      gene_family/  pre-midsem results CSVs

## Ownership

| File | Owner | Guide |
|---|---|---|
| `src/disease_data.py` | NAME1 | [docs/disease_dataset.md](docs/disease_dataset.md) |
| `src/dataio.py` | NAME1 | [pre-midsem guide](docs/gene_family/name1.md) |
| `src/represent.py` | NAME2 | [pre-midsem guide](docs/gene_family/name2.md) |
| `src/models.py` | NAME3 | [pre-midsem guide](docs/gene_family/name3.md) |
| `src/models_nn.py` | NAME1 | [pre-midsem guide](docs/gene_family/name1.md) |
| `src/controls.py` | NAME4 | [pre-midsem guide](docs/gene_family/name4.md) |

Only the owner edits their file.

The contracts below are binding — everyone codes against them. The docs
are guides for getting your own part done; follow them or don't.

---

# Contracts

## 1. `data/disease/disease_genes.csv` — produced by NAME1

15,665 human protein-coding genes, one full-length coding sequence (ORF)
each. Committed. Rebuild with `python src/disease_data.py`. Label rules,
sources and caveats are in [docs/disease_dataset.md](docs/disease_dataset.md).

| column | type | notes |
|---|---|---|
| seq_id | str | unique, e.g. `dg_00042` |
| sequence | str | uppercase, ATGC only, starts with ATG, stop codon removed |
| label | int | 1 = disease (OMIM Mendelian), 0 = non-disease (no known link) |
| species | str | always `human` |
| length | int | `len(sequence)` |
| gc | float | GC fraction, 0-1 |
| group_id | int | homology family: paralogs share this |

The file has extra metadata columns after these (gene symbol, evidence
flags). Models must use only `sequence`; the metadata is for analysis.

## 2. Split files — produced by NAME1

Columns: `seq_id`, `split` (`train` or `test`). Join on `seq_id`.
**Nobody makes their own split.** Committed to git, so they're
identical for everyone.

| name | file (in `data/disease/`) | what it does |
|---|---|---|
| `random` | `splits_disease_random.csv` | stratified 80/20; paralogs can land on both sides (leaky baseline) |
| `family` | `splits_disease_family.csv` | stratified 80/20, each homology family kept on one side |
| `lenmatch` | `splits_disease_lenmatch.csv` | `family` split restricted to 8,902 genes where both classes have the same length × GC distribution; balanced 50/50 |

Difficulty increases down the table. Running the same model on all
three separates two effects: memorisation of paralogs (random → family)
and reliance on length/GC (family → lenmatch). **`lenmatch` is the
headline number**: it's the only split where length can't explain the
result.

Load them with:

```python
from src.disease_data import load_disease_split_data
train, test = load_disease_split_data("family")   # or "random", "lenmatch"
```

Returns two dataframes with contract-1 columns, the metadata columns,
and `split`.

## 3. `src/represent.py` — produced by NAME2

    vectorize(train_seqs, test_seqs, k, mode="tfidf", overlap=True)

      train_seqs, test_seqs : list[str]
      k       : int, or tuple (kmin, kmax) for multi-k
      mode    : "count" | "tfidf"
      overlap : True  = sliding window (standard k-mers)
                False = non-overlapping frame-0 codons
      returns : X_train (sparse), X_test (sparse), feature_names (list[str])

Fit on train only, transform test. Fitting the vocabulary on all data
leaks and invalidates every number in the talk.

Every sequence is a complete coding sequence starting with ATG, so
reading frame 0 is the true frame. The `overlap=False` codon tokenisation
is the biologically correct unit for this data, not an arbitrary variant.

## 4. `results/disease/results_<name>.csv` — NAME3, NAME1, NAME4

Identical schema, separate files (CSVs merge badly in git).
Concatenated at integration.

    representation, k, model, split_type, accuracy, macro_f1, roc_auc, n_features, seed, notes

    representation : count | tfidf | composition_only | <new ones, agree first>
    k              : 3 | 4 | 6 | 3-6 | codon | none
    model          : logreg | rf | mlp
    split_type     : random | family | lenmatch | shuffled_bases
    roc_auc        : area under the ROC curve for class 1 (disease)

**Use these exact strings.** `tfidf`, not `TF-IDF` or `tf_idf`. The
comparison across representations is a pivot on this table; inconsistent
labels make it impossible.

Set `seed` on every model and record it. Two people using different
seeds makes small differences meaningless.

**Every model must handle class imbalance.** Only 29% of genes are
disease genes. An unweighted model predicts "non-disease" for everything
and scores 71% accuracy with macro-F1 0.42 without reading any DNA.
Use `class_weight="balanced"` (logreg, RF) or oversample disease genes
in the training set (MLP). Never resample the test set.

## 5. `figures/disease/`

PNG, 150 dpi, descriptive names: `pca_tfidf_k6.png`,
`confusion_rf_k4_lenmatch.png`. Slides pull from here only.

---

# Disease dataset findings

**Classes are imbalanced, 29% disease.** 4,483 disease vs 11,182
non-disease. Report **macro-F1 and ROC-AUC**, not accuracy.

**Disease genes are longer; GC is the same.**

| label | median length (nt) | mean GC |
|---|---|---|
| 0 non-disease | 1,074 | 0.527 |
| 1 disease | 1,425 | 0.529 |

A model can score above chance on length alone, knowing nothing about
DNA. The composition controls in `controls.py` are the most important
comparison, and `lenmatch` removes length entirely.

**First results** (pre-midsem code with class weighting, k = 3, 4, 6, codon):

| split | length + GC only | k-mer models (ROC-AUC) |
|---|---|---|
| family | ~0.59 | 0.57–0.66 |
| lenmatch | ~0.48 (chance) | 0.56–0.60 |

There is a small but real sequence signal beyond length. Codons are at
or near the top for every model. The MLP doesn't beat logreg/RF, and the
PCA plot shows no visible separation: the signal is weak and spread
across many k-mers. Expect AUC around 0.6, not 0.95. That's a finding,
not a bug.

---

## Open items (second half)

- Rerun the full grid on all three splits (everyone, own module)
- NAME1: vocabulary exploration — top/bottom k-mers per class on `lenmatch`
- Embeddings: word2vec on k-mers, average into one vector per gene;
  check whether k-mers in similar contexts get similar vectors
- Interpretation: which k-mers/codons drive the `lenmatch` models, and
  are they just codon bias or GC in disguise?
- AI explanations: give the top k-mers to 2-3 chatbots, turn each claim
  into a computable test, check it
- Differences of 1-2 points between k values are probably noise at
  this dataset size. Report the ranking, say which gaps are large
  enough to mean anything.

## Rules

1. Nobody reads anybody's intermediate state. Outputs are files matching
   a contract, or functions with fixed signatures.
2. `git pull` before you start. Commit small. Push when it works.

## Scope

**In:** k-mer tokenisation, BoW/TF-IDF, LR + RF + small MLP, control
experiments, PCA, word2vec-style embeddings, interpretation of top
features, LLM-generated explanations checked against results.

**Stretch, only once the above works:** transformers / DNABERT, BPE
tokenisation.

## Scope note for the talk

The proposal asks about disease-causing vs non-disease-causing
**organisms**. We classify human **genes** instead: every sequence is
the same kind of text (a coding sequence), and only the label differs,
which makes it a cleaner test of whether a model reads the "language".
Say this before anyone asks.

"Non-disease" means **no known disease link as of Sept 2026**, not proven
harmless. Some negatives are undiscovered disease genes, which caps the
achievable accuracy.

BoW and TF-IDF are histograms and discard word order entirely, so they
capture no context. That's what embeddings are for in the second half.

---

# Archive: pre-midsem gene-family data

Everything from before the switch lives in `gene_family/` folders:
`data/gene_family/`, `results/gene_family/`, `figures/gene_family/` and
`docs/gene_family/`. The code itself has since been switched to the disease
data; the exact midsem version is the git tag `midsem`
(`git checkout midsem` to look, `git checkout main` to come back).

To rebuild the gene-family data, download
https://www.kaggle.com/datasets/nageshsingh/dna-sequence-dataset into
`data/gene_family/` as `human.txt`, `chimpanzee.txt`, `dog.txt` (gitignored),
then run `python src/dataio.py`. `from src.dataio import load_data;
print(load_data().shape)` should print `(5708, 7)`.

- `data/gene_family/sequences.csv`: same 7 columns as contract 1, but
  `label` is 0-6 (gene family), `species` is `human` / `chimp` / `dog`, and
  `group_id` groups sequences sharing their first 100 bases. Cleaning:
  6,882 raw → 379 non-ATGC → 795 exact duplicates → 5,708.
- Splits, loaded with `from src.dataio import load_split_data`:
  `random` (`splits_random.csv`), `nodup` (`splits_nodup.csv`,
  near-duplicates on one side), `species` (`splits.csv`, train
  human+chimp, test dog).
- Findings: classes imbalanced ~5:1 (class 6 is 31%, class 5 is 6%);
  mean length ranged from 1,146 (class 4) to 2,539 (class 5) while GC
  barely varied; 2,827 sequences shared their first 100 bases with
  another (1,029 groups, largest 43).
- Midsem results are in `results/gene_family/results_<name>.csv` and
  the midsem PNGs in `figures/gene_family/`.
