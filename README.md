# DNA as Language — CS F407 Course Project

Can an AI learn the "language" of DNA? We treat sequences as text,
tokenise with k-mers, and test whether classifiers learn real sequence
structure or just base composition and length.


## Setup

```bash
git clone https://github.com/green-turtle18/dna-lang.git
cd dna-lang
pip install -r requirements.txt
```

Download the dataset from
https://www.kaggle.com/datasets/nageshsingh/dna-sequence-dataset
into `data/` as `human.txt`, `chimpanzee.txt`, `dog.txt`.
Raw files are gitignored — everyone downloads their own.

Then build the dataset:

```bash
python src/dataio.py
```

Check it worked (run from the repo root, not inside `src/`):

```bash
python -c "from src.dataio import load_data; print(load_data().shape)"
```

Should print `(5708, 7)`. If it prints `(30, 6)` you're on stub data —
`sequences.csv` hasn't been built yet.

## Layout

    data/        sequences.csv (gitignored), split files (committed)
    src/         one module per person
    docs/        per-person guides — suggestions, not requirements
    figures/     PNGs for slides, 150 dpi
    results/     results_<name>.csv

## Ownership

| File | Owner | Guide |
|---|---|---|
| `src/dataio.py` | NAME1 | [docs/name1.md](docs/name1.md) |
| `src/represent.py` | NAME2 | [docs/name2.md](docs/name2.md) |
| `src/models.py` | NAME3 | [docs/name3.md](docs/name3.md) |
| `src/models_nn.py` | NAME1 | [docs/name1.md](docs/name1.md) |
| `src/controls.py` | NAME4 | [docs/name4.md](docs/name4.md) |

Only the owner edits their file.

The contracts below are binding — everyone codes against them. The docs
are guides for getting your own part done; follow them or don't.

---

# Contracts

## 1. `data/sequences.csv` — produced by NAME1

5,708 sequences after cleaning. Gitignored — run `python src/dataio.py`
to regenerate after downloading the Kaggle files into `data/`.

| column | type | notes |
|---|---|---|
| seq_id | str | unique, e.g. `hum_00042` |
| sequence | str | uppercase, ATGC only |
| label | int | 0-6, gene family |
| species | str | `human` / `chimp` / `dog` |
| length | int | `len(sequence)` |
| gc | float | GC fraction, 0-1 |
| group_id | int | sequences sharing their first 100 bases share this |

Cleaning: 6,882 raw → 379 dropped (non-ATGC) → 795 dropped (exact
duplicates) → 5,708 final.

## 2. Split files — produced by NAME1

Columns: `seq_id`, `split` (`train` or `test`). Join on `seq_id`.
**Nobody makes their own split.** Committed to git, so they're
identical for everyone.

| name | file | what it does |
|---|---|---|
| `random` | `splits_random.csv` | plain random; near-duplicates separated |
| `nodup` | `splits_nodup.csv` | random, but near-duplicates kept on one side |
| `species` | `splits.csv` | train human+chimp, test dog |

Difficulty increases down the table. Running the same model on all
three separates two effects: memorisation of near-duplicates
(random → nodup) and cross-species generalisation (nodup → species).

Load them with:

```python
from src.dataio import load_split_data
train, test = load_split_data("species")   # or "random", "nodup"
```

Defaults to `"species"` — the hardest — so nobody reports leaky numbers
by forgetting an argument. Returns two dataframes with contract-1
columns plus `split`.

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

Most sequences start with ATG (the start codon), so these are coding
sequences in reading frame 0 — the `overlap=False` codon tokenisation
is biologically the correct frame for this data, not an arbitrary variant.

## 4. `results/results_<name>.csv` — NAME3, NAME1, NAME4

Identical schema, separate files (CSVs merge badly in git).
Concatenated at integration.

    representation, k, model, split_type, accuracy, macro_f1, n_features, seed, notes

    representation : count | tfidf | composition_only
    k              : 3 | 4 | 6 | 3-6 | codon | none
    model          : logreg | rf | mlp
    split_type     : random | nodup | species | shuffled_bases

**Use these exact strings.** `tfidf`, not `TF-IDF` or `tf_idf`. The
comparison across representations is a pivot on this table; inconsistent
labels make it impossible.

Set `seed` on every model and record it. Two people using different
seeds makes small differences meaningless.

## 5. `figures/`

PNG, 150 dpi, descriptive names: `pca_tfidf_k6.png`,
`confusion_rf_k4.png`. Slides pull from here only.

---

# Dataset findings 

**Classes are imbalanced, nearly 5:1.**

| label | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| count | 769 | 633 | 467 | 831 | 898 | 357 | 1753 |

Class 6 is 31% of the data; class 5 is 6%. **Report macro-F1, not
accuracy** — a model that over-predicts class 6 will post a flattering
accuracy number.

**Length varies a lot by class; GC barely does.**

| label | mean length | mean GC |
|---|---|---|
| 0 | 1901 | 0.538 |
| 1 | 1918 | 0.546 |
| 2 | 2347 | 0.506 |
| 3 | 1294 | 0.516 |
| 4 | 1146 | 0.532 |
| 5 | 2539 | 0.510 |
| 6 | 1283 | 0.541 |

Class 5 averages over 2x the length of class 4. A model could score
well on length alone, knowing nothing about DNA. This makes the
composition baseline in `controls.py` the most important experiment in
the project, not a formality.

**Near-duplicates are widespread.** 2,827 of 5,708 sequences share
their first 100 bases with another; 1,029 such groups, largest has 43
members. A random split scatters these across train and test, so
`splits_random.csv` scores are partly memorisation. It's kept as the
leaky baseline to compare against.

---


## Open items

- NAME1: vocabulary exploration at k=4 (top/bottom 20 k-mers per class)
- NAME2: once results land, compare across representations and report
  which tokenisation won — that finding is slide 1
- Differences of 1-2 percentage points between k values are probably
  noise at this dataset size. Report the ranking, say which gaps are
  large enough to mean anything.

## Rules

1. Nobody reads anybody's intermediate state. Outputs are files matching
   a contract, or functions with fixed signatures.
2. `git pull` before you start. Commit small. Push when it works.

## Scope

**In:** k-mer tokenisation, BoW/TF-IDF, LR + RF + small MLP, control
experiments, one visualisation.

**Out, mention as next steps:** word2vec-style embeddings, transformers
/ DNABERT, BPE tokenisation, LLM-generated explanations, pathogen
dataset. Attempting these costs us the working pipeline.

## Scope note for the talk

Our dataset classifies **gene families**, not disease. The proposal's
framing question is disease vs non-disease. Say this before anyone asks:
we validated the pipeline on a labelled gene-family corpus; the disease
framing needs a curated pathogen dataset, which is the next step. No
ready-made one exists — every published attempt builds its own labels
from genome metadata.

Also worth knowing: BoW and TF-IDF are histograms and discard word order
entirely, so we have **not** captured context yet. That's what
embeddings are for in the second half. If asked, say so — it's a good
answer, and a bad moment if unprepared.