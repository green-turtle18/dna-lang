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
into `data/`. Raw files are gitignored — everyone downloads their own.

## Layout

    data/        sequences.csv (gitignored), splits.csv (committed)
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
| `src/controls.py` | NAME4 | [docs/name4.md](docs/name4.md) |

Only the owner edits their file.

The contracts below are binding — everyone codes against them. The docs
are guides for getting your own part done; follow them or don't.

---

# Contracts

## 1. `data/sequences.csv` — produced by [NAME1](docs/name1.md)

| column | type | notes |
|---|---|---|
| seq_id | str | unique, stable, e.g. `hum_0001` |
| sequence | str | uppercase, ATGC only, no N |
| label | int | 0-6, gene family |
| species | str | `human` / `chimp` / `dog` |
| length | int | `len(sequence)` |
| gc | float | GC fraction, 0-1 |

## 2. `data/splits.csv` — produced by [NAME1](docs/name1.md)

Columns: `seq_id`, `split` where split is `train` or `test`.
Join on `seq_id`. **Nobody makes their own split.**

- `splits.csv` — grouped by species: train on human+chimp, test on dog
- `splits_random.csv` — stratified random

The gap between the two estimates homology leakage. That gap is a result.

## 3. `src/represent.py` — produced by [NAME2](docs/name2.md)

    vectorize(train_seqs, test_seqs, k, mode="tfidf", overlap=True)

      train_seqs, test_seqs : list[str]
      k       : int, or tuple (kmin, kmax) for multi-k
      mode    : "count" | "tfidf"
      overlap : True  = sliding window (standard k-mers)
                False = non-overlapping frame-0 codons
      returns : X_train (sparse), X_test (sparse), feature_names (list[str])

Fit on train only, transform test. Fitting the vocabulary on all data
leaks and invalidates every number in the talk.

## 4. `results/results_<name>.csv` — [NAME3](docs/name3.md) and [NAME4](docs/name4.md)

Identical schema, separate files (CSVs merge badly in git).
Concatenated at integration.

    representation, k, model, split_type, accuracy, macro_f1, n_features, notes

    representation : count | tfidf | composition_only
    split_type     : grouped | random | shuffled_bases | cross_species

## 5. `figures/`

PNG, 150 dpi, descriptive names: `pca_tfidf_k6.png`,
`confusion_rf_k4.png`. Slides pull from here only.

---

## Timeline

| By | What |
|---|---|
| Sep 28 | Stubs pushed — everyone unblocked |
| Sep 29 | Sync call, confirm everyone's code runs against stubs |
| Oct 1 | LEORA ships real data + splits; NAME2 ships full `vectorize()` |
| Oct 3 | NAME3 ships results table; NAME4 ships controls + figures. **Results frozen.** |
| Oct 4 | Integration — every number reproduces from the shared split |
| Oct 5 | Slides + two timed rehearsals |
| Oct 6 | Talk |

## Rules

1. Nobody reads anybody's intermediate state. Outputs are files matching
   a contract, or functions with fixed signatures.
2. `git pull` before you start. Commit small. Push when it works.
3. A number that can't be reproduced on Oct 4 doesn't go on a slide.

## Scope

**In:** k-mer tokenisation, BoW/TF-IDF, LR + RF + small MLP, control
experiments, one visualisation.

**Out, mention as next steps:** word2vec-style embeddings, transformers
/ DNABERT, BPE tokenisation, LLM-generated explanations, pathogen
dataset. Attempting these costs us the working pipeline.
