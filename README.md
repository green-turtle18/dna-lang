# DNA as Language — CS F407 Course Project

Can an AI learn the "language" of DNA? We treat sequences as text,
tokenise with k-mers, and test whether classifiers learn sequence
structure or just base composition.

Midsem flash talk: 6 October. 3-4 min, 2 slides.

## Setup

    git clone <repo-url>
    cd dna-lang
    pip install -r requirements.txt

Download the dataset from
https://www.kaggle.com/datasets/nageshsingh/dna-sequence-dataset
into `data/`. Raw files are gitignored — everyone downloads their own.

Verify the skeleton works:

    python -c "from src.dataio import load_data; print(load_data().shape)"

Run from the repo root, not from inside `src/`.

## Layout

    data/        sequences.csv (gitignored), splits.csv (committed)
    src/         one module per person
    figures/     PNGs for slides, 150 dpi
    results/     results_c.csv, results_d.csv

## Ownership

| File | Owner |
|---|---|
| `src/dataio.py` | A |
| `src/represent.py` | B |
| `src/models.py` | C |
| `src/controls.py` | D |

Only the owner edits their file. Signature changes get announced in
the group chat before pushing.

## Contracts

### 1. data/sequences.csv — produced by A

| column | type | notes |
|---|---|---|
| seq_id | str | unique, stable, e.g. hum_0001 |
| sequence | str | uppercase, ATGC only, no N |
| label | int | 0-6, gene family |
| species | str | human / chimp / dog |
| length | int | len(sequence) |
| gc | float | GC fraction, 0-1 |

### 2. data/splits.csv — produced by A

Columns: `seq_id`, `split` where split is train or test.
Join on seq_id. **Nobody makes their own split.**
`splits.csv` is grouped by species (train human+chimp, test dog).
`splits_random.csv` is stratified random. The gap between them is a result.

### 3. src/represent.py — produced by B

    vectorize(train_seqs, test_seqs, k, mode="tfidf", overlap=True)
      train_seqs, test_seqs : list[str]
      k       : int, or tuple (kmin, kmax) for multi-k
      mode    : "count" | "tfidf"
      overlap : True  = sliding window (standard k-mers)
                False = non-overlapping frame-0 codons
      returns : X_train (sparse), X_test (sparse), feature_names (list[str])

Fit on train only, transform test. Fitting on all data leaks and
invalidates every number in the talk.

### 4. results/results_*.csv — C and D

Identical schema, separate files (CSVs merge badly in git).
Concatenated at integration.

    representation, k, model, split_type, accuracy, macro_f1, n_features, notes

    representation : count | tfidf | composition_only
    split_type     : grouped | random | shuffled_bases | cross_species

### 5. figures/

PNG, 150 dpi, descriptive names: `pca_tfidf_k6.png`,
`confusion_rf_k4.png`. Slides pull from here only.

## Timeline

| By | What |
|---|---|
| Sep 28 | Stubs pushed, everyone unblocked |
| Sep 29 | Sync call — confirm everyone's code runs on stubs |
| Oct 1 | A ships real data + splits; B ships full vectorize() |
| Oct 3 | C ships results table; D ships controls + figures. **results frozen** |
| Oct 4 | Integration — every number reproduces from the shared split |
| Oct 5 | Slides + two timed rehearsals |
| Oct 6 | Talk |

## Rules

1. Nobody reads anybody's intermediate state. Outputs are files
   matching a contract, or functions with fixed signatures.
2. `git pull` before you start. Commit small. Push when it works.
3. A number that can't be reproduced on Oct 4 doesn't go on a slide.
