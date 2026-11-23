# NAME1 — Data & Splits (guide)

> **Pre-midsem guide** for the gene-family data. Paths and contracts here
> describe that version; the current ones are in the [README](../../README.md).

Owns `src/dataio.py`. Binding spec is
[contracts 1 and 2](../../README.md#archive-pre-midsem-gene-family-data) — this page is just a
suggested route there.

Three people block on Oct 1, so the stub matters more than the data.

## Tonight

Fake data first, real data later:

    import pandas as pd, numpy as np

    def load_data(n=20):
        rng = np.random.default_rng(0)
        seqs = ["".join(rng.choice(list("ATGC"), rng.integers(300, 1500)))
                for _ in range(n)]
        return pd.DataFrame({
            "seq_id":  [f"stub_{i:04d}" for i in range(n)],
            "sequence": seqs,
            "label":   rng.integers(0, 7, n),
            "species": rng.choice(["human", "chimp", "dog"], n),
            "length":  [len(s) for s in seqs],
            "gc":      [(s.count("G") + s.count("C")) / len(s) for s in seqs],
        })

Push, then say so in the group chat. Unannounced stubs don't unblock anyone.

## Sep 29-30

The Kaggle files are tab-separated with a sequence column and a class
column. Read each, tag with species, concatenate.

Cleaning: uppercase, then drop or filter anything outside ATGC. Decide
which — dropping whole sequences with N is simpler than patching them,
and at this dataset size you can afford to lose a few. Write down which
you chose.

Duplicate check: `df.sequence.duplicated().sum()` first. For near-
duplicates, sorting by label and eyeballing a few adjacent pairs is
enough — a real similarity search isn't worth the days.

## Oct 1

Grouped split is just a mask on `species`. Random split via
`train_test_split(..., stratify=df.label)`.

Post the class distribution and the per-class length/GC summary.
NAME3 needs the imbalance to justify macro-F1; NAME4 needs the GC
numbers for the composition baseline.

## Oct 2-3

Vocabulary exploration at k=4 — top/bottom 20 k-mers per class by raw
frequency. Optional but it directly answers task 2 of the proposal.

## Watch for

- Column names may not match what the contract assumes. Open the file
  **tomorrow**, not Oct 1. If it differs, the fix is yours, and slack
  is the only thing that saves you.
- You produce raw data and index files only. No transformations here —
  that's NAME2's boundary, and doing it in both places causes leaks.
