# NAME2 — Representation (guide)

> **Pre-midsem guide** for the gene-family data. Paths and contracts here
> describe that version; the current ones are in the [README](../../README.md).

Owns `src/represent.py`. Binding spec is
[contract 3](../../README.md#archive-pre-midsem-gene-family-data).
This page is a suggested route.

## Tonight

Get the signature working with the simplest possible body, on Leora's
stub. Correct shape beats correct features today.

    from sklearn.feature_extraction.text import CountVectorizer

    def vectorize(train_seqs, test_seqs, k, mode="tfidf", overlap=True):
        rng = (k, k) if isinstance(k, int) else k
        vec = CountVectorizer(analyzer="char", ngram_range=rng)
        Xtr = vec.fit_transform(train_seqs)      # fit on TRAIN only
        Xte = vec.transform(test_seqs)
        return Xtr, Xte, vec.get_feature_names_out().tolist()

## Sep 29

Three additions:

**TF-IDF** — swap in `TfidfVectorizer`, same arguments.

**Multi-k** — already handled above if `k` is a tuple. `(3, 6)` gives
the union of 3-, 4-, 5- and 6-mers, ~5,400 features. Worth its own row
in the results table: if multi-k beats every fixed k, that's evidence
DNA's units aren't uniform length.

**Non-overlapping frame-0** — chunk into consecutive triples, join with
spaces, use a whitespace analyzer:

    def codons(s):
        return " ".join(s[i:i+3] for i in range(0, len(s) - 2, 3))

Biologically this is the real tokenisation for coding DNA — codons are
exactly 3 bases read in frame. If it beats overlapping 3-mers, reading
frame carries information, which is a sharper finding than "k=6 won".

## Oct 1

Report vocab size and sparsity for k=3, 4, 6, (3,6), and codons.
Sparsity: `1 - X.nnz / (X.shape[0] * X.shape[1])`.

Confirm to the group that fit-on-train-only holds everywhere.

## Oct 2-3

Top-20 features per class by TF-IDF weight, for slide 2 and for Leora
to cross-check against raw frequency.

## Watch for

- `fit_transform` on combined train+test is the one mistake that
  quietly invalidates the whole talk. Grep your own file for it.
- k=6 with `analyzer="char"` is fine memory-wise at this dataset size;
  don't pre-emptively optimise.
