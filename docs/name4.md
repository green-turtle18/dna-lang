# NAME4 — Controls & Visualisation (guide)

Owns `src/controls.py`. Binding spec is
[contract 4](../README.md#4-resultsresults_namecsv--name3-and-name4).
This page is a suggested route.

This role produces slide 2. The classifier numbers are ordinary;
the controls are what make the talk worth four minutes.

## Tonight

Both of these run on stub data with no dependency on NAME2 or NAME3:

    import numpy as np

    def composition_features(seqs):
        return np.array([[(s.count("G") + s.count("C")) / len(s), len(s)]
                         for s in seqs])

    def shuffle_bases(seqs, seed=0):
        rng = np.random.default_rng(seed)
        return ["".join(rng.permutation(list(s))) for s in seqs]

## Sep 29-30

**Composition baseline** — feed `composition_features` straight into
NAME3's `run()`. It's just a different X. If this gets 88% and the full
k-mer model gets 91%, the language framing is barely earning its keep.
Say that out loud in the talk.

**Shuffle control** — shuffle train and test, vectorise, train, compare.
Shuffling destroys order while preserving composition exactly, so the
gap measures how much the model uses *arrangement* — which is the
difference between grammar and vocabulary.

## Oct 1-2

Cross-species: run the same model under `splits.csv` (grouped) and
`splits_random.csv`. The gap estimates homology leakage and is probably
the most quotable number in the talk.

## Oct 2-3

PCA of the TF-IDF matrix at the best k, coloured by class, into
`figures/`. `TruncatedSVD` handles sparse input directly. t-SNE only if
PCA is done and there's time.

## Watch for

- Fixed seeds on every shuffle, or Oct 4 won't reproduce.
- A near-zero shuffle gap is a real result, not a bug. Report it.
- Don't over-interpret PCA structure — clusters in two dimensions can
  be artefacts of the projection.
