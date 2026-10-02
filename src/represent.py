"""DNA k-mer bag-of-words and TF-IDF representations (README contract 3).

Example, from the repository root after the real data has been built::

    from src.dataio import load_split_data
    from src.represent import vectorize

    train, test = load_split_data("species")
    X_train, X_test, features = vectorize(
        train["sequence"].tolist(), test["sequence"].tolist(),
        k=(3, 6), mode="tfidf",
    )

Rows preserve input order; column j represents features[j]. Both matrices
are CSR sparse matrices. Only training sequences determine the vocabulary
and, for TF-IDF, the inverse document frequencies.
"""

from functools import partial
from numbers import Integral

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


__all__ = ["vectorize"]

_BASES = frozenset("ACGT")


def _k_range(k):
    """Validate a positive integer or an inclusive (minimum, maximum) tuple."""
    if isinstance(k, Integral) and not isinstance(k, bool):
        bounds = (k, k)
    elif isinstance(k, tuple) and len(k) == 2:
        bounds = k
    else:
        raise TypeError("k must be a positive integer or a (kmin, kmax) tuple")

    if any(not isinstance(n, Integral) or isinstance(n, bool) for n in bounds):
        raise TypeError("k and both k-range bounds must be integers, not booleans")
    kmin, kmax = map(int, bounds)
    if kmin < 1 or kmax < kmin:
        raise ValueError("k must satisfy 1 <= kmin <= kmax")
    return kmin, kmax


def _sequences(seqs, name):
    """Materialize input once and enforce the cleaned-sequence contract."""
    if isinstance(seqs, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of strings, not one string")
    try:
        sequences = list(seqs)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings") from exc

    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise TypeError(f"{name}[{index}] must be a string")
        invalid = set(sequence) - _BASES
        if invalid:
            raise ValueError(
                f"{name}[{index}] contains invalid bases {sorted(invalid)!r}; "
                "expected cleaned uppercase A/C/G/T sequences"
            )
    return sequences


def _kmers(sequence, kmin, kmax, overlap):
    """Emit complete tokens; each width starts at base zero."""
    for width in range(kmin, min(kmax, len(sequence)) + 1):
        step = 1 if overlap else width
        for start in range(0, len(sequence) - width + 1, step):
            yield sequence[start:start + width]


def vectorize(train_seqs, test_seqs, k, mode="tfidf", overlap=True):
    """Convert DNA sequences into aligned, sparse k-mer feature matrices.

    Parameters
    ----------
    train_seqs, test_seqs : iterable of str
        Cleaned uppercase A/C/G/T sequences, normally lists from dataio.
        An empty or shorter-than-k sequence produces a zero row. Training
        must contain at least one sequence with a complete requested k-mer.
        An empty test collection is supported.
    k : int or tuple of two ints
        Positive token length, or inclusive (kmin, kmax). For example,
        (3, 6) combines 3-, 4-, 5- and 6-mers in one vocabulary.
    mode : {"count", "tfidf"}, default="tfidf"
        "count" is bag-of-words with raw integer occurrence counts.
        "tfidf" uses raw term counts, smoothed IDF and L2 normalization:
        idf(t) = log((1 + n_train) / (1 + df_train(t))) + 1.
    overlap : bool, default=True
        True uses sliding windows with stride 1. False uses stride k,
        starting at base 0, and drops incomplete trailing tokens; k=3
        therefore gives frame-0 codons. For multi-k, each width is chunked
        independently from base 0.

    Returns
    -------
    X_train, X_test : scipy.sparse.csr_matrix
        Shapes (n_train, n_features) and (n_test, n_features). Counts are
        int64; TF-IDF weights are float64. Unseen test tokens are ignored.
    feature_names : list of str
        Uppercase k-mers in the same column order as both matrices.

    Raises
    ------
    TypeError, ValueError
        Invalid parameters, uncleaned sequences, or no training vocabulary.

    Notes
    -----
    This function never creates splits, changes labels, or fits on test
    sequences. A callable analyzer preserves exact tokens, including k=1,
    without text preprocessing or a word-token regular expression.
    """
    kmin, kmax = _k_range(k)
    if not isinstance(mode, str) or mode not in ("count", "tfidf"):
        raise ValueError("mode must be 'count' or 'tfidf'")
    if not isinstance(overlap, bool):
        raise TypeError("overlap must be True or False")

    train = _sequences(train_seqs, "train_seqs")
    test = _sequences(test_seqs, "test_seqs")
    if not train:
        raise ValueError("train_seqs must contain at least one sequence")
    if not any(len(sequence) >= kmin for sequence in train):
        raise ValueError(
            "training sequences contain no complete k-mers for the requested k"
        )

    analyzer = partial(_kmers, kmin=kmin, kmax=kmax, overlap=overlap)
    if mode == "count":
        vectorizer = CountVectorizer(analyzer=analyzer, lowercase=False)
    else:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer, lowercase=False,
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
        )

    X_train = vectorizer.fit_transform(train).tocsr()
    X_test = (
        vectorizer.transform(test).tocsr()
        if test else csr_matrix((0, X_train.shape[1]), dtype=X_train.dtype)
    )
    return X_train, X_test, vectorizer.get_feature_names_out().tolist()
