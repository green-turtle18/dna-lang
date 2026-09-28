"""
dataio.py — data loading and splits.
Owner: Leora.

Produces data/sequences.csv (contract 1) and data/splits.csv,
data/splits_random.csv (contract 2).

Stub mode works before the real files exist, so nobody is blocked:
    from src.dataio import load_data
    df = load_data()          # real data if present, else fake rows

Build the real files:
    python src/dataio.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

SPECIES_FILES = {
    "human": "human.txt",
    "chimp": "chimpanzee.txt",
    "dog":   "dog.txt",
}

PREFIX = {"human": "hum", "chimp": "chi", "dog": "dog"}

VALID = set("ATGC")


# ---------------------------------------------------------------- stub

def _stub(n=30, seed=0):
    """Fake rows matching contract 1. Right shape, meaningless content."""
    rng = np.random.default_rng(seed)
    species = rng.choice(list(PREFIX), n)
    seqs = ["".join(rng.choice(list("ATGC"), rng.integers(300, 1500)))
            for _ in range(n)]
    return pd.DataFrame({
        "seq_id":   [f"{PREFIX[s]}_stub_{i:04d}" for i, s in enumerate(species)],
        "sequence": seqs,
        "label":    rng.integers(0, 7, n),
        "species":  species,
        "length":   [len(s) for s in seqs],
        "gc":       [gc_fraction(s) for s in seqs],
    })


# ------------------------------------------------------------- helpers

def gc_fraction(s):
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


def clean(s):
    """Uppercase, strip whitespace. Returns None if any base is not ATGC."""
    s = str(s).strip().upper()
    return s if s and set(s) <= VALID else None


# --------------------------------------------------------------- build

def build(verbose=True):
    """Read the three Kaggle files, clean, dedupe, write sequences.csv."""
    frames = []
    for sp, fname in SPECIES_FILES.items():
        path = DATA / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the Kaggle dataset into data/."
            )
        d = pd.read_csv(path, sep="\t")
        d.columns = [c.strip().lower() for c in d.columns]
        d = d.rename(columns={"class": "label"})
        d["species"] = sp
        frames.append(d[["sequence", "label", "species"]])

    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)

    # clean; drop sequences containing N or other ambiguous bases
    df["sequence"] = df["sequence"].map(clean)
    n_bad = df["sequence"].isna().sum()
    df = df.dropna(subset=["sequence"]).reset_index(drop=True)

    # exact duplicates — keep first occurrence
    n_dup = df["sequence"].duplicated().sum()
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)

    df["length"] = df["sequence"].str.len()
    df["gc"] = df["sequence"].map(gc_fraction)
    df["label"] = df["label"].astype(int)
    df["seq_id"] = [f"{PREFIX[s]}_{i:05d}" for i, s in enumerate(df["species"])]

    df["group_id"] = df.sequence.str[:100].factorize()[0]
    df = df[["seq_id", "sequence", "label", "species", "length", "gc", "group_id"]]
    df.to_csv(DATA / "sequences.csv", index=False)

    if verbose:
        print(f"raw rows            {n_raw}")
        print(f"dropped (non-ATGC)  {n_bad}")
        print(f"dropped (duplicate) {n_dup}")
        print(f"final rows          {len(df)}\n")
        print("labels present:", sorted(df.label.unique()), "\n")
        print(pd.crosstab(df.label, df.species), "\n")
        print(df.groupby("label")[["length", "gc"]].mean().round(3))

    return df


# -------------------------------------------------------------- splits

def make_splits(df, seed=0, verbose=True):
    """Grouped split (train human+chimp, test dog) and stratified random."""
    from sklearn.model_selection import train_test_split

    grouped = pd.DataFrame({
        "seq_id": df.seq_id,
        "split": np.where(df.species == "dog", "test", "train"),
    })
    grouped.to_csv(DATA / "splits.csv", index=False)

    tr, te = train_test_split(
        df.seq_id, test_size=0.2, random_state=seed, stratify=df.label
    )
    random = pd.DataFrame({
        "seq_id": df.seq_id,
        "split": np.where(df.seq_id.isin(set(te)), "test", "train"),
    })
    random.to_csv(DATA / "splits_random.csv", index=False)

    if verbose:
        for name, s in [("grouped", grouped), ("random", random)]:
            n_tr = (s.split == "train").sum()
            print(f"{name:8s} train {n_tr}  test {len(s) - n_tr}")

    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=df.group_id))
    pd.DataFrame({
        "seq_id": df.seq_id,
        "split": np.where(df.index.isin(te_idx), "test", "train"),
    }).to_csv(DATA / "splits_nodup.csv", index=False)

    return grouped, random

# ---------------------------------------------------------------- load

def load_data():
    """Real data if sequences.csv exists, otherwise stub rows."""
    path = DATA / "sequences.csv"
    if path.exists():
        return pd.read_csv(path)
    return _stub()


SPLITS = {
    "random": "splits_random.csv",   # leaky baseline
    "nodup":  "splits_nodup.csv",    # near-duplicates kept together
    "species": "splits.csv",         # train human+chimp, test dog
}


def load_splits(split="species"):
    """Split index. Falls back to a stub split if the file is absent."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    path = DATA / SPLITS[split]
    if path.exists():
        return pd.read_csv(path)
    df = _stub()
    return pd.DataFrame({
        "seq_id": df.seq_id,
        "split": np.where(df.species == "dog", "test", "train"),
    })


def load_split_data(split="species"):
    """Convenience: returns (train_df, test_df) already joined."""
    df = load_data()
    sp = load_splits(split)
    df = df.merge(sp, on="seq_id", how="inner")
    return df[df.split == "train"].copy(), df[df.split == "test"].copy()


if __name__ == "__main__":
    df = build()
    print()
    make_splits(df)