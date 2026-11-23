"""
transformer.py — a small DNABERT-style transformer on codons (proposal task 4).


DNABERT is a BERT model pretrained on DNA. This is the same recipe at
course scale, trained from scratch:

  1. Tokens  : in-frame codons (61 sense codons + <pad>, <cls>, <mask>).
  2. Pretrain: masked-codon modelling, the BERT objective. Hide 15% of
               codons and predict them from context. No labels used.
  3. Finetune: add a classification head and train disease vs non-disease.

Unlike bag-of-words, a transformer sees ORDER and long-range context, so
this is the proposal's test of whether context beats counting.

Genes are long (median ~450 codons, max ~4,500), so the model reads windows
of 256 codons: a random window per gene each training epoch; at test time,
windows spread across the whole gene (up to 16), averaging the predictions.

Ablation: the same model fine-tuned WITHOUT pretraining, to see whether
learning the language first helps.

Pretraining uses the family split's training genes only, so no test gene is
ever seen. lenmatch shares the family partition. The random split is
skipped: it's the leaky baseline, and the headline is lenmatch.

Run from the repo root (needs torch: pip install torch):
    python src/transformer.py            # ~1 hour on a laptop CPU
    python src/transformer.py --quick    # tiny run to check it works

Outputs:
    results/disease/results_leora_transformer.csv   (README contract 4)
    results/disease/transformer/                    (training logs, codon similarity)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.disease_data import load_disease_split_data, CODON   # noqa: E402

RESULTS = ROOT / "results" / "disease" / "results_leora_transformer.csv"
OUT = ROOT / "results" / "disease" / "transformer"
SEED = 0
COLUMNS = ["representation", "k", "model", "split_type", "accuracy", "macro_f1",
           "roc_auc", "n_features", "seed", "notes"]

SENSE = sorted(c for c, a in CODON.items() if a != "*")
PAD, CLS, MASK = 0, 1, 2
VOCAB = ["<pad>", "<cls>", "<mask>"] + SENSE
TOK = {t: i for i, t in enumerate(VOCAB)}

WIN = 256          # codons per window (attention cost grows with WIN squared)
MAX_WINDOWS = 16   # windows averaged per gene at test time (covers 4,096 codons)
CFG = dict(d=64, layers=2, heads=4, ff=128, dropout=0.1)


def encode(seq):
    return np.array([TOK.get(seq[i:i + 3], MASK) for i in range(0, len(seq) - 2, 3)],
                    dtype=np.int64)


# ------------------------------------------------------------------- model

def build_model():
    import torch.nn as nn

    class CodonTransformer(nn.Module):
        def __init__(self, d, layers, heads, ff, dropout):
            super().__init__()
            self.tok = nn.Embedding(len(VOCAB), d, padding_idx=PAD)
            self.pos = nn.Embedding(WIN + 1, d)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.mlm_head = nn.Linear(d, len(VOCAB))
            self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

        def hidden(self, x):
            import torch
            pos = torch.arange(x.shape[1], device=x.device)
            pad = x == PAD
            h = self.enc(self.tok(x) + self.pos(pos), src_key_padding_mask=pad)
            return self.norm(h), pad

        def mlm(self, x):
            return self.mlm_head(self.hidden(x)[0])

        def classify(self, x):
            h, pad = self.hidden(x)
            keep = (~pad).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)   # mean over codons
            return self.cls_head(pooled).squeeze(-1)

    return CodonTransformer(**CFG)


def batchify(windows):
    """List of codon-id arrays -> padded LongTensor with <cls> in front."""
    import torch
    L = max(len(w) for w in windows) + 1
    out = np.zeros((len(windows), L), dtype=np.int64)
    for i, w in enumerate(windows):
        out[i, 0] = CLS
        out[i, 1:len(w) + 1] = w
    return torch.from_numpy(out)


def random_window(ids, rng):
    if len(ids) <= WIN:
        return ids
    s = rng.integers(0, len(ids) - WIN + 1)
    return ids[s:s + WIN]


def all_windows(ids):
    if len(ids) <= WIN:
        return [ids]
    starts = np.linspace(0, len(ids) - WIN, min(MAX_WINDOWS, int(np.ceil(len(ids) / WIN)))).astype(int)
    return [ids[s:s + WIN] for s in starts]


# -------------------------------------------------------------- pretraining

def mask_tokens(x, rng, p=0.15):
    """BERT masking: of 15% chosen codons, 80% -> <mask>, 10% random, 10% kept."""
    import torch
    x = x.clone()
    labels = torch.full_like(x, -100)
    cand = (x > MASK)
    chosen = torch.from_numpy(rng.random(x.shape) < p) & cand
    labels[chosen] = x[chosen]
    r = torch.from_numpy(rng.random(x.shape))
    x[chosen & (r < 0.8)] = MASK
    rand = chosen & (r >= 0.8) & (r < 0.9)
    x[rand] = torch.from_numpy(rng.integers(3, len(VOCAB), x.shape))[rand]
    return x, labels


def pretrain(model, genes, epochs, batch=32, lr=1e-3, log=None):
    import torch
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    n = len(genes)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        tot, cnt, correct, masked, t0 = 0.0, 0, 0, 0, time.time()
        for b in range(0, n, batch):
            x = batchify([random_window(genes[i], rng) for i in order[b:b + batch]])
            xm, y = mask_tokens(x, rng)
            logits = model.mlm(xm)
            loss = lossf(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            cnt += 1
            m = y != -100
            correct += (logits.argmax(-1)[m] == y[m]).sum().item()
            masked += m.sum().item()
        row = {"epoch": ep + 1, "mlm_loss": round(tot / cnt, 4),
               "masked_codon_accuracy": round(correct / masked, 4),
               "secs": round(time.time() - t0)}
        print(f"  pretrain epoch {ep + 1}: loss {row['mlm_loss']:.3f}  "
              f"masked-codon accuracy {row['masked_codon_accuracy']:.3f}  ({row['secs']}s)",
              flush=True)
        if log is not None:
            log.append(row)
    return model


# -------------------------------------------------------------- fine-tuning

def predict(model, genes, batch=64):
    import torch
    model.eval()
    wins, owner = [], []
    for g, ids in enumerate(genes):
        for w in all_windows(ids):
            wins.append(w)
            owner.append(g)
    owner = np.array(owner)
    probs = np.zeros(len(wins))
    order = np.argsort([len(w) for w in wins])          # similar lengths together
    with torch.no_grad():
        for b in range(0, len(wins), batch):
            idx = order[b:b + batch]
            probs[idx] = torch.sigmoid(model.classify(batchify([wins[i] for i in idx]))).numpy()
    return np.bincount(owner, probs) / np.bincount(owner)


def finetune(model, tr_genes, ytr, groups, epochs, batch=32, lr=3e-4, patience=2, log=None):
    """Weighted BCE for class imbalance; early stopping on a grouped 10%
    validation slice of TRAIN (paralog families kept together)."""
    import copy
    import torch
    rng = np.random.default_rng(SEED)
    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=0.1, random_state=SEED)
                          .split(tr_genes, ytr, groups))
    pos_weight = torch.tensor((ytr[tr_idx] == 0).sum() / max(1, (ytr[tr_idx] == 1).sum()),
                              dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr_idx)
        t0 = time.time()
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            x = batchify([random_window(tr_genes[i], rng) for i in idx])
            loss = lossf(model.classify(x), torch.tensor(ytr[idx], dtype=torch.float32))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        va_auc = roc_auc_score(ytr[va_idx], predict(model, [tr_genes[i] for i in va_idx]))
        print(f"    finetune epoch {ep + 1}: validation AUC {va_auc:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
        if log is not None:
            log.append({"epoch": ep + 1, "val_auc": round(va_auc, 4)})
        if va_auc > best:
            best, best_state, bad = va_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


# ---------------------------------------------- what did pretraining learn?

def codon_embedding_semantics(model):
    """Same synonym test as embeddings.py, on two layers of the pretrained
    transformer: the input embedding (what each codon means going in) and
    the masked-codon output layer (which codons can fill the same gap)."""
    from src.embeddings import semantic_tests
    from itertools import combinations
    summaries, tables = {}, []
    for layer, weight in [("input embedding", model.tok.weight),
                          ("output (masked-codon) layer", model.mlm_head.weight)]:
        W = weight.detach().numpy()
        V = {c: W[TOK[c]] / np.linalg.norm(W[TOK[c]]) for c in SENSE}
        rows = [{"codon_a": a, "codon_b": b, "synonymous": CODON[a] == CODON[b],
                 "same_third_base_type": (a[2] in "GC") == (b[2] in "GC"),
                 "cosine": float(V[a] @ V[b])} for a, b in combinations(SENSE, 2)]
        summary, table = semantic_tests(pd.DataFrame(rows))
        summaries[layer] = summary
        tables.append(table.assign(layer=layer))
    return pd.DataFrame(summaries), pd.concat(tables)


# --------------------------------------------------------------------- main

def main(quick=False):
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    pre_epochs, ft_epochs = (1, 1) if quick else (4, 6)
    fam_tr, _ = load_disease_split_data("family")
    if quick:
        fam_tr = fam_tr.sample(600, random_state=SEED)
    pre_genes = [encode(s) for s in fam_tr.sequence]

    print(f"--- pretraining (masked codons) on {len(pre_genes)} family-train genes ---")
    t0 = time.time()
    plog = []
    pretrained = pretrain(build_model(), pre_genes, pre_epochs, log=plog)
    pd.DataFrame(plog).to_csv(OUT / "pretrain_log.csv", index=False)
    torch.save(pretrained.state_dict(), OUT / "pretrained_codon_transformer.pt")
    print(f"pretraining took {time.time() - t0:.0f}s; "
          f"chance masked-codon accuracy would be ~{1 / 61:.3f}\n")

    summary, groups_table = codon_embedding_semantics(pretrained)
    summary.to_csv(OUT / "codon_semantics_summary.csv")
    groups_table.to_csv(OUT / "codon_similarity_groups.csv", index=False)
    print("does the pretrained transformer learn synonymous codons?")
    print(summary.to_string(), "\n")

    rows, flog = [], []
    runs = [("lenmatch", True), ("lenmatch", False), ("family", True)]   # headline first
    for split, use_pre in runs:
        tr, te = load_disease_split_data(split)
        if quick:
            tr = tr.sample(min(600, len(tr)), random_state=SEED)
            te = te.sample(min(300, len(te)), random_state=SEED)
        name = "pretrained" if use_pre else "no pretraining"
        print(f"--- {split}: fine-tune, {name} ---")
        model = build_model()
        if use_pre:
            model.load_state_dict(pretrained.state_dict())
        t0 = time.time()
        log = []
        model, val = finetune(model, [encode(s) for s in tr.sequence], tr.label.to_numpy(),
                              tr.group_id.to_numpy(), ft_epochs, log=log)
        for r in log:
            flog.append({"split": split, "pretrained": use_pre, **r})
        prob = predict(model, [encode(s) for s in te.sequence])
        y = te.label.to_numpy()
        # threshold chosen on balance: predict disease when prob > 0.5 under weighted loss
        pred = (prob > 0.5).astype(int)
        r = {"representation": "transformer_mlm" if use_pre else "transformer_scratch",
             "k": "codon", "model": "transformer", "split_type": split,
             "accuracy": round(accuracy_score(y, pred), 4),
             "macro_f1": round(f1_score(y, pred, average="macro"), 4),
             "roc_auc": round(roc_auc_score(y, prob), 4),
             "n_features": sum(p.numel() for p in model.parameters()), "seed": SEED,
             "notes": f"{name}; val AUC {val:.3f}; {time.time() - t0:.0f}s"
                      + ("; QUICK" if quick else "")}
        rows.append(r)
        print(f"  TEST AUC {r['roc_auc']:.3f}  macro-F1 {r['macro_f1']:.3f}\n", flush=True)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        if not quick:
            pd.DataFrame(rows)[COLUMNS].to_csv(RESULTS, index=False)
    pd.DataFrame(flog).to_csv(OUT / "finetune_log.csv", index=False)
    print(pd.DataFrame(rows)[["split_type", "representation", "roc_auc", "macro_f1"]]
          .to_string(index=False))
    if quick:
        print("\n(quick mode: results not written to the results CSV)")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
