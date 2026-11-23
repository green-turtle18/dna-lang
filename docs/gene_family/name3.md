# NAME3 — Models & Evaluation (guide)

> **Pre-midsem guide** for the gene-family data. Paths and contracts here
> describe that version; the current ones are in the [README](../../README.md).

Owns `src/models.py`. Binding spec is
[contract 4](../../README.md#archive-pre-midsem-gene-family-data).
This page is a suggested route.

## Tonight

One function, returning a contract-4 row. Run it on stub data with
NAME2's count vectorizer. Accuracy will be meaningless — that's fine,
the pipeline runs end to end.

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    def run(Xtr, ytr, Xte, yte, model, name, **meta):
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        return {"model": name,
                "accuracy": accuracy_score(yte, p),
                "macro_f1": f1_score(yte, p, average="macro"),
                "n_features": Xtr.shape[1], **meta}

## Sep 29-30

Add random forest and `MLPClassifier(hidden_layer_sizes=(128,),
max_iter=300)`. Don't reach for PyTorch — you don't have the days, and
an MLP on TF-IDF is a legitimate "neural" entry in the table.

Add confusion matrices saved to `figures/`.

## Oct 1-3

Grid: {tfidf} × {k=3, 4, 6, (3,6), codons} × {LR, RF, MLP}. Add `count`
only if time allows. Append every row to `results/results_name3.csv`.

Oct 3: post the pivot table as markdown. That table is slide 1.

## Watch for

- **Lead with macro-F1.** Seven imbalanced classes mean accuracy
  overstates you, and a grader who knows that will ask.
- Set `random_state` everywhere so Oct 4 reproduces.
- If the MLP fails to converge, say so in the notes column rather than
  tuning it for hours. A converged LR beats a fiddled MLP for this talk.
