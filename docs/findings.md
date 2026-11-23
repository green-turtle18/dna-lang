# Findings: can AI read the language of DNA?

The slides are built from this document. Every number comes from a file in
`results/disease/`, and every figure is in `figures/disease/`. Rerun
`python src/integrate.py` after any change to the models.

**Data.** 15,665 human protein-coding genes: 4,483 disease (OMIM Mendelian),
11,182 non-disease. One full-length coding sequence each. See
[disease_dataset.md](disease_dataset.md).

**Headline split: `lenmatch`.** Disease and non-disease genes have the same
length and GC distribution there, and paralogs stay on one side of the
train/test split. The test set has 1,797 genes, so the 95% CI on an AUC is
about ±0.026. **Differences smaller than about 0.03 are noise.**

---

## The answer in one slide

> **Partly.** A model that only sees the DNA text can tell disease genes
> from non-disease genes better than chance (ROC-AUC ≈ 0.60), and not
> because of length or GC. It reads real vocabulary: CpG-containing words,
> codon choice, and the protein those codons spell. Along the way, word2vec
> rediscovered the genetic code from context alone. But the signal is weak.
> Most of what makes a gene cause disease (its role, dosage sensitivity,
> tissue) isn't written in its coding sequence, and part of what the model
> learns is which genes scientists have studied most.

Figure: `summary_leaky_to_honest.png`

---

## Task 1: DNA as a language (k-mers, effect of k)

ROC-AUC on the two honest splits (from `results/disease/summary_auc_pivot.csv`):

| Representation | family, logreg | family, RF | lenmatch, logreg | lenmatch, RF |
|---|---|---|---|---|
| k = 3 | 0.623 | 0.652 | 0.578 | 0.584 |
| k = 4 | 0.643 | 0.661 | 0.585 | 0.597 |
| k = 6 | 0.635 | 0.642 | 0.569 | 0.567 |
| k = 3–6 | 0.651 | 0.655 | 0.584 | 0.580 |
| codons | 0.635 | 0.661 | **0.599** | 0.592 |

- **Bigger k doesn't help.** k = 6 (4,096 words) does no better than k = 4
  (256 words), and the MLP gets *worse* with big vocabularies (family:
  0.574 at k = 6 vs 0.632 at k = 3), a sign of overfitting.
- **Codons, the biologically correct words, are at or near the top**, with
  only 61 features.
- On this task the choice of k matters much less than it did for gene
  families. All k values land within about 0.03 on `lenmatch`.

Figure: `summary_auc_by_representation.png`

## Task 2: Exploring the vocabulary

From `results/disease/vocab/` (`python src/vocab.py`):

- **Both classes use the same vocabulary.** Word frequencies correlate at
  0.995 (Spearman, 6-mers), and 94 of the top 100 6-mers are shared. The
  most frequent words are the same in both classes: CTGCTG, CTGGAG,
  GAGGAG, GAAGAA (leucine and glutamate runs).
- **The rare words reveal grammar rules.** 92% of the 100 rarest 6-mers
  contain **CG**, against 29% of all 6-mers and 0% of the top 100. The
  genome avoids CpG, because methylated CG mutates away. The rarest words
  also contain stop codons (CGTTAG, TAGCGT), since a coding sequence can't
  contain a stop in frame. Both rules hold equally in both classes.
- **Differences between classes are real but small.** On `lenmatch`
  (FDR 5%), 33 of 61 codons, 116 of 256 4-mers and 188 of 4,096 6-mers
  differ. The largest codon effects are about ±15% (log2 ±0.2).
- **CpG stands out.** Disease genes are *less* CpG-depleted: observed/expected
  0.467 vs 0.438, p < 10⁻⁶, even with GC matched. Many top disease-enriched
  6-mers contain CG (ACTTCG, TACGTC, TTGGCG).
- **Clustering:** a PCA of TF-IDF 6-mers shows the classes fully overlapping
  (PC1 6.3%, PC2 1.9% of variance). The classes don't separate in any
  simple direction.

Figures: `vocab_zipf_k6.png`, `vocab_differential_codons.png`,
`vocab_volcano_k6.png`, `pca_tfidf_k6.png`

## Task 3: NLP ideas, embeddings and word2vec

From `results/disease/embeddings/` (`python src/embeddings.py`). word2vec
(skip-gram) was trained on codons and on 6-mers, using training genes only.

- **Words in similar contexts get similar vectors: word2vec learned the
  genetic code.** It was never told which codons encode which amino acid,
  yet:
  - synonymous codons have cosine similarity 0.85, against 0.71 for
    non-synonymous pairs (permutation test p = 0.0005)
  - for 57% of codons the nearest neighbour is a synonym, against 5%
    expected by chance
  - synonyms stay closer even when their third base differs (0.83, against
    0.75 for non-synonyms with the same third base), so this isn't just
    GC-rich codons clustering
- **Word order matters.** Shuffling codon order inside each gene before
  training halves the effect: the synonym advantage falls from 0.13 to
  0.06, and nearest-neighbour synonyms fall from 57% to 36%. About half of
  what word2vec learns comes from *local context*, which bag-of-words
  can't see. The rest comes from which genes a codon appears in.
- **The main axis of the learned space is third-base GC.** On a 2-D PCA,
  G/C-ending codons separate from A/T-ending ones. The synonym structure
  sits in the other dimensions, which is why the heatmap shows it better.
- **Gene vectors** (the mean of word vectors, like a sentence embedding)
  classify as well as TF-IDF, but no better. Best on `lenmatch`: codon
  vectors + random forest, 0.601, against TF-IDF codons + logreg, 0.599.
  The embeddings capture meaning without adding predictive power.

Figures: `embedding_codon_similarity_heatmap.png`,
`embedding_similarity_groups.png`, `embedding_codon_space.png`

## Task 4: Models

- **Simple beats complex.** The MLP never beats logistic regression or
  the random forest on `lenmatch`; its best is 0.587 (word2vec 6-mers).
- **Class imbalance had to be fixed.** With 29% disease genes, unweighted
  models predicted "non-disease" for everything (71% accuracy, macro-F1
  0.42). Class weights (LR, RF) and oversampling (MLP) fixed it.
- **Final model:** TF-IDF codons + logistic regression. Its AUC on
  `lenmatch` is 0.599 (macro-F1 0.562), tied with the best, which is 0.601.
  The top five models are all within 0.01 of each other, well inside the
  noise. We pick the simplest and most interpretable.

Figures: `summary_auc_by_representation.png`, `confusion_logreg_kcodon_lenmatch.png`

## Task 5: What has the AI learned?

**Artefact checks** (`python src/controls.py`):

| Check | family | lenmatch |
|---|---|---|
| length + GC only, best model | 0.615 | **0.488** (chance) |
| best sequence model | 0.661 | **0.601** |
| bases shuffled within each gene (k=6) | — | 0.494 / 0.514 (chance) |

- Length explains most of the easy wins: the control reaches 0.615 on the
  family split.
- Once length and GC are matched, the control drops to chance but the
  sequence models stay at about 0.60. Shuffling bases destroys the signal,
  so **base order carries information, not just composition**.
- Unseen genes: every number is on held-out genes, and on `family` and
  `lenmatch` no close paralog of a test gene was seen in training.

**Where the signal lives** (`python src/interpret.py`, `lenmatch` AUC with 95% CI):

| Features | AUC |
|---|---|
| length + GC (control) | 0.478 [0.452, 0.504] |
| CpG observed/expected alone | 0.530 [0.503, 0.556] |
| amino-acid composition (protein only) | 0.578 [0.552, 0.605] |
| synonymous codon choice only (protein removed) | 0.573 [0.545, 0.600] |
| full codon usage | **0.598** [0.572, 0.624] |

- **The protein and the codon "spelling" each carry part of the signal**,
  about equally, and they partly overlap.
- **Part of the signal is composition, but not all of it.** Composition
  alone (GC at each codon position, CpG, all 16 dinucleotides) reaches
  0.579. With composition regressed out of codon usage, the codon model
  still reaches **0.558 [0.529, 0.587]**, above chance. This test was
  suggested by ChatGPT in task 6.
- **It's not just gene families.** Removing zinc fingers, olfactory
  receptors and keratin-associated proteins (mostly non-disease) barely
  changes the result: 0.598 → 0.586.
- **Top codons for disease:** GGC (Gly), ATT (Ile), TAC (Tyr), CGT and CGC
  (Arg, *contain CG*). **Top codons for non-disease:** TGT and TGC (Cys),
  AGA and AGG (Arg, *no CG*), AGT and TCC (Ser), GAG (Glu).
- **The arginine story:** disease genes encode arginine with CG-containing
  codons more often. CpG sites mutate about 10× faster, so these genes
  collect more pathogenic variants, and their disease links get found.

Figures: `interpret_feature_sets.png`, `interpret_top_codons.png`

## Task 6: AI explanations vs verified results

See [ai_explanations.md](ai_explanations.md): the prompt, the responses,
and the verification table. 23 claim tests plus 5 targeted experiments were run on `lenmatch`
(`python src/verify_claims.py`):

- **Supported:** CpG higher in disease genes; more CG-arginine codons;
  fewer cysteines and zinc-finger motifs; fewer serines; more hydrophobic
  residues; more glycine.
- **Not supported:** CGA "nonsense hotspot" codons; transmembrane segments;
  collagen Gly-X-Y repeats; low-complexity regions; GC3 differences;
  poly-amino-acid runs.
- **Opposite of the claim:** "more codons one mutation from a stop" and
  "longer CAG/CAA (poly-Q) runs" both point the *other* way.
- 3 AI systems, 36 testable claims. Comparison table in `ai_explanations.md`.
- **Claude** scored 8 of 13 testable claims: the broad mechanisms held,
  the specific stories often failed.
- **ChatGPT** scored 5 supported plus 1 partly, out of 11. Its main
  alternative explanation (GC-biased gene conversion raising GC3) was
  wrong for our data. Its suggested control experiment was the most
  useful contribution: about 40% of the signal above chance is
  compositional, and the rest survives.
- **Gemini** made the most specific claims and scored worst: 3 of 12
  supported, 2 the *opposite* way. Its CpG predictions failed: the excess
  is strongest at synonymous-spanning sites rather than protein-changing
  ones, and it persists along the whole gene instead of sitting at the
  5' end. It was right that zinc-finger families drive the cysteine
  effect: removing them shrinks it from −0.18 to −0.07.
- **Takeaway:** all three AIs agreed on the core (CpG mutability and
  ascertainment, gene families, protein class), and all three offered
  specific mechanisms the data rejected. Confidence and specificity didn't
  predict correctness.

## Task 7: Final challenge

*"If DNA is a four-letter language, can an AI learn enough of its grammar,
vocabulary and context to recognise sequence patterns associated with disease?"*

- **Grammar:** yes. The rare-word analysis exposes CpG avoidance and
  in-frame stop avoidance.
- **Vocabulary and context:** yes. word2vec recovered the genetic code from
  context, and about half of that came from word order.
- **Disease:** only weakly. ROC-AUC is 0.60 once length is removed. The
  signal is real and interpretable (CpG mutability, protein class, gene
  families), but most of what makes a gene a disease gene isn't in its
  coding text. Part of the signal reflects how well studied a gene is,
  not biology alone.

---

## Suggested slide order

1. The question, and why we use genes rather than organisms (README scope note)
2. Dataset: 15,665 genes, how the labels were made, and the caveats
3. The length trap: `summary_leaky_to_honest.png`
4. Representations × models: `summary_auc_by_representation.png`
5. The vocabulary: `vocab_zipf_k6.png` plus the CpG / rare-words point
6. What differs: `vocab_differential_codons.png`
7. Where the signal lives: `interpret_feature_sets.png`
8. word2vec learned the genetic code: `embedding_codon_similarity_heatmap.png`
   + `embedding_similarity_groups.png`
9. AI explanations vs data: the "Comparing the systems" table from `ai_explanations.md`
10. Answer and limitations

## Limitations to say out loud

- "Non-disease" means no disease link known *yet*. Label noise caps
  accuracy.
- Discovery bias: well-studied genes are more likely to be labelled disease
  genes.
- Very long genes (TTN, SCN1A, HTT) aren't in the source data.
- One random seed and one train/test split per split type. Gaps under
  about 0.03 AUC are noise.
- No transformer (DNABERT): out of scope for the course tools and time.
  It's the natural next step.
