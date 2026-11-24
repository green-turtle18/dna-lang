# Nidhi Deepak Mekaraj (2024A7PS0014G)

## Part: controls and interpretation

What the models actually learned, and whether it's real (proposal tasks 5 and 6).

- **Controls:** length-only, GC-only and length+GC baselines,
  shuffled-bases control (`src/controls.py`).
- **PCA visualizations:** PCA of the k-mer TF-IDF features, to check whether
  the classes separate (`figures/disease/pca_tfidf_k6.png`).
- **Model result interpretation:** reading the combined results across splits
  and representations: what the length/GC controls rule out, and which gaps
  between models are within noise.
- **Interpretation:** models restricted to one kind of information
  (amino acids, synonymous codon choice, CpG, dinucleotides), synonymous
  shuffling, composition regression, and the gene-family check
  (`src/interpret.py`).
- **LLM explanations:** the fixed prompt to Claude, ChatGPT and Gemini, and
  turning their claims into statistical tests (`src/verify_claims.py`,
  `docs/ai_explanations.md`).

Results: `results/disease/interpret/`, `results/disease/ai_claims/`,
`results/disease/results_nidhi.csv`.

## Verification

Ran the full pipeline end to end (dataset build, models, controls, analyses and integration) and checked that it reproduces the numbers reported in `report/project_report.pdf` and `docs/findings.md`. Every team member did this independently.

## Use of generative AI

The code for this part was written by Claude (Anthropic) and verified by the team, as stated in the report's AI disclosure.
