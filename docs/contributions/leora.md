# Leora Modi (2023B4AD0866G)

## Part: data and repository

The dataset everything else is built on, and the repository that holds it
(proposal task 1 and the dataset deliverable).

- **Disease dataset:** 15,665 human protein-coding genes with disease and
  non-disease labels from OMIM via HPO, sequence cleaning, and genes with
  weak evidence set aside (`src/disease_data.py`, `docs/disease_dataset.md`).
- **Splits:** `random`, `family` (homology families kept on one side) and
  `lenmatch` (length and GC matched).
- **Gene-family data:** the pre-midsem Kaggle loader and splits
  (`src/dataio.py`).
- **Transformer:** the DNABERT-style codon transformer, which is pretrained
  by predicting hidden codons and then fine-tuned to classify disease genes
  (`src/transformer.py`, `results/disease/transformer/`).
- **Repository maintenance:**
  - `disease/` and `gene_family/` folders for data, results and figures
  - the `midsem` tag
  - one `main` branch kept in sync
  - the README contracts

## Verification

Ran the full pipeline end to end (dataset build, models, controls, analyses and integration) and checked that it reproduces the numbers reported in `report/project_report.pdf` and `docs/findings.md`. Every team member did this independently.

## Use of generative AI

The code for this part was written by Claude (Anthropic) and verified by the team, as stated in the report's AI disclosure.
