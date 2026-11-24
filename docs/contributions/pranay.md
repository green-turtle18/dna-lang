# Pranay Pankaj Ghumatkar (2024A3PS0328G)

## Part: models and evaluation

Training, comparing and choosing the classifiers (proposal tasks 4 and 7).

- **Models:** logistic regression and random forest (`src/models.py`) and the
  MLP (`src/models_nn.py`) across every representation and split.
- **Class imbalance:** class weights and oversampling. Without them, models
  predicted "non-disease" for everything.
- **Evaluation:** ROC-AUC and macro-F1 on every split.
- **Comparison and final model:** combining all results into one table and
  the summary figures, and selecting the final model
  (`src/integrate.py`, `results/disease/all_results.csv`).

## Verification

Ran the full pipeline end to end (dataset build, models, controls, analyses and integration) and checked that it reproduces the numbers reported in `report/project_report.pdf` and `docs/findings.md`. Every team member did this independently.

## Use of generative AI

The code for this part was written by Claude (Anthropic) and verified by the team, as stated in the report's AI disclosure.
