# Dev Raman Taneja (2023A7PS0336G)

## Part: representations

How DNA is turned into "language" a model can read (proposal tasks 1–3).

- **k-mers and codons:** overlapping k-mers (k = 3, 4, 6 and 3–6) and
  non-overlapping in-frame codons, as Count or TF-IDF vectors, with the
  vocabulary fitted on training genes only (`src/represent.py`).
- **Vocabulary analysis:** frequent and rare words per class, the
  rank-frequency curve, and words that differ between disease and
  non-disease genes (`src/vocab.py`).
- **Embeddings:** word2vec on codons and 6-mers, whole-gene vectors, and the
  test of whether codons used in similar contexts get similar vectors
  (`src/embeddings.py`).
- **LLM explanations:** ran the fixed prompt through ChatGPT and Gemini and
  collected their responses for verification (`docs/ai_responses/`).

Results: `results/disease/vocab/`, `results/disease/embeddings/`.

## Verification

Ran the full pipeline end to end (dataset build, models, controls, analyses and integration) and checked that it reproduces the numbers reported in `report/project_report.pdf` and `docs/findings.md`. Every team member did this independently.

## Use of generative AI

The code for this part was written by Claude (Anthropic) and verified by the team, as stated in the report's AI disclosure.
