# AI-generated explanations vs verified results (proposal task 6)

The proposal asks us to have several AI systems interpret our patterns,
compare their explanations, and check them against the computed results.

## How to do it (about 30 minutes)

1. Paste **the prompt below, unchanged**, into a fresh chat in each AI
   system: Claude is done below; add ChatGPT, Gemini, and one more if you
   like. Fresh chat, no extra context, so every system answers the same
   question.
2. Save each answer under "Responses" in this file, word for word.
3. Pull out every **specific, testable** claim. "Disease genes have more X"
   is testable; "the model learns complex biological patterns" is not.
4. For each claim, find the matching test in `results/disease/ai_claims/claim_tests.csv`.
   If none matches, add one to `CLAIMS` in `src/verify_claims.py` (a
   function computing a number per gene plus the expected direction) and
   rerun `python src/verify_claims.py`.
5. Fill in the verification table at the bottom.

## The prompt (identical for every AI system)

```text
We trained machine-learning models to classify human protein-coding genes as
"disease" (has a Mendelian disease link in OMIM) or "non-disease" (no known
disease link) using only the DNA coding sequence, treated as text (k-mers and
codons). On a test set where both classes have the same length and GC-content
distribution, models reach ROC-AUC about 0.60 (0.5 = chance).

Patterns found (disease vs non-disease genes, length and GC matched):
- Codons more frequent in disease genes: CGT and CGC (arginine), ATC and ATT
  (isoleucine), TAC (tyrosine), GGC and GGT (glycine), GCG (alanine).
- Codons less frequent in disease genes: TGT and TGC (cysteine), AGA and AGG
  (arginine), TCA, AGT and TCT (serine), CCA (proline).
- CpG observed/expected ratio is higher in disease genes (0.467 vs 0.438).
- Amino-acid composition alone (20 features) gives AUC 0.58; synonymous codon
  choice alone, with the protein removed, gives AUC 0.57; full codon usage
  gives AUC 0.60.
- The 6-letter k-mers most enriched in disease genes often contain "CG"
  (e.g. ACTTCG, TACGTC, TTGGCG).

Questions:
1. What biological explanations could account for these patterns?
2. For each explanation, state a specific, testable prediction about the
   sequences (what should be higher or lower in disease genes).
3. Which of the patterns do you think are artefacts of how the dataset was
   built rather than biology?
```

## Responses

### Claude (Anthropic)

> Caveat for the slides: this Claude instance also built the analysis, so
> its answer is not blind. Treat it as one data point and weight the other
> systems' fresh answers more.

1. **CpG mutability and ascertainment.** CpG sites mutate about 10x faster
   than other sites (methylated cytosine deaminates to thymine). Genes with
   more CpG, and especially CG-containing arginine codons (CGA, CGC, CGG,
   CGT), collect more pathogenic variants, so their disease links are found
   sooner. *Predictions:* higher CpG observed/expected in disease genes;
   a higher share of arginine encoded by CGN rather than AGR codons; more
   CGA codons, since CGA→TGA creates a stop codon.
2. **Gene-family composition.** Large, young families such as KRAB zinc
   fingers are cysteine-rich, often redundant, and rarely Mendelian.
   *Predictions:* fewer cysteines and fewer C2H2 zinc-finger motifs in
   disease genes.
3. **Protein class.** Membrane channels, transporters and enzymes are
   over-represented among Mendelian genes. *Predictions:* more hydrophobic
   residues and more transmembrane-like segments in disease genes.
4. **Disorder and serine.** Serine-rich, low-complexity disordered regions
   are common in signalling and regulatory proteins with redundant
   paralogs. *Predictions:* fewer serines and fewer low-complexity regions
   in disease genes.
5. **Structural proteins.** Collagens and other structural proteins are
   glycine-rich and classic disease genes. *Predictions:* more glycine and
   more Gly-X-Y repeats in disease genes.
6. **Repeat expansions** (Huntington's, the spinocerebellar ataxias) are
   famous, but involve few genes. *Prediction:* no overall difference in
   poly-glutamine or amino-acid runs.
7. **Artefacts.** Part of the signal is likely "how well studied a gene
   is" rather than biology, and the codon-choice signal may reflect
   regional GC (isochores). *Prediction:* GC at codon position 3 differs
   between classes.

### ChatGPT (OpenAI)

*(paste response here)*

### Gemini (Google)

*(paste response here)*

## Verification

The tests were run on the lenmatch subset (8,902 genes, length and GC
matched). They use a Mann-Whitney U test with Benjamini-Hochberg FDR across
all claims. Full table: `results/disease/ai_claims/claim_tests.csv`.

### Claude's claims

| # | Claim | Test result | Verdict |
|---|---|---|---|
| 1 | Higher CpG obs/exp in disease genes | 0.467 vs 0.438, q < 10⁻⁶ | ✅ supported |
| 1 | More CGN (vs AGR) arginine codons | log2 effect +0.09, q < 10⁻⁶ | ✅ supported |
| 1 | More CGA codons (nonsense hotspot) | +0.02, q = 0.07 | ❌ not supported |
| 2 | Fewer cysteines | log2 effect −0.18, q < 10⁻⁶ | ✅ supported |
| 2 | Fewer C2H2 zinc-finger motifs | strongly lower, q < 10⁻⁶ | ✅ supported |
| 3 | More hydrophobic residues | +0.04, q < 10⁻⁶ | ✅ supported |
| 3 | More transmembrane-like segments | +0.10, q = 0.11 | ❌ not supported |
| 4 | Fewer serines | −0.09, q < 10⁻⁶ | ✅ supported |
| 4 | Fewer low-complexity regions | opposite direction, q = 0.15 | ❌ not supported |
| 5 | More glycine | +0.06, q < 10⁻⁶ | ✅ supported |
| 5 | More Gly-X-Y collagen repeats | q = 0.12 | ❌ not supported |
| 6 | No difference in amino-acid runs | runs ≥5: q = 0.10; CAG/CAA runs slightly *lower* in disease (q = 0.007) | ✅ mostly (no excess of repeats in disease genes) |
| 7 | GC3 differs (isochore artefact) | q = 0.12 | ❌ not supported |

Claude: 8 of 13 testable claims held, one of them a "no difference"
prediction. The broad mechanisms hold up: CpG mutability, gene family, and
protein class. Several sharper, more specific
predictions fail (CGA hotspots, TM segments, collagen repeats, low
complexity).

### Other systems (fill in)

| System | Claim | Test (row in claim_tests.csv or new) | Result | Verdict |
|---|---|---|---|---|
| ChatGPT | | | | |
| Gemini | | | | |

### What to say in the talk

- AI explanations are **plausible-sounding hypotheses, not findings**. In
  Claude's case 8 of 13 held up (update this once the other systems are
scored).
- The ones that held are the **broad** ones: CpG mutability, gene families,
  protein class. The **specific** mechanisms often failed. The appealing
  "nonsense-mutation hotspot" story for CGA codons wasn't significant, and
  "more codons one mutation away from a stop" went the *opposite* way.
- The proposal's point stands: **verify, don't trust**. Every claim needs a
  number computed from the data.
