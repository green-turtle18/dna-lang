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

Full response: [ai_responses/chatgpt.md](ai_responses/chatgpt.md). In short:
1. Evolutionary constraint keeps particular codons.
2. CpG and methylation history.
3. Translational selection (tRNA supply, mRNA stability).
4. Protein function shapes amino-acid composition.
5. GC-biased gene conversion raises GC3. This is its main alternative
   explanation.
6. mRNA structure and splicing.

Its main warning is that matching overall GC doesn't match GC by codon
position or dinucleotides. Its "most important experiment" is to control
for GC1–3 and dinucleotide composition, then retrain.

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

### ChatGPT's claims

Two experiments were added to `src/verify_claims.py` to test ChatGPT's
suggestions: `composition_control()` and `enriched_codon_gc()`.
Results are in `results/disease/ai_claims/`.

| # | Claim | Test result | Verdict |
|---|---|---|---|
| E | Disease genes have higher GC3 (GC-biased gene conversion) | log2 +0.01, q = 0.12 | ❌ not supported |
| E | The enriched codons tend to be GC-rich | mean codon GC 0.59 vs 0.56, p = 0.68; G/C third base 72% vs 47% of significant codons, Fisher p = 0.17 | ❌ not supported |
| — | GC differs at codon positions 1 and 2 | GC1: q = 0.10. GC2 is *lower* in disease genes (q = 0.0001), which fits their more hydrophobic proteins | ❌ / opposite |
| 3 | The CG-containing 6-mers are mostly composition | 6-mer TF-IDF AUC 0.569 vs 16 dinucleotides 0.559 on lenmatch | ✅ supported |
| 4 | Controlling GC1–3 + CpG + dinucleotides should cut the AUC if the signal is compositional | composition alone 0.579; codon usage 0.598 → **0.558 [0.529, 0.587]** with composition regressed out | ◐ partly: part of the signal is compositional, but some survives (CI excludes 0.5) |
| D | A homology-aware split should lower the AUC if related genes leak | random 0.666 vs family split 0.661 | ❌ not supported (no meaningful leakage) |
| 1 | The arginine shift is genuine synonymous preference, not arginine abundance (RSCU) | CGN share *within* arginine higher in disease, q < 10⁻⁶; synonymous-only model AUC 0.573 | ✅ supported |
| 1 | Synonymous shuffling keeping the protein: the signal partly persists | 0.598 → 0.575 | ✅ supported |
| D | Amino-acid frequencies really differ (fewer Cys, more Gly) | Cys −0.18, Gly +0.06, both q < 10⁻⁶ | ✅ supported |
| D | More transmembrane domains in disease proteins | q = 0.11 | ❌ not supported |
| — | Amino-acid and synonymous information are complementary | each alone 0.578 / 0.573, combined 0.598 | ✅ supported |

**Can't be tested with sequence-only data.** These need outside
information: conservation and dS, methylation profiles, tRNA adaptation,
ribosome profiling, expression levels, GO-category matching, publication
counts, recombination rates, splicing motifs, and local genomic CpG.
ChatGPT also assumed canonical transcripts; our sequences are ORFeome
clones.

ChatGPT: 5 supported, 1 partly, 5 not supported, out of 11 testable
claims. Many of its points were experiment suggestions rather than
claims.

### Comparing the systems

| | Claude | ChatGPT | Gemini |
|---|---|---|---|
| Style | Bold, specific mechanisms | Hedged; methodology and confounders | Most specific, mechanistic, confident |
| Testable claims held | 8 / 13 (62%) | 5 + 1 partly / 11 (~50%) | 3 / 12 (25%), 2 opposite |
| Best call | CpG mutability; gene families | composition explains part of the signal; 6-mers ≈ dinucleotides | zinc fingers drive the cysteine effect; fewer paralogs |
| Main miss | CGA nonsense hotspots; collagen repeats | GC3 / GC-biased gene conversion | where the CpG excess sits (codon position, along the gene); TpG direction |
| Untestable share | low | high (many experiment suggestions) | medium |

All three agreed on the core: CpG mutability and ascertainment, gene-family
composition, and protein class. All three proposed GC3 or collagen
explanations that the data rejected. Note: Claude built the analysis, so
its score isn't blind.

### Gemini's claims

Full response: [ai_responses/gemini.md](ai_responses/gemini.md). Gemini gave
the most specific, mechanistic answer:
1. Purifying selection retains CpG.
2. Germline hypomethylation and CpG islands at the 5' end.
3. CpG hypermutability drives discovery.
4. Gene families explain the amino-acid signal.
5. Translational selection / GC3.

Three new tests were added to `src/verify_claims.py`: `cpg_by_position()`,
`paralog_families()` and `family_removal()`. Results are in
`results/disease/ai_claims/`.

| # | Claim | Test result | Verdict |
|---|---|---|---|
| 1 | Fewer TpG (CpG deamination product) in disease genes | TpG *higher* in disease, q < 10⁻⁶ | ❌ opposite |
| 1 | Fewer CpA (the other deamination product) | log2 −0.02, q < 10⁻⁶ | ✅ supported |
| 1 | CpG excess stronger at codon positions 1–2 (non-synonymous) than at synonymous sites | positions 1–2: log2 +0.062. Spanning the 3rd position into the next codon: **+0.127**, twice as strong | ❌ opposite |
| 2 | CpG excess concentrated in the first ~500 bp, diminishing downstream | first 500 bp +0.104, after 500 bp +0.073, last 500 bp +0.088, all p < 10⁻⁶ | ❌ not supported (slight 5' tilt, but it doesn't fade) |
| 4 | Removing ZNF/OR/KRTAP/collagen families sharply reduces the cysteine depletion | Cys log2 −0.18 → −0.07 | ✅ supported |
| 4 | …and the glycine enrichment | Gly +0.057 → +0.048 | ❌ not supported |
| 4 | …and drops the amino-acid-only AUC substantially | 0.578 → 0.574 | ❌ not supported |
| 4 | Disease genes have fewer close paralogs | in a multi-gene family: 18.5% vs 20.3%, p = 0.03 | ✅ supported (weak) |
| 4 | Collagens (Gly-X-Y repeats) enriched among disease genes | q = 0.12 | ❌ not supported |
| 5 | Higher GC3 at matched total GC | q = 0.12 | ❌ not supported |
| 5 | Enriched codons are C/G-ending, depleted ones A/T-ending | 72% vs 47% G/C-ending, Fisher p = 0.17 | ❌ not supported |
| Artefact | Paralog leakage is the strongest artefact | it assumed a random split; our homology-aware split gives the same AUC (0.661 vs 0.666) | ❌ not supported |

**Not testable with sequence-only data:** LOEUF/phyloP stratification, germline
methylation, which mutations in OMIM are CpG transitions, tAI and
expression, and exon count. The ascertainment-bias point is plausible but
also untestable here.

Gemini: 3 supported and 2 opposite, out of 12 testable claims.

Note: the tests in this table that aren't in `claim_tests.csv` report
p-values without FDR correction, so treat p = 0.03 as weak evidence.

### What to say in the talk

- AI explanations are **plausible-sounding hypotheses, not findings**. In
  Claude's case 8 of 13 held up (update this once the other systems are
scored).
- The ones that held are the **broad** ones: CpG mutability, gene families,
  protein class. The **specific** mechanisms often failed.
- **The more specific and confident the explanation, the more often it was
  wrong.** Gemini made the sharpest predictions (CpG at non-synonymous
  sites, concentrated at the 5' end, fewer TpG). Two came out the
  *opposite* way, and its overall hit rate was 25%.
- **ChatGPT's alternative explanation, GC-biased gene conversion raising
  GC3, was wrong for our data** (GC3 doesn't differ). Its *method*
  advice was the most useful thing either system said: after controlling
  for composition, the AUC drops from 0.60 to 0.56, so about 40% of the
  signal above chance is compositional. The rest survives. The appealing
  "nonsense-mutation hotspot" story for CGA codons wasn't significant, and
  "more codons one mutation away from a stop" went the *opposite* way.
- The proposal's point stands: **verify, don't trust**. Every claim needs a
  number computed from the data.
