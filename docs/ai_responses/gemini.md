# Gemini (Google) response

The prompt from `docs/ai_explanations.md`, pasted into a fresh chat, saved word for word. The verification is in `docs/ai_explanations.md`.

---

An AUC of ~0.60 on length- and GC-matched coding sequences is a realistic, modest signal—and the specific features your models picked up point to a tight cluster of evolutionary, mutational, and compositional mechanisms rather than a single "disease code."

## 1 & 2. Biological Explanations and Testable Predictions

### 1. Purifying Selection and CpG Retention

In mammalian genomes, methylated CpG dinucleotides mutate to TpG/CpA via spontaneous C-to-T deamination at 10–50× the background rate. Over evolutionary time, unconstrained sequences lose CpGs rapidly, whereas strongly constrained genes—which make up the bulk of Mendelian disease genes—retain them because purifying selection removes deleterious mutations (especially at functionally critical arginine residues encoded by `CGT`/`CGC`).

* **Testable prediction:** Disease genes should have **lower** TpG and CpA dinucleotide frequencies (the deamination products) than non-disease genes, and the CpG O/E excess should be **stronger at nonsynonymous (1st/2nd codon) sites** than at 4-fold degenerate synonymous sites. Stratifying the test set by evolutionary constraint (e.g., LOEUF or phyloP) should collapse the AUC toward 0.50.

### 2. Germline Hypomethylation and CpG-Island Overlap

Broadly expressed and developmental genes (heavily enriched in OMIM) frequently have unmethylated CpG islands extending from their promoters into the first exon. Because unmethylated CpGs do not deaminate at high rates, coding sequences overlapping CpG islands maintain elevated CpG O/E—including at synonymous sites (`GCG`, `CGC`, `CGT`). This directly explains why synonymous codon choice alone reaches AUC 0.57 and why CG-containing 6-mers dominate.

* **Testable prediction:** The CpG O/E excess and CG-rich 6-mer enrichment in disease genes should be **concentrated in the 5' end (first exon / first ~500 bp)** of the coding sequence and **diminish or disappear in downstream exons**. Disease genes should also show **lower germline (sperm/oocyte) DNA methylation** across their coding exons.

### 3. Mutational Target Size (CpG Hypermutability)

Even when two genes are equally capable of causing a Mendelian phenotype, the gene with more CpG dinucleotides—particularly `CGN` arginine codons (`CGT`, `CGC`, `CGA`)—has a much higher *de novo* mutation rate for missense and nonsense variants. Higher mutation rates produce more affected patients, making those genes more likely to be discovered and cataloged in OMIM.

* **Testable prediction:** Within your disease-gene set, **CpG-transition variants** (e.g., `CGT`/`CGC` to `TGT`/`TGC` or `CAT`/`CAC`) should account for a **disproportionately high fraction** of known pathogenic OMIM mutations relative to their frequency in the coding sequence.

### 4. Gene-Family and Protein-Domain Composition

Amino-acid composition alone gives AUC 0.58, and the depleted/enriched residues map cleanly onto major gene-family differences:

* **Depleted in disease genes (`TGT`/`TGC` Cys, `TCA`/`AGT`/`TCT` Ser, `CCA` Pro, `AGA`/`AGG` Arg):** Non-disease genes are heavily enriched for large, rapidly evolving, functionally redundant families—especially **C2H2/KRAB zinc-finger proteins** (very cysteine-rich), **keratin-associated proteins** (cysteine- and serine-rich), and **olfactory receptors** (single-exon GPCRs with distinct codon/AA bias).
* **Enriched in disease genes (`GGC`/`GGT` Gly, `ATC`/`ATT` Ile, `TAC` Tyr):** Mendelian disease genes are enriched for **collagens** (glycine-rich `GGN` repeats), core metabolic enzymes, and structured transmembrane/channel proteins.
* **Testable prediction:** Removing or down-weighting **C2H2 zinc fingers, olfactory receptors, keratin-associated proteins, and collagens** should sharply reduce the depletion of `TGT`/`TGC` (Cys) and enrichment of `GGC`/`GGT` (Gly), and drop the amino-acid-only AUC substantially. Disease genes should also have **fewer close paralogs** than non-disease genes.

### 5. Translational Selection / GC3-vs-GC12 Partitioning

Matching total GC content does not match how GC is distributed across codon positions. Many of your enriched codons (`CGC`, `ATC`, `TAC`, `GGC`, `GCG`) are C/G-ending codons preferred in highly expressed, evolutionarily ancient human genes, whereas many depleted codons (`AGA`, `AGG`, `TCA`, `AGT`, `TCT`, `CCA`) are A/T-ending or less optimal.

* **Testable prediction:** Even at identical total GC%, disease genes should have **higher GC3** (GC at the 3rd codon position) and **higher tRNA Adaptation Index (tAI)** than non-disease genes, along with **higher median expression** across tissues.

---

## 3. Which Patterns Are Likely Dataset Artefacts?

Several of the signals are likely driven by how the labels and train/test splits were constructed rather than intrinsic "disease biology":

* **Homology / Paralog Leakage (Strongest likely artefact for the AA and 6-mer signals):** If genes were split randomly into train and test sets rather than clustered by sequence similarity (e.g., 30% identity or paralog family), then large non-disease families (hundreds of olfactory receptors and KRAB-ZFPs) and disease families (collagens, ion channels) appear in *both* sets. The model can memorize family-specific signatures (like Cys-rich zinc-finger motifs or Gly-rich collagen repeats), inflating the 0.58 AA-only AUC and 6-mer performance.
* **Open-World Labeling & Ascertainment Bias (Artefact in the CpG signal):** "Non-disease" in OMIM means *"not yet linked to a Mendelian disease"*—a mixture of dispensable genes, embryonic-lethal genes (which rarely present as live-born Mendelian patients), and undiscovered disease genes. The CpG enrichment is partly an ascertainment artefact: CpG-rich genes mutate more frequently and are therefore discovered in the clinic earlier.
* **Unmatched Exon Structure & Splicing Constraints:** Matching only total CDS length and total GC% ignores **exon count**. Non-disease genes contain far more single-exon genes (e.g., olfactory receptors, intronless retrogenes), whereas multi-exon genes carry strong codon and k-mer biases near splice junctions (exonic splicing enhancers/silencers). That structural mismatch alone can create a synonymous-codon signal.
