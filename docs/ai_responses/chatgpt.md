# ChatGPT (OpenAI) response

The prompt from `docs/ai_explanations.md`, pasted into a fresh chat, saved word for word (only the citation icons from copying were removed). The verification is in `docs/ai_explanations.md`.

---

The most interesting aspect of your results is that the model appears to be detecting a **weak sequence-level signature of Mendelian disease genes that is not explained entirely by gene length or GC content**.

However, I would be careful about interpreting this as a disease-specific DNA signature. There are several possible explanations, and they are not mutually exclusive:

1. Disease genes may have distinctive evolutionary and functional constraints.
2. Their nucleotide composition may reflect differences in mutation and DNA repair.
3. Codon usage may reflect differences in gene expression and translational selection.
4. The patterns may arise from the way OMIM disease genes and non-disease genes were selected.

The distinction between genuine biology and dataset-construction artefacts is particularly important here. An AUC of 0.60 indicates a detectable statistical association, not necessarily a biological mechanism that causes disease.

## **1. Possible biological explanations and their testable predictions**

### **A. Evolutionary constraint and purifying selection**

Genes associated with Mendelian diseases are often subject to strong evolutionary constraints. Mutations in these genes can disrupt essential biological functions, making them more likely to be removed by purifying selection.

This could influence nucleotide composition over evolutionary time.

For example:

- Some amino acids may be preferentially retained because of the functional requirements of the encoded proteins.
- Particular synonymous codons may be retained because they influence translation efficiency, accuracy or mRNA stability.
- Codon usage could therefore reflect evolutionary constraints beyond the amino-acid sequence itself.

**Connection to your results:**

The higher frequency of CGT and CGC (arginine), and lower frequency of AGA and AGG (also arginine), is particularly interesting.

These codons encode the same amino acid, so the difference cannot be explained by amino-acid composition alone.

**Testable predictions:**

- Disease genes should exhibit stronger evolutionary conservation at synonymous sites than non-disease genes, if synonymous-site selection is involved.
- Disease genes should have lower synonymous substitution rates (dS), after controlling for gene expression and other relevant factors.
- The difference in synonymous codon usage should persist when comparing genes with similar protein sequences, expression levels and GC content.

An important qualification: strong evolutionary constraint does not automatically imply unusual codon usage. It needs to be demonstrated independently.

### **B. CpG enrichment and DNA methylation**

This is one of the more interesting explanations for your results.

A CpG dinucleotide is a cytosine followed by a guanine. In vertebrate genomes, methylated cytosines in CpG sites can undergo spontaneous deamination, producing thymine.

This makes CpG sites particularly susceptible to mutation.

Over evolutionary time, this process contributes to the depletion of CpG dinucleotides in many genomic regions.

Your observation:

- Disease genes: CpG observed/expected = 0.467
- Non-disease genes: CpG observed/expected = 0.438

suggests that disease-associated genes have somewhat greater CpG representation.

Possible explanations include:

1. Disease genes may be enriched in genomic regions with different methylation patterns.
2. Their promoters and regulatory regions may have different CpG-island distributions, indirectly reflecting genomic context.
3. Differences in mutation and repair processes may influence their coding-sequence composition.
4. Selection may preserve particular CpG-containing codons when they are functionally important.

**Testable predictions:**

- Disease genes should have higher CpG density even after controlling for local genomic CpG content.
- They may exhibit different methylation profiles, particularly in their gene bodies.
- If methylation-driven mutation is responsible, CpG-containing sequences should show corresponding differences in historical C-to-T substitution patterns.
- The association should be examined separately in CpG-rich and CpG-poor genomic regions.

One important distinction: a higher CpG ratio does not necessarily mean that the genes are more methylated. Sequence composition and methylation are different measurements.

### **C. Codon usage, translation efficiency and accuracy**

Different synonymous codons are translated by different tRNAs. Their availability and decoding efficiency can influence:

- Translation speed.
- Translation accuracy.
- mRNA stability.
- Co-translational protein folding.

This is particularly relevant for genes whose protein products must be produced in precise amounts or whose folding is functionally important.

For example, a gene encoding a transcription factor or a developmental regulator may be sensitive to changes in protein abundance.

If disease-associated genes have distinctive expression or dosage requirements, their codon usage might reflect these requirements.

**Connection to your results:**

Your synonymous-codon-only model achieves an AUC of 0.57. This suggests that some predictive information remains even after removing amino-acid identity.

The arginine substitution pattern is worth investigating:

| Codon | Disease-associated trend |
| ----- | ------------------------ |
| CGT   | Higher                   |
| CGC   | Higher                   |
| AGA   | Lower                    |
| AGG   | Lower                    |

The shift from AGA/AGG towards CGT/CGC could reflect differences in codon optimality, RNA structure or nucleotide-context preferences.

However, it is not sufficient to conclude that CGT and CGC are more efficiently translated in these genes. Human codon usage is strongly influenced by genomic nucleotide composition and GC-biased gene conversion.

**Testable predictions:**

- Disease genes should exhibit different tRNA adaptation indices (tAI) if translational selection explains the codon differences.
- They should exhibit different codon stability coefficients (CSCs), if codon-dependent mRNA decay is involved.
- Ribosome profiling should reveal differences in translation rates or ribosome occupancy.
- The codon-usage differences should correlate with measured protein abundance or expression levels.
- If translation accuracy is important, disease genes may preferentially use optimal codons at evolutionarily conserved amino-acid positions.

A particularly useful experiment would be to compare synonymous codon usage at highly conserved versus poorly conserved protein positions.

### **D. Differences in amino-acid composition and protein function**

Your amino-acid-only model achieves an AUC of 0.58. This is important because it suggests that the encoded proteins themselves carry information about disease association.

Mendelian disease genes are not a random sample of all human genes. They include many genes involved in:

- Transcriptional regulation.
- Development and differentiation.
- Cell signalling.
- Structural integrity.
- DNA repair.
- Metabolism.
- Ion transport.

These functional groups have different protein compositions.

For example, proteins involved in structural functions may have different requirements for glycine, proline and cysteine than proteins involved in transcriptional regulation.

This could explain some of your observations:

- Higher glycine codons (GGC, GGT).
- Lower proline codon CCA.
- Lower cysteine codons (TGT, TGC).

Cysteine is particularly interesting because it can form disulfide bonds, which are important for protein structure and stability. Glycine provides flexibility, while proline can introduce structural constraints in protein backbones.

However, these observations do not establish that disease proteins have fewer cysteine residues or more glycine residues without examining the actual amino-acid frequencies.

**Testable predictions:**

- Disease genes should show different amino-acid frequencies even after controlling for protein family and functional category.
- Their encoded proteins may have different proportions of intrinsically disordered regions, coiled-coils, transmembrane domains and conserved structural domains.
- Differences in cysteine and glycine frequency should be concentrated in particular protein domains or functional classes if they have a structural explanation.
- If the signal is mainly functional-category composition, matching genes by Gene Ontology categories or protein families should substantially reduce the AUC.

This last prediction is particularly important. It distinguishes a general disease-gene signature from a signature of the biological functions that happen to be overrepresented among known disease genes.

### **E. Mutation bias and GC-biased gene conversion**

This explanation deserves separate consideration from translational selection.

GC-biased gene conversion (gBGC) is a process associated with recombination in which G/C alleles are preferentially transmitted over A/T alleles.

It can produce GC enrichment without any direct functional advantage.

Human genes can therefore have different codon distributions because they occupy different genomic environments.

For example, genes located in regions with different recombination histories may exhibit different GC3 values (GC content at the third codon position).

This is relevant because many of your enriched codons end in G or C:

- CGT versus AGA: both encode arginine, but CGT has more GC nucleotides.
- CGC versus AGG: both encode arginine, but CGC is more GC-rich.
- GGC and GGT: glycine codons.
- GCG: alanine codon.

The pattern is not completely uniform, but there is a noticeable tendency towards GC-rich codons.

**Testable predictions:**

- Disease genes should have higher GC3 content, even when overall GC content is matched.
- Their intronic and flanking regions may also have higher GC content.
- Disease genes may occupy genomic regions with different recombination rates.
- Controlling for GC3, intronic GC content and recombination rate should reduce the predictive power of synonymous codon usage.

This is a particularly strong alternative hypothesis because research on human genes has shown that GC-biased gene conversion can explain a substantial fraction of synonymous codon usage variation.

### **F. mRNA secondary structure and splicing regulation**

A coding sequence is not merely a blueprint for a protein. It is also an RNA molecule with its own structural and regulatory properties.

Nucleotide sequences can influence:

- mRNA secondary structure.
- Splice-site recognition.
- Exonic splicing enhancers and silencers.
- RNA-binding protein interactions.
- mRNA stability.

Synonymous codon changes can therefore affect gene expression without changing the encoded protein.

This could be relevant to disease genes because incorrect splicing or altered expression can cause Mendelian disorders.

The CpG-containing hexamers might also be part of broader nucleotide-context patterns affecting RNA structure or regulatory motifs.

**Testable predictions:**

- Disease genes should have different predicted RNA folding energies or structural accessibility profiles.
- They may have different frequencies of known exonic splicing enhancer and silencer motifs.
- Their synonymous codon differences may be enriched near exon boundaries.
- The differences should be reduced when comparing only constitutive exons or controlling for exon position.
- If splicing is important, disease genes should exhibit different levels of exon inclusion or alternative splicing.

This explanation is especially interesting because it provides a possible biological reason for a sequence classifier to work even when the amino-acid sequence is removed.

## **2. Which patterns are likely to be dataset artefacts?**

I would distinguish between three types of potential artefacts:

1. Confounding caused by biological variables that were not matched.
2. Bias introduced by how disease genes were identified.
3. Statistical artefacts caused by sequence representation and model training.

Here is how I would assess your observations.

| Observed pattern                | Potential artefact                                 | My assessment                             |
| ------------------------------- | -------------------------------------------------- | ----------------------------------------- |
| Higher CGT and CGC              | GC3 and genomic GC composition                     | Strong possibility                        |
| Lower AGA and AGG               | GC-related codon bias                              | Strong possibility                        |
| Higher CpG O/E                  | Genomic location and methylation history           | Possible                                  |
| Higher CG-containing 6-mers     | Dinucleotide and GC composition                    | Very likely contributor                   |
| Higher glycine codons           | Amino-acid composition                             | Partly biological, potentially confounded |
| Lower cysteine codons           | Protein functional composition                     | Partly biological, potentially confounded |
| Lower serine and proline codons | Amino-acid composition and synonymous codon choice | Uncertain                                 |
| AUC of 0.60                     | Unmatched biological and annotation variables      | Potentially inflated                      |

### **A. GC content matching does not eliminate GC-related confounding**

This is probably the most important technical issue.

Matching overall GC content does not guarantee matching GC content at individual codon positions.

Consider two genes with identical overall GC content:

- Gene A has high GC content at the first and second codon positions but low GC3.
- Gene B has low GC content at the first and second positions but high GC3.

They can have the same overall GC content but very different codon distributions.

Furthermore, matching GC content does not match dinucleotide frequencies.

Therefore, the model could be learning:

- GC3 differences.
- CpG frequencies.
- Other dinucleotide frequencies.
- Local nucleotide composition.

**How to test this:**

Train another classifier after controlling for:

1. Overall GC content.
2. GC1, GC2 and GC3 separately.
3. CpG O/E.
4. All 16 dinucleotide frequencies.

If the AUC drops substantially towards 0.50, much of the original signal was compositional rather than disease-specific.

### **B. OMIM ascertainment bias**

This is arguably the biggest dataset-construction concern.

Your labels are:

- Disease: gene has a known Mendelian disease association in OMIM.
- Non-disease: gene has no known disease association in OMIM.

But these labels do not necessarily represent:

- Disease-causing gene.
- Healthy gene.

A gene without an OMIM association may simply not have been sufficiently studied.

Some genes are much more likely to be investigated because:

- They are associated with recognizable clinical syndromes.
- They have well-characterized biological functions.
- Their mutations produce strong phenotypes.
- They have been studied extensively in model organisms.
- They have known clinical relevance.

This creates ascertainment bias.

There is also a possibility that the disease class is enriched for particular gene families, such as developmental regulators or transcription factors.

**Testable predictions:**

- Disease genes should have higher publication counts and greater research attention.
- They may have more known variants and more extensive functional annotations.
- The model's AUC should decrease when disease genes are compared against non-disease genes with similar publication counts and annotation coverage.
- The model should perform less well on recently discovered disease genes than on extensively studied ones, if annotation history contributes to the signal.

Importantly, a gene with no known Mendelian disease association may still be essential or disease-relevant.

### **C. Transcript selection and sequence representation**

Another possible artefact is how the coding sequences were obtained.

A single gene can have multiple transcript isoforms, alternative exons and different translation start sites.

If you selected the canonical transcript for each gene, you may be introducing systematic differences between the two classes.

For example:

- Some genes have many transcript isoforms.
- Some have unusually long coding sequences.
- Some have alternatively spliced exons enriched in particular codons.

**Testable predictions:**

- The classification performance should be similar when using alternative representative transcripts.
- The signal should persist when comparing constitutive coding regions only.
- The results should remain similar when genes are represented using the same transcript-selection procedure and transcript-quality criteria.

If the AUC changes dramatically depending on transcript selection, the classifier may be learning transcript-annotation characteristics rather than gene-level biological properties.

### **D. Homology and gene-family leakage**

This is a machine-learning-specific concern.

Suppose a disease-associated gene and a closely related non-disease gene belong to the same protein family.

Their sequences may share considerable similarity.

If related genes appear in both training and test sets, the model might learn family-specific sequence patterns rather than generalizable disease-associated patterns.

**Testable predictions:**

- Performance should decrease when genes are split into training and test sets by sequence similarity or protein family.
- A homology-aware split should produce a lower AUC than a random gene-level split if leakage was contributing.
- The model should retain predictive performance across completely unseen protein families if it has learned general biological properties.

This is particularly important for k-mer models.

A six-letter sequence can be shared across related genes, so the presence of similar k-mers in training and test data can make classification easier.

## **3. My interpretation of your specific observations**

I would rank the findings as follows.

**1. CG-containing 6-mer enrichment**

Likely confounded

The enriched hexamers may largely reflect elevated CpG representation and local GC composition. This is not necessarily an independent biological signal.

**2. CGT/CGC versus AGA/AGG**

Likely confounded

This is an interesting synonymous-codon difference, but GC3, mutation bias and gBGC should be investigated before attributing it to translational selection.

**3. Higher CpG O/E**

Biologically interesting

This could reflect differences in genomic context, historical methylation-associated mutation, or selection. It needs to be checked against local genomic background.

**4. Amino-acid composition**

Likely functional signal

The AUC of 0.58 suggests that disease genes encode proteins with somewhat different compositional characteristics. Functional-category enrichment is a plausible explanation.

**5. Synonymous codon usage**

Needs investigation

The AUC of 0.57 suggests information beyond amino-acid composition, but this could arise from genomic composition rather than selection on translation.

**6. Full codon usage**

Combined signal

The AUC of 0.60 suggests that amino-acid and synonymous-codon information may provide complementary predictive information. It does not establish that the two mechanisms are independent.

## **4. What experiments would I prioritise?**

Rather than immediately proposing many biological explanations, I would perform a sequence of controlled experiments.

| Experiment                            | What it tests                      | Expected result if hypothesis is correct                  |
| ------------------------------------- | ---------------------------------- | --------------------------------------------------------- |
| Match GC1, GC2, GC3                   | GC-related confounding             | AUC decreases if GC3 drives the signal                    |
| Match dinucleotide frequencies        | CpG and nucleotide-context effects | k-mer advantage decreases                                 |
| Use synonymous codon frequencies only | Synonymous information             | AUC remains above 0.5                                     |
| Match amino-acid composition          | Protein composition                | AUC decreases if composition drives prediction            |
| Match GO categories                   | Functional enrichment              | AUC decreases if functional composition drives prediction |
| Match expression levels               | Expression-related codon bias      | Synonymous signal decreases if expression explains it     |
| Match publication counts              | OMIM ascertainment bias            | AUC decreases if study intensity contributes              |
| Perform homology-aware splitting      | Sequence leakage                   | AUC decreases if related genes leak across splits         |

### **Three particularly useful additional analyses**

**1. Calculate relative synonymous codon usage (RSCU).**

Raw codon frequencies can be misleading because amino acids occur at different frequencies.

RSCU measures the frequency of a particular codon relative to the expected frequency if all synonymous codons for that amino acid were used equally.

For codon \\(j\\):

\\[ RSCU_j=\frac{X_j}{\frac{1}{n}\sum\_{i=1}^{n}X_i} \\]

where \\(n\\) is the number of synonymous codons for that amino acid.

This would help determine whether your arginine result reflects genuine synonymous-codon preference rather than simply differences in arginine abundance.

**2. Perform amino-acid-preserving synonymous shuffling.**

This would be particularly informative.

For every gene:

- Preserve its amino-acid sequence exactly.
- Randomly shuffle synonymous codons.
- Generate multiple alternative coding sequences.
- Recalculate the classifier's performance.

If the original disease-associated signal disappears after shuffling, synonymous codon arrangement or usage contains information beyond the protein sequence.

If it persists, amino-acid composition is likely contributing substantially.

You could also shuffle synonymous codons while preserving the gene's overall codon counts. That would distinguish codon-frequency effects from local sequence-order effects.

**3. Compare against a composition-aware null model.**

Generate synthetic sequences that preserve:

- Gene length.
- Amino-acid sequence.
- GC content at each codon position.
- Dinucleotide composition, where possible.

Then compare the original model's performance against the performance on these synthetic sequences.

This would help establish whether the model is learning something beyond basic sequence composition.

## **5. Relevant scientific literature**

These papers would be useful when interpreting your results and building the biological justification for your project.

1\. **Natural Selection Shapes Codon Usage in the Human Genome (2020).** Demonstrates that synonymous codons can be subject to natural selection in humans, including in dosage-sensitive genes and DNA-damage-response genes. [Read the paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7332603/?utm_source=chatgpt.com).

2\. **Recombination, meiotic expression and human codon usage (2017).** Particularly important for your project. Shows that GC-biased gene conversion can explain much of the variation in human synonymous codon usage. [Read the paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5576983/?utm_source=chatgpt.com).

3\. **Codon usage bias (2022).** A review of the mechanisms influencing codon usage, including mutation, selection, GC content, expression and tRNA availability. [Read the review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8613526/?utm_source=chatgpt.com).

4\. **The impact of epigenetic information on genome evolution (2021).** Useful for understanding CpG depletion and how historical DNA methylation can influence present-day sequence composition. [Read the paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8059978/?utm_source=chatgpt.com).

5\. **Codon optimality, bias and usage in translation and mRNA decay (2018).** Explains how synonymous codons can affect translation efficiency and mRNA stability. [Read the review](https://www.nature.com/articles/nrm.2017.91?utm_source=chatgpt.com).

## **6. Overall conclusion**

My preferred interpretation is that your classifier is detecting a mixture of three signals:

1\. **Protein-level functional differences:** Mendelian disease genes encode proteins with somewhat different amino-acid compositions because of their biological roles.

2\. **Synonymous sequence differences:** Disease-associated genes exhibit different codon preferences, potentially reflecting evolutionary constraints, expression requirements or genomic nucleotide composition.

3\. **Dataset-associated differences:** OMIM ascertainment, genomic location, GC3, CpG composition and gene-family representation may contribute to the observed classification performance.

I would be particularly cautious about claiming that higher CpG content or increased CGT/CGC usage is a consequence of disease-associated evolutionary selection. Those are hypotheses, not established conclusions.

**The most important experiment, in my opinion, is to control for GC3 and dinucleotide composition and then retrain the classifier.**

If you still obtain an AUC meaningfully above 0.5, you have stronger evidence that there is additional sequence information associated with Mendelian disease genes.

And if the AUC falls substantially, that is still a valuable result: it tells you that the original machine-learning model was detecting genomic compositional structure rather than a disease-specific sequence signature.

Ultimately, the scientifically interesting question is not simply whether disease genes can be distinguished from non-disease genes using DNA sequence. It is:

**What biological properties of Mendelian disease-associated genes are encoded in their nucleotide sequences, and how much of that information remains after accounting for ordinary genomic composition and annotation bias?**
