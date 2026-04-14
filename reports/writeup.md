# Epigenome-Wide Association Study of Alzheimer's Disease Pathology
## DNA Methylation and Braak Stage in Human Frontal Cortex

**Author:** Alexander Hannah, PhD  
**Date:** April 2026  
**Dataset:** GSE59685 (GEO, NCBI)  
**Code:** github.com/YOUR_USERNAME/alzheimers-methylation-ewas

---

## Abstract

We performed an epigenome-wide association study (EWAS) of DNA methylation in
human frontal cortex tissue to identify CpG sites associated with Alzheimer's
disease (AD) pathology severity. Using publicly available 450k array data from
80 donors spanning Braak stages 0–6, we tested 145,089 CpG sites for
association with Braak stage using linear regression implemented in PySpark.
Two CpG sites reached Bonferroni-corrected significance (p < 3.45 × 10⁻⁷):
cg21022002 and cg07227671, both showing hypomethylation with increasing AD
pathology. A ridge regression model trained on the top 50 CpGs by p-value
predicted Braak stage with a cross-validated R² of 0.577 ± 0.196, suggesting
that a multi-CpG methylation biosignature carries meaningful prognostic
information. These findings are consistent with published EWAS literature and
demonstrate the feasibility of large-scale epigenomic analysis using distributed
computing frameworks.

---

## 1. Background

Alzheimer's disease is the most common neurodegenerative disorder, affecting
over 55 million people worldwide. The neuropathological hallmarks of AD —
amyloid plaques and neurofibrillary tangles — accumulate progressively, and
their burden is quantified post-mortem using the Braak staging system, a
six-point ordinal scale (0–VI) reflecting the extent of tau pathology across
brain regions.

Epigenome-wide association studies (EWAS) have emerged as a powerful approach
for identifying disease-associated DNA methylation changes at scale. Unlike
genome-wide association studies (GWAS), which measure fixed genetic variants,
EWAS measures DNA methylation — a dynamic epigenetic modification that can
change in response to aging, environment, and disease. Methylation at cytosine-
phosphate-guanine (CpG) dinucleotides is quantified as a beta value between 0
(unmethylated) and 1 (fully methylated), providing a continuous, quantitative
measure of epigenetic state at each of hundreds of thousands of genomic
positions.

Previous EWAS of AD have identified consistent methylation differences at loci
including ANK1, BIN1, RHBDF2, and HOXA cluster genes, particularly in cortical
brain regions. The present analysis applies these established methods to a
publicly available dataset using a modern distributed computing stack,
demonstrating both the biological signal in the data and the scalability of
the analytical approach.

---

## 2. Data

### 2.1 Dataset
Data were obtained from the NCBI Gene Expression Omnibus (GEO) under accession
**GSE59685**. This dataset comprises Illumina Infinium HumanMethylation450
BeadChip (450k array) data generated from post-mortem brain tissue across
five regions: frontal cortex, superior temporal gyrus, entorhinal cortex,
cerebellum, and whole blood.

### 2.2 Sample Selection
Analysis was restricted to **frontal cortex** samples, consistent with prior
EWAS literature emphasizing this region's strong methylation signal in AD.
Samples flagged as "Exclude" in either AD disease status or Braak stage were
removed. The final analytical cohort comprised **80 donors** with Braak stages
ranging from 0 to 6.

| Braak Stage | N Samples |
|-------------|-----------|
| 0           | 8         |
| 1           | 10        |
| 2           | 5         |
| 3           | 6         |
| 4           | 4         |
| 5           | 12        |
| 6           | 35        |

The distribution is right-skewed, with Braak stage 6 (end-stage AD)
representing 44% of samples — a common feature of post-mortem brain bank
cohorts, where advanced-stage donors are disproportionately represented.

### 2.3 CpG Filtering
Of 485,577 CpG sites on the array, sites were removed if they had:
- Missing data in more than 5% of samples
- Variance below 0.001 across samples (near-constant methylation)

After filtering, **145,089 CpG sites** were retained for analysis.

---

## 3. Methods

### 3.1 Computing Environment
All analyses were performed in Python 3.14 using the following core libraries:

| Library | Version | Role |
|---------|---------|------|
| PySpark | 4.1.1 | Distributed regression across CpG sites |
| pandas | — | Data wrangling and preprocessing |
| scipy | — | Linear regression (per CpG) |
| scikit-learn | — | Ridge regression predictive model |
| matplotlib/seaborn | — | Visualization |

### 3.2 Data Processing Pipeline
Raw beta values were loaded from the GEO FTP server using GEOparse and
processed through the following pipeline:

1. **Download** — GSE59685 soft file and beta matrix retrieved programmatically
2. **Metadata parsing** — sample characteristics extracted and parsed into
   structured format
3. **Filtering** — restricted to frontal cortex; excluded flagged samples
4. **QC filtering** — low-variance and high-missingness CpGs removed
5. **Format conversion** — wide matrix (CpGs × samples) converted to long
   format (one row per CpG–sample pair) for efficient Spark processing
6. **Parquet serialization** — cleaned data saved in columnar Parquet format

### 3.3 EWAS — Linear Regression in PySpark
For each of the 145,089 retained CpG sites, a simple linear regression was
fitted: beta_i ~ braak_stage_i + ε

Where `beta_i` is the methylation beta value for sample *i* and
`braak_stage_i` is the continuous Braak stage (0–6).

Regression was implemented as a **Pandas UDF** (User Defined Function) applied
via `groupBy("cpg").apply()` in PySpark, enabling parallel execution across all
CpG groups simultaneously. For each CpG the following statistics were extracted:

- **Slope** — change in beta value per unit increase in Braak stage
- **Intercept**
- **P-value** — from the two-tailed t-test on the slope coefficient
- **R²** — proportion of variance in methylation explained by Braak stage

### 3.4 Multiple Testing Correction
Given 145,089 simultaneous tests, Bonferroni correction was applied: threshold = 0.05 / 145,089 = 3.45 × 10⁻⁷

Bonferroni is the most conservative correction available and controls the
family-wise error rate — the probability of any false positive across all tests.
This is the standard approach in published EWAS.

### 3.5 Predictive Modeling
A ridge regression model was trained to predict Braak stage from the top 50
CpG sites by p-value. Ridge regression applies L2 regularization, penalizing
large coefficients to prevent overfitting — particularly important given the
high feature-to-sample ratio (50 features, 80 samples).

Features were standardized prior to modeling. Regularization strength (alpha)
was selected via 5-fold cross-validation from a grid of candidate values
{0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}. Model performance was evaluated on a
held-out test set (20% of samples) and via 5-fold cross-validation on the
full dataset.

---

## 4. Results

### 4.1 EWAS Results
Of 145,089 CpG sites tested, **2 reached Bonferroni-corrected significance**
(p < 3.45 × 10⁻⁷):

| CpG | Slope | P-value | R² | Direction |
|-----|-------|---------|-----|-----------|
| cg21022002 | -0.0103 | 1.02 × 10⁻⁷ | 0.306 | Hypomethylated |
| cg07227671 | -0.0096 | 2.33 × 10⁻⁷ | 0.292 | Hypomethylated |

Both significant CpGs show **negative slopes**, indicating that methylation
decreases as Braak stage increases. This hypomethylation pattern is consistent
with the broader AD methylation literature. Each CpG explains approximately
29–31% of variance in Braak stage individually (R² ~ 0.30), which is a
meaningful effect size for a single CpG in a bulk tissue EWAS.

The Manhattan plot reveals several additional chromosomal regions with
suggestive associations (p ~ 10⁻⁵) on chromosomes 1, 5, 7, 9, 11, 17, and 19
that do not reach Bonferroni significance, likely due to insufficient statistical
power at this sample size.

The volcano plot shows a modest asymmetry toward the negative slope direction,
suggesting a global trend toward hypomethylation with increasing AD pathology
across the dataset — consistent with published reports of widespread
hypomethylation in AD brain tissue.

### 4.2 Predictive Model
A ridge regression model trained on the top 50 CpGs predicted Braak stage with
the following performance:

| Metric | Value |
|--------|-------|
| Test set R² | 0.704 |
| Test set MAE | 1.100 Braak stages |
| Cross-validated R² (5-fold) | 0.577 ± 0.196 |
| Selected alpha | 100.0 |

The cross-validated R² of 0.577 indicates that methylation patterns at 50 CpG
sites collectively explain approximately 58% of variance in Braak stage — a
meaningful predictive signal given the dataset size. The mean absolute error of
1.1 Braak stages reflects moderate precision on a 0–6 scale.

The selected alpha of 100.0 indicates that moderately strong regularization
was optimal, consistent with the high feature-to-sample ratio and the modest
individual effect sizes of the CpG features.

The coefficient plot reveals that `cg12513911` and `cg14620572` carry the
largest weights in the predictive model, with both Bonferroni-significant CpGs
(`cg21022002`, `cg07227671`) also contributing negative coefficients consistent
with their hypomethylation direction.

---

## 5. Discussion

This analysis identified two CpG sites significantly associated with AD
neuropathology severity in frontal cortex, both showing progressive
hypomethylation with increasing Braak stage. The findings are consistent with
the established biology of AD — widespread epigenetic dysregulation, including
both hypo- and hypermethylation, has been documented across multiple cortical
regions in multiple independent cohorts.

The multi-CpG predictive model achieved a cross-validated R² of 0.577,
supporting the concept of a DNA methylation biosignature for AD pathology
staging. This is consistent with the conclusions of published EWAS studies,
including work from the London Brain Bank cohorts, which found that no single
CpG had sufficient standalone clinical utility but that multi-CpG signatures
showed promise as prognostic biomarkers.

The use of PySpark for the EWAS regression step demonstrates the scalability
of this analytical framework. Running 145,089 independent regressions in
parallel via a Pandas UDF reduced computation time substantially compared to
a sequential Python implementation, and the approach scales naturally to larger
datasets (e.g., EPIC array with ~850,000 CpGs) or larger sample sizes without
changes to the core analytical code.

---

## 6. Limitations

**Sample size.** With 80 frontal cortex samples, this analysis is underpowered
relative to published EWAS, which typically use 200–1,000+ donors. Small sample
size reduces the number of CpGs reaching Bonferroni significance and increases
variance in cross-validated model performance (CV R² SD = ±0.196).

**Imbalanced Braak distribution.** Braak stage 6 represents 44% of samples,
with very few donors at intermediate stages (2–4). This imbalance may reduce
the model's ability to distinguish intermediate disease stages and could
inflate apparent effect sizes at high Braak stages.

**Bonferroni conservativeness.** Bonferroni correction is the most stringent
multiple testing approach and likely produces false negatives — real signals
that don't reach the threshold. Several suggestive loci (p ~ 10⁻⁵) visible in
the Manhattan plot may represent true associations that would reach significance
in a larger cohort. Alternative corrections such as Benjamini-Hochberg FDR
would identify more hits at the cost of accepting some false positives.

**Bulk tissue methylation.** The 450k array measures average methylation across
all cell types in the tissue sample. Frontal cortex contains neurons, glia,
endothelial cells, and other cell types with very different methylation
profiles. Cell type composition varies between donors and with disease stage,
which can confound methylation associations. Cell type deconvolution was not
applied in this analysis.

**No gene annotation.** The significant CpGs were not annotated to specific
genes in this analysis. Future work should map CpG positions to nearby genes
and regulatory elements to assess biological plausibility and compare with
published AD-associated loci (ANK1, BIN1, HOXA cluster, PM20D1).

**Cross-sectional design.** This dataset captures a single timepoint per donor.
Longitudinal designs — such as the ADNI cohort used in the paper that motivated
this project — provide stronger causal inference by measuring methylation before
disease progression occurs.

---

## 7. Future Directions

- Annotate significant CpGs to genes using the Illumina 450k manifest
- Apply FDR correction to identify additional candidate loci
- Add cell type deconvolution as a covariate
- Extend analysis to other brain regions in the dataset (temporal gyrus,
  entorhinal cortex) and compare findings across regions
- Apply the same pipeline to a larger dataset (e.g., the BDR cohort,
  N=590, EPIC array) to increase power
- Explore deep learning approaches (e.g., PyTorch) for Braak stage prediction
  when larger training datasets are available

---

## 8. Conclusion

This project demonstrates an end-to-end EWAS pipeline for Alzheimer's disease
using publicly available DNA methylation data and a modern distributed computing
stack. Two CpG sites were identified as significantly associated with AD
pathology severity in frontal cortex, both showing hypomethylation consistent
with the published literature. A multi-CpG ridge regression model predicted
Braak stage with a cross-validated R² of 0.577, supporting the utility of
DNA methylation as a molecular correlate of AD neuropathology. The PySpark-
based implementation provides a scalable framework readily applicable to larger
epigenomic datasets.

---

## References

1. Lunnon K, et al. Methylomic profiling implicates cortical deregulation of
   ANK1 in Alzheimer's disease. *Nature Neuroscience* 17, 1156–1163 (2014).

2. De Jager PL, et al. Alzheimer's disease: early alterations in brain DNA
   methylation at ANK1, BIN1, RHBDF2 and other loci. *Nature Neuroscience*
   17, 1156–1163 (2014).

3. Smith RG, et al. A meta-analysis of epigenome-wide association studies in
   Alzheimer's disease highlights novel differentially methylated loci across
   cortex. *Nature Communications* 12, 3331 (2021).

4. Zhang W, et al. Epigenome-wide association study of Alzheimer's disease
   progression using the ADNI cohort. (Paper motivating this analysis.)