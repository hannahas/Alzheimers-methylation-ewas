Alzheimer's Methylation EWAS — Project Progress
Status: Scripts 01 and 02 complete ✅

What We're Building
An epigenome-wide association study (EWAS) of DNA methylation in Alzheimer's disease using public brain tissue data. We're testing whether methylation levels at 145,089 CpG sites across the genome are associated with Braak stage (a continuous 0–6 measure of AD pathology severity) in frontal cortex tissue.

Dataset

GEO Accession: GSE59685
Array: Illumina HumanMethylation 450k
Tissue: Frontal cortex
Samples: 80 (after filtering Excludes)
CpGs after QC filtering: 145,089
Outcome variable: Braak stage (continuous, 0–6)


Completed
Script 01 — scripts/01_download_data.py ✅

Downloads GSE59685 soft file via GEOparse
Parses sample metadata from characteristics field
Filters to frontal cortex samples, removes Excluded samples
Downloads full beta matrix from GEO FTP (~500MB)
Outputs: data/metadata_clean.csv, data/beta_matrix_full.csv

Script 02 — scripts/02_preprocess_spark.py ✅

Loads full beta matrix (485,577 CpGs × 531 samples)
Filters to 80 frontal cortex barcodes
Removes low-variance CpGs (var < 0.001) and high-missingness CpGs (>5%)
Converts to long format (cpg, barcode, beta, braak_stage)
Loads into PySpark and saves as Parquet
Outputs: data/beta_frontal_clean.parquet (11,607,120 rows)

Exploratory Notebook — notebooks/ ✅

Metadata preview
Beta matrix preview
Braak stage distribution bar chart
Full beta matrix shape and memory estimate


Up Next
Script 03 — EWAS in PySpark

Load Parquet file into Spark
groupBy("cpg") to parallelize regression across all 145,089 CpGs
Fit linear regression: beta ~ braak_stage per CpG
Extract slope (effect size) and p-value for each CpG
Apply Bonferroni correction (threshold: p < 0.05 / 145,089 ≈ 3.4 × 10⁻⁷)
Save results table: data/ewas_results.parquet

Script 04 — scikit-learn Model

Take top significant CpGs from Script 03
Build ridge regression model predicting Braak stage from CpG methylation
Cross-validate and report performance

Script 05 — Visualization

Manhattan plot (chromosome position vs. -log10 p-value)
Volcano plot (effect size vs. -log10 p-value)
Save figures to figures/

Script 06 — Writeup

Markdown report summarizing methods, results, and interpretation
Annotate top CpGs with gene names
Compare findings to published literature (ANK1, HOXA3, etc.)


Environment Notes

Python 3.14, PySpark 4.1.1
Java 17 required (not Java 11) — set in ~/.zshrc
Always activate venv first: source venv/bin/activate
Data files are in data/ which is gitignored — do not delete locally