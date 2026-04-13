"""
02_preprocess_spark.py

Loads beta matrix in native orientation (CpGs x samples),
filters to frontal cortex samples and low-quality CpGs,
saves clean Parquet file in long format for EWAS.
"""

import pandas as pd
import numpy as np
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── 1. Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
BETA_PATH = os.path.join(DATA_DIR, "beta_matrix_full.csv")
METADATA_PATH = os.path.join(DATA_DIR, "metadata_clean.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "beta_frontal_clean.parquet")

MAX_MISSING_RATE = 0.05
MIN_VARIANCE = 0.001

# ── 2. Start Spark session ────────────────────────────────────────────────────
print("Starting Spark session...")
spark = SparkSession.builder \
    .appName("EWAS_Preprocessing") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.debug.maxToStringFields", "100") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("Spark session started.")

# ── 3. Load metadata ──────────────────────────────────────────────────────────
print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH)
fc_barcodes = metadata["barcode"].tolist()
barcode_to_braak = dict(zip(metadata["barcode"], metadata["braak.stage"]))
print(f"Frontal cortex samples: {len(fc_barcodes)}")

# ── 4. Load beta matrix, filter to frontal cortex columns ────────────────────
print("Loading beta matrix...")
with open(BETA_PATH, 'r') as f:
    for i, line in enumerate(f):
        if i == 4:
            col_names = ['cpg'] + line.strip().split(',')[1:]
            break

# Read column names from line 4
with open(BETA_PATH, 'r') as f:
    for i, line in enumerate(f):
        if i == 4:
            col_names = line.strip().split(',')
            col_names[0] = 'cpg'  # rename first column
            break

# Read data starting from line 7 (0-indexed), no header
beta_pd = pd.read_csv(
    BETA_PATH,
    skiprows=7,
    header=None,
    names=col_names,
    index_col='cpg',
    low_memory=False,
    na_values=['NA', 'NaN', 'null', '']
)

# Convert to float explicitly after load
beta_pd = beta_pd.apply(pd.to_numeric, errors='coerce')

print("Dtypes sample:")
print(beta_pd.dtypes.value_counts())
print("\nFirst 3 rows, first 3 cols:")
print(beta_pd.iloc[:3, :3])

available = [b for b in fc_barcodes if b in beta_pd.columns]
beta_fc = beta_pd[available].copy()
print(f"Filtered to {len(available)} frontal cortex samples: {beta_fc.shape}")

# ── 5. Filter CpGs using pandas (column-wise stats) ──────────────────────────
print("Filtering low-quality CpGs...")
missing_rates = beta_fc.isnull().mean(axis=1)
variances = beta_fc.var(axis=1)

keep_mask = (missing_rates <= MAX_MISSING_RATE) & (variances >= MIN_VARIANCE)
beta_filtered = beta_fc[keep_mask].copy()
print(f"CpGs before filtering: {len(beta_fc)}")
print(f"CpGs after filtering:  {len(beta_filtered)}")

# ── 6. Convert to long format ─────────────────────────────────────────────────
# Long format: one row per (CpG, sample) pair
# This is the natural format for Spark and for per-CpG regression
print("Converting to long format (cpg, barcode, beta, braak_stage)...")
beta_filtered.index.name = "cpg"
beta_long = beta_filtered.reset_index().melt(
    id_vars="cpg",
    var_name="barcode",
    value_name="beta"
)
beta_long["braak_stage"] = beta_long["barcode"].map(barcode_to_braak)
print(f"Long format shape: {beta_long.shape}")
print(f"Expected: ~{len(beta_filtered) * len(available):,} rows")
print(beta_long.head())

# ── 7. Load into Spark and save as Parquet ────────────────────────────────────
print("\nLoading into Spark...")
spark_df = spark.createDataFrame(beta_long)
print(f"Spark DataFrame: {spark_df.count():,} rows, {len(spark_df.columns)} columns")
spark_df.printSchema()

print(f"\nSaving to {OUTPUT_PATH}...")
spark_df.write.mode("overwrite").parquet(OUTPUT_PATH)
print("Done! Parquet file saved.")

spark.stop()