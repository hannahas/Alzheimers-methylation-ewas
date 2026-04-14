"""
03_ewas_spark.py

Runs epigenome-wide association study (EWAS) using PySpark.
For each CpG site, fits linear regression: beta ~ braak_stage
Extracts slope, p-value, and applies Bonferroni correction.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# ── 1. Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
INPUT_PATH = os.path.join(DATA_DIR, "beta_frontal_clean.parquet")
OUTPUT_PATH = os.path.join(DATA_DIR, "ewas_results.parquet")

N_CPGS = 145089
BONFERRONI_THRESHOLD = 0.05 / N_CPGS
print(f"Bonferroni threshold: {BONFERRONI_THRESHOLD:.2e}")

# ── 2. Start Spark session ────────────────────────────────────────────────────
print("Starting Spark session...")
spark = SparkSession.builder \
    .appName("EWAS_Regression") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("Spark session started.")

# ── 3. Load Parquet ───────────────────────────────────────────────────────────
print(f"Loading data from {INPUT_PATH}...")
df = spark.read.parquet(INPUT_PATH)
print(f"Rows: {df.count():,}")
df.printSchema()

# ── 4. Define Pandas UDF for per-CpG linear regression ───────────────────────
result_schema = StructType([
    StructField("cpg",       StringType(), True),
    StructField("slope",     DoubleType(), True),
    StructField("intercept", DoubleType(), True),
    StructField("pvalue",    DoubleType(), True),
    StructField("r_squared", DoubleType(), True),
    StructField("n_samples", IntegerType(), True),
])

@F.pandas_udf(result_schema, F.PandasUDFType.GROUPED_MAP)
def run_linear_regression(group_df):
    """
    For one CpG group, fit: beta ~ braak_stage
    Returns slope, intercept, p-value, r-squared, n_samples.
    """
    cpg_id = group_df["cpg"].iloc[0]

    # Drop missing values
    clean = group_df[["beta", "braak_stage"]].dropna()
    n = len(clean)

    # Need at least 3 samples to fit a regression
    if n < 3:
        return pd.DataFrame([{
            "cpg": cpg_id, "slope": np.nan, "intercept": np.nan,
            "pvalue": np.nan, "r_squared": np.nan, "n_samples": n
        }])

    x = clean["braak_stage"].values
    y = clean["beta"].values

    slope, intercept, r, pvalue, _ = stats.linregress(x, y)

    return pd.DataFrame([{
        "cpg":       cpg_id,
        "slope":     float(slope),
        "intercept": float(intercept),
        "pvalue":    float(pvalue),
        "r_squared": float(r ** 2),
        "n_samples": int(n)
    }])

# ── 5. Run EWAS ───────────────────────────────────────────────────────────────
print("Running EWAS — fitting regression for each CpG...")
print("This is the slow step — may take 10-20 minutes on a laptop.")

results_df = df.groupby("cpg").apply(run_linear_regression)

# ── 6. Apply Bonferroni correction ────────────────────────────────────────────
print("Applying Bonferroni correction...")
results_df = results_df.withColumn(
    "significant",
    F.col("pvalue") < BONFERRONI_THRESHOLD
)

# ── 7. Save results ───────────────────────────────────────────────────────────
print(f"Saving results to {OUTPUT_PATH}...")
results_df.write.mode("overwrite").parquet(OUTPUT_PATH)
print("Saved.")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n=== EWAS Summary ===")
total = results_df.count()
sig = results_df.filter(F.col("significant") == True).count()
print(f"Total CpGs tested: {total:,}")
print(f"Significant after Bonferroni (p < {BONFERRONI_THRESHOLD:.2e}): {sig}")

print("\nTop 20 most significant CpGs:")
results_df.orderBy("pvalue").show(20, truncate=False)

spark.stop()