"""
01_download_data.py

Downloads GSE59685 methylation data from GEO using GEOparse.
Parses metadata from soft file, fetches beta matrix from GEO FTP.
"""

import GEOparse
import pandas as pd
import os
import urllib.request
import gzip
import shutil

# ── 1. Configuration ──────────────────────────────────────────────────────────
GEO_ID = "GSE59685"
DATA_DIR = "data"
TISSUE = "frontal cortex"
os.makedirs(DATA_DIR, exist_ok=True)

# ── 2. Load GSE from local soft file ─────────────────────────────────────────
print(f"Loading {GEO_ID} from local soft file...")
gse = GEOparse.get_GEO(geo=GEO_ID, destdir=DATA_DIR, silent=True)
print(f"Total samples: {len(gse.gsms)}")

# ── 3. Parse characteristics into clean metadata ──────────────────────────────
print("Parsing metadata...")

def parse_characteristics(char_list):
    """Parse the characteristics_ch1 list into a dict."""
    result = {}
    for item in char_list:
        if ": " in item:
            key, val = item.split(": ", 1)
            result[key.strip()] = val.strip()
    return result

records = []
for gsm_name, gsm in gse.gsms.items():
    char_list = gsm.metadata.get("characteristics_ch1", [])
    parsed = parse_characteristics(char_list)
    parsed["sample_id"] = gsm_name
    records.append(parsed)

metadata_df = pd.DataFrame(records)
print(f"Metadata shape: {metadata_df.shape}")
print(f"Columns: {metadata_df.columns.tolist()}")
print(f"\nTissue counts:\n{metadata_df['source tissue'].value_counts()}")
print(f"\nAD status counts:\n{metadata_df['ad.disease.status'].value_counts()}")
print(f"\nBraak stage counts:\n{metadata_df['braak.stage'].value_counts().sort_index()}")

# ── 4. Filter to frontal cortex, exclude bad samples ─────────────────────────
print(f"\nFiltering to tissue: {TISSUE}")
mask = (
    (metadata_df["source tissue"] == TISSUE) &
    (metadata_df["ad.disease.status"] != "Exclude") &
    (metadata_df["braak.stage"] != "Exclude")
)
metadata_clean = metadata_df[mask].copy()
metadata_clean["braak.stage"] = metadata_clean["braak.stage"].astype(float)
print(f"Samples after filtering: {len(metadata_clean)}")
print(f"Braak stage distribution:\n{metadata_clean['braak.stage'].value_counts().sort_index()}")

metadata_path = os.path.join(DATA_DIR, "metadata_clean.csv")
metadata_clean.to_csv(metadata_path, index=False)
print(f"Clean metadata saved to {metadata_path}")

# ── 5. Download beta matrix from GEO FTP ─────────────────────────────────────
# GSE59685 hosts its matrix file on the GEO FTP server
MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE59nnn/GSE59685/suppl/"
    "GSE59685_betas.csv.gz"
)
matrix_gz_path = os.path.join(DATA_DIR, "GSE59685_betas.csv.gz")
matrix_csv_path = os.path.join(DATA_DIR, "beta_matrix_full.csv")

if not os.path.exists(matrix_csv_path):
    print(f"\nDownloading beta matrix from GEO FTP...")
    print("This file is large (~500MB) and may take several minutes.")
    urllib.request.urlretrieve(MATRIX_URL, matrix_gz_path)
    print("Download complete. Decompressing...")
    with gzip.open(matrix_gz_path, 'rb') as f_in:
        with open(matrix_csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(matrix_gz_path)
    print(f"Beta matrix saved to {matrix_csv_path}")
else:
    print(f"\nBeta matrix already exists at {matrix_csv_path}, skipping download.")

# ── 6. Preview beta matrix ────────────────────────────────────────────────────
print("\nPreviewing beta matrix (first 5 rows, 5 cols)...")
beta_preview = pd.read_csv(matrix_csv_path, index_col=0, nrows=5, skiprows=4)
print(beta_preview.iloc[:, :5])
print(f"\nRows (CpG sites): {beta_preview.shape[0]}")
print(f"Columns (samples): {len(beta_preview.columns)}")
print(f"\nFirst few column names (barcodes):")
print(beta_preview.columns[:5].tolist())

# ── 7. Confirm barcode overlap with metadata ──────────────────────────────────
print("\nChecking barcode overlap with filtered metadata...")
metadata_clean = pd.read_csv(os.path.join(DATA_DIR, "metadata_clean.csv"))
matrix_barcodes = set(beta_preview.columns.tolist())
metadata_barcodes = set(metadata_clean["barcode"].tolist())
overlap = matrix_barcodes & metadata_barcodes
print(f"Barcodes in beta matrix: {len(matrix_barcodes)}")
print(f"Barcodes in filtered metadata: {len(metadata_barcodes)}")
print(f"Overlapping barcodes: {len(overlap)}")