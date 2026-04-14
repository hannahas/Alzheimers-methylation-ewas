"""
05_visualize.py

Generates key EWAS visualizations:
1. Manhattan plot
2. Volcano plot
3. Top CpGs heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import GEOparse

# ── 1. Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
FIGURES_DIR = "figures"
N_CPGS = 145089
BONFERRONI_THRESHOLD = 0.05 / N_CPGS
TOP_N = 20

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 2. Load EWAS results ──────────────────────────────────────────────────────
print("Loading EWAS results...")
ewas = pd.read_parquet(os.path.join(DATA_DIR, "ewas_results.parquet"))
ewas = ewas.dropna(subset=["pvalue", "slope"])
ewas["-log10p"] = -np.log10(ewas["pvalue"])
print(f"CpGs loaded: {len(ewas):,}")

# ── 3. Load chromosome annotations from platform file ────────────────────────
print("Loading chromosome annotations from GEO platform...")
gse = GEOparse.get_GEO(geo="GSE59685", destdir=DATA_DIR, silent=True)
platform = list(gse.gpls.values())[0]
annot = platform.table[["ID", "CHR", "MAPINFO"]].copy()
annot.columns = ["cpg", "chr", "position"]
annot = annot.dropna(subset=["chr", "position"])
annot["position"] = pd.to_numeric(annot["position"], errors="coerce")
annot["chr"] = annot["chr"].astype(str).str.strip()

# Remove non-standard chromosomes
valid_chrs = [str(i) for i in range(1, 23)] + ["X", "Y"]
annot = annot[annot["chr"].isin(valid_chrs)]
print(f"CpGs with chromosome annotations: {len(annot):,}")

# Merge with EWAS results
ewas = ewas.merge(annot, on="cpg", how="inner")
print(f"CpGs after merging with annotations: {len(ewas):,}")

# ── 4. Manhattan plot ─────────────────────────────────────────────────────────
print("Generating Manhattan plot...")

# Sort chromosomes numerically
chr_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
ewas["chr"] = pd.Categorical(ewas["chr"], categories=chr_order, ordered=True)
ewas = ewas.sort_values(["chr", "position"])

# Compute cumulative x-position across chromosomes
ewas["cum_pos"] = 0
chr_offsets = {}
offset = 0
for chr_name in chr_order:
    chr_data = ewas[ewas["chr"] == chr_name]
    if len(chr_data) == 0:
        continue
    chr_offsets[chr_name] = offset + (chr_data["position"].max() - chr_data["position"].min()) / 2
    ewas.loc[ewas["chr"] == chr_name, "cum_pos"] = chr_data["position"] - chr_data["position"].min() + offset
    offset += chr_data["position"].max() - chr_data["position"].min() + 5e7

fig, ax = plt.subplots(figsize=(16, 5))

colors = ["steelblue", "lightsteelblue"]
for i, chr_name in enumerate(chr_order):
    chr_data = ewas[ewas["chr"] == chr_name]
    if len(chr_data) == 0:
        continue
    ax.scatter(
        chr_data["cum_pos"],
        chr_data["-log10p"],
        c=colors[i % 2],
        s=4,
        alpha=0.7,
        rasterized=True
    )

# Bonferroni line
bonf_line = -np.log10(BONFERRONI_THRESHOLD)
ax.axhline(bonf_line, color="crimson", linestyle="--", linewidth=1.2,
           label=f"Bonferroni (p={BONFERRONI_THRESHOLD:.1e})")

# Label significant hits
sig = ewas[ewas["pvalue"] < BONFERRONI_THRESHOLD]
for _, row in sig.iterrows():
    ax.annotate(
        row["cpg"],
        xy=(row["cum_pos"], row["-log10p"]),
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        color="crimson"
    )

# X-axis chromosome labels
ax.set_xticks(list(chr_offsets.values()))
ax.set_xticklabels(list(chr_offsets.keys()), fontsize=7)
ax.set_xlabel("Chromosome", fontsize=12)
ax.set_ylabel("-log10(p-value)", fontsize=12)
ax.set_title("Manhattan Plot — EWAS of Braak Stage (Frontal Cortex)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "manhattan_plot.png"), dpi=150)
plt.show()
print("Manhattan plot saved.")

# ── 5. Volcano plot ───────────────────────────────────────────────────────────
print("Generating volcano plot...")

fig, ax = plt.subplots(figsize=(9, 6))

# Color points by significance and direction
def get_color(row):
    if row["pvalue"] >= BONFERRONI_THRESHOLD:
        return "lightgray"
    elif row["slope"] < 0:
        return "steelblue"
    else:
        return "crimson"

colors = ewas.apply(get_color, axis=1)

ax.scatter(ewas["slope"], ewas["-log10p"], c=colors, s=4, alpha=0.6, rasterized=True)
ax.axhline(bonf_line, color="black", linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

# Label significant hits
for _, row in sig.iterrows():
    ax.annotate(
        row["cpg"],
        xy=(row["slope"], row["-log10p"]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=8,
        color="black"
    )

# Legend
gray_patch = mpatches.Patch(color="lightgray", label="Not significant")
blue_patch = mpatches.Patch(color="steelblue", label="Significant — hypomethylated")
red_patch = mpatches.Patch(color="crimson", label="Significant — hypermethylated")
ax.legend(handles=[gray_patch, blue_patch, red_patch], fontsize=9)

ax.set_xlabel("Slope (effect size)", fontsize=12)
ax.set_ylabel("-log10(p-value)", fontsize=12)
ax.set_title("Volcano Plot — EWAS of Braak Stage", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "volcano_plot.png"), dpi=150)
plt.show()
print("Volcano plot saved.")

# ── 6. Top CpGs heatmap ───────────────────────────────────────────────────────
print("Generating top CpGs heatmap...")

beta_long = pd.read_parquet(os.path.join(DATA_DIR, "beta_frontal_clean.parquet"))
top_cpgs_df = ewas.sort_values("pvalue").head(TOP_N)
top_cpgs = top_cpgs_df["cpg"].tolist()
sig_cpgs = set(ewas[ewas["pvalue"] < BONFERRONI_THRESHOLD]["cpg"].tolist())

beta_top = beta_long[beta_long["cpg"].isin(top_cpgs)]
beta_wide = beta_top.pivot_table(index="barcode", columns="cpg", values="beta")

# Get Braak stage per sample and sort
braak = beta_long[["barcode", "braak_stage"]].drop_duplicates().set_index("barcode")
beta_wide = beta_wide.join(braak).sort_values("braak_stage")
braak_sorted = beta_wide["braak_stage"]
beta_wide = beta_wide.drop(columns="braak_stage")

# Reorder columns by p-value
beta_wide = beta_wide[top_cpgs]

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    beta_wide.T,
    cmap="RdBu_r",
    center=0.5,
    ax=ax,
    xticklabels=False,
    yticklabels=True,
    cbar_kws={"label": "Beta (methylation)", "shrink": 0.8}
)

# Fix y-axis labels — add asterisk for significant CpGs
yticklabels = []
for cpg in top_cpgs:
    if cpg in sig_cpgs:
        yticklabels.append(f"{cpg}  ✱")
    else:
        yticklabels.append(cpg)

ax.set_yticklabels(
    yticklabels,
    fontsize=9,
    rotation=0,      # horizontal labels — no overlap
    ha="right"
)

# Add Braak stage color bar along x-axis
braak_colors = plt.cm.viridis(braak_sorted.values / 6)
for i, color in enumerate(braak_colors):
    ax.add_patch(plt.Rectangle(
        (i, -1), 1, 0.5,
        color=color,
        transform=ax.get_xaxis_transform(),
        clip_on=False
    ))

# Add Braak stage color bar legend
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=6))
sm.set_array([])
cbar2 = plt.colorbar(sm, ax=ax, orientation="horizontal",
                     fraction=0.02, pad=0.12, aspect=40)
cbar2.set_label("Braak Stage", fontsize=10)

ax.set_xlabel("Samples (ordered by Braak stage →)", fontsize=11, labelpad=30)
ax.set_ylabel("CpG Site", fontsize=11)
ax.set_title(
    f"Top {TOP_N} CpGs by p-value — Methylation Across Samples\n✱ = Bonferroni significant",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "top_cpgs_heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Heatmap saved.")